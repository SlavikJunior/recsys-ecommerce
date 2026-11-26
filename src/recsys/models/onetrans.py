"""OneTrans: единый causal S/NS stack, mixed параметры и pyramid queries.

Математические блоки соответствуют arXiv:2510.26104v3 §3. Это уменьшенный
ranker с собственным next-item head, а не генеративная модель OneRec.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class OneTransConfig:
    width: int = 32
    heads: int = 4
    ns_tokens: int = 3
    history_length: int = 32
    pyramid: tuple[int, ...] = (8, 0)
    expansion: int = 2

    def __post_init__(self) -> None:
        if self.width < 4 or self.heads < 1 or self.width % self.heads:
            raise ValueError("width должен делиться на положительное heads")
        if self.ns_tokens < 1 or self.history_length < 1 or self.expansion < 1:
            raise ValueError("размеры токенов должны быть положительными")
        if not self.pyramid or any(v < 0 for v in self.pyramid):
            raise ValueError("pyramid должен содержать неотрицательные длины")
        if any(b > a for a, b in zip(self.pyramid, self.pyramid[1:], strict=False)):
            raise ValueError("pyramid не может увеличивать число S queries")

    def payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class PrefixLayer:
    keys: Tensor
    values: Tensor
    valid: Tensor


@dataclass
class PrefixCache:
    layers: list[PrefixLayer]
    batch_size: int

    def repeat_candidates(self, candidates: int) -> PrefixCache:
        return PrefixCache(
            [
                PrefixLayer(
                    layer.keys.repeat_interleave(candidates, dim=0),
                    layer.values.repeat_interleave(candidates, dim=0),
                    layer.valid.repeat_interleave(candidates, dim=0),
                )
                for layer in self.layers
            ],
            self.batch_size * candidates,
        )


class MixedProjection(nn.Module):
    def __init__(self, width: int, output: int, ns_tokens: int) -> None:
        super().__init__()
        self.shared = nn.Linear(width, output, bias=False)
        self.specific = nn.ModuleList(
            [nn.Linear(width, output, bias=False) for _ in range(ns_tokens)]
        )

    def sequential(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.shared(x))

    def nonsequential(self, x: Tensor) -> Tensor:
        return torch.stack([layer(x[:, i]) for i, layer in enumerate(self.specific)], dim=1)


class MixedFFN(nn.Module):
    def __init__(self, config: OneTransConfig) -> None:
        super().__init__()
        width = config.width

        def ffn() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(width, width * config.expansion),
                nn.GELU(),
                nn.Linear(width * config.expansion, width),
            )

        self.shared = ffn()
        self.specific = nn.ModuleList([ffn() for _ in range(config.ns_tokens)])

    def sequential(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.shared(x))

    def nonsequential(self, x: Tensor) -> Tensor:
        return torch.stack([layer(x[:, i]) for i, layer in enumerate(self.specific)], dim=1)


class OneTransBlock(nn.Module):
    def __init__(self, config: OneTransConfig) -> None:
        super().__init__()
        self.width = config.width
        self.heads = config.heads
        self.ns_tokens = config.ns_tokens
        self.attn_norm = nn.RMSNorm(config.width)
        self.ffn_norm = nn.RMSNorm(config.width)
        self.qkv = MixedProjection(config.width, config.width * 3, config.ns_tokens)
        self.output = nn.Linear(config.width, config.width, bias=False)
        self.ffn = MixedFFN(config)

    def split(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return tuple(
            part.reshape(x.shape[0], x.shape[1], self.heads, self.width // self.heads).transpose(
                1, 2
            )
            for part in x.chunk(3, dim=-1)
        )  # type: ignore[return-value]

    def attend(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor) -> Tensor:
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None])
        return cast(
            Tensor, self.output(out.transpose(1, 2).reshape(q.shape[0], q.shape[2], self.width))
        )

    def prefix(
        self,
        x: Tensor,
        valid: Tensor,
        keep: int,
        reusable: PrefixLayer | None = None,
        reuse_length: int = 0,
    ) -> tuple[Tensor, Tensor, PrefixLayer]:
        length = x.shape[1]
        keep = min(keep, length)
        normalized = self.attn_norm(x)
        if reusable is not None and reuse_length:
            # При append неизменные causal префиксы сохраняют K/V первого слоя.
            q = F.linear(normalized, self.qkv.shared.weight[: self.width])
            q = q.reshape(x.shape[0], length, self.heads, self.width // self.heads).transpose(1, 2)
            _, new_k, new_v = self.split(self.qkv.sequential(normalized[:, reuse_length:]))
            k = torch.cat([reusable.keys[:, :, :reuse_length], new_k], dim=2)
            v = torch.cat([reusable.values[:, :, :reuse_length], new_v], dim=2)
        else:
            q, k, v = self.split(self.qkv.sequential(normalized))
        cache = PrefixLayer(k, v, valid)
        selected = torch.arange(length - keep, length, device=x.device)
        causal = selected[:, None] >= torch.arange(length, device=x.device)[None, :]
        mask = causal[None] & valid[:, None, :]
        out = x[:, selected] + self.attend(q[:, :, selected], k, v, mask)
        out = out + self.ffn.sequential(self.ffn_norm(out))
        selected_valid = valid[:, selected]
        return out * selected_valid[:, :, None], selected_valid, cache

    def candidates(self, x: Tensor, prefix: PrefixLayer) -> Tensor:
        q, ns_k, ns_v = self.split(self.qkv.nonsequential(self.attn_norm(x)))
        k = torch.cat([prefix.keys, ns_k], dim=2)
        v = torch.cat([prefix.values, ns_v], dim=2)
        ns_mask = torch.ones(self.ns_tokens, self.ns_tokens, dtype=torch.bool, device=x.device)
        ns_mask = ns_mask.tril()[None].expand(x.shape[0], -1, -1)
        mask = torch.cat([prefix.valid[:, None].expand(-1, self.ns_tokens, -1), ns_mask], dim=2)
        out = x + self.attend(q, k, v, mask)
        return out + self.ffn.nonsequential(self.ffn_norm(out))

    def forward(self, x: Tensor, valid: Tensor, keep: int) -> tuple[Tensor, Tensor]:
        """Независимый uncached reference: общий attention S+NS до slicing."""
        length = x.shape[1] - self.ns_tokens
        keep = min(keep, length)
        norm = self.attn_norm(x)
        qkv = torch.cat(
            [self.qkv.sequential(norm[:, :length]), self.qkv.nonsequential(norm[:, length:])],
            dim=1,
        )
        q, k, v = self.split(qkv)
        selected = torch.arange(length - keep, x.shape[1], device=x.device)
        key_valid = torch.cat(
            [valid, torch.ones(x.shape[0], self.ns_tokens, device=x.device, dtype=torch.bool)],
            dim=1,
        )
        causal = selected[:, None] >= torch.arange(x.shape[1], device=x.device)[None, :]
        out = x[:, selected] + self.attend(
            q[:, :, selected], k, v, causal[None] & key_valid[:, None]
        )
        norm = self.ffn_norm(out)
        out = out + torch.cat(
            [self.ffn.sequential(norm[:, :keep]), self.ffn.nonsequential(norm[:, keep:])],
            dim=1,
        )
        retained = valid[:, length - keep :]
        out = out * torch.cat([retained, key_valid[:, length:]], dim=1)[:, :, None]
        return out, retained


class OneTrans(nn.Module):
    item_features: Tensor

    def __init__(
        self,
        config: OneTransConfig,
        item_features: Tensor,
        feature_sizes: tuple[int, int, int],
        n_users: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("item_features", item_features.to(torch.long))
        self.fields = nn.ModuleList(
            [nn.Embedding(size, config.width, padding_idx=0) for size in feature_sizes]
        )
        self.users = nn.Embedding(n_users + 1, config.width, padding_idx=0)
        self.sequence_projection = nn.Linear(config.width * 3, config.width)
        self.type_embedding = nn.Embedding(2, config.width, padding_idx=0)
        self.auto_split = nn.Sequential(
            nn.Linear(config.width * 4 + 1, config.width * config.ns_tokens),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([OneTransBlock(config) for _ in config.pyramid])
        self.final_norm = nn.RMSNorm(config.width)
        self.head = nn.Linear(config.width * config.ns_tokens, 1)

    def item_embeddings(self, catalog_ids: Tensor) -> Tensor:
        features = self.item_features[catalog_ids + 1]
        return torch.cat([field(features[..., i]) for i, field in enumerate(self.fields)], dim=-1)

    def sequential_tokens(self, history: Tensor) -> tuple[Tensor, Tensor]:
        valid = history >= 0
        x = self.sequence_projection(self.item_embeddings(history))
        x = x + self.type_embedding(valid.to(torch.long))
        return x * valid[:, :, None], valid

    def nonsequential_tokens(self, users: Tensor, candidates: Tensor, lengths: Tensor) -> Tensor:
        features = torch.cat(
            [self.users(users + 1), self.item_embeddings(candidates), lengths[:, None].log1p()],
            dim=-1,
        )
        return cast(Tensor, self.auto_split(features)).reshape(
            -1, self.config.ns_tokens, self.config.width
        )

    def encode_prefix(
        self,
        history: Tensor,
        reusable: PrefixCache | None = None,
        reuse_length: int = 0,
    ) -> PrefixCache:
        x, valid = self.sequential_tokens(history)
        layers = []
        for index, (raw, keep) in enumerate(zip(self.blocks, self.config.pyramid, strict=True)):
            block = cast(OneTransBlock, raw)
            old = reusable.layers[0] if reusable is not None and index == 0 else None
            x, valid, cache = block.prefix(x, valid, keep, old, reuse_length if index == 0 else 0)
            layers.append(cache)
        return PrefixCache(layers, history.shape[0])

    def score_cached(
        self, cache: PrefixCache, users: Tensor, candidates: Tensor, lengths: Tensor
    ) -> Tensor:
        if candidates.ndim != 2 or candidates.shape[0] != cache.batch_size:
            raise ValueError("candidates должны иметь форму [prefix batch, candidates]")
        count = candidates.shape[1]
        expanded = cache.repeat_candidates(count)
        ns = self.nonsequential_tokens(
            users.repeat_interleave(count), candidates.flatten(), lengths.repeat_interleave(count)
        )
        for raw, layer in zip(self.blocks, expanded.layers, strict=True):
            block = cast(OneTransBlock, raw)
            ns = block.candidates(ns, layer)
        return cast(Tensor, self.head(self.final_norm(ns).flatten(1))).reshape(candidates.shape)

    def forward(self, history: Tensor, users: Tensor, candidates: Tensor) -> Tensor:
        cache = self.encode_prefix(history)
        return self.score_cached(cache, users, candidates, (history >= 0).sum(dim=1).float())

    def uncached(self, history: Tensor, users: Tensor, candidates: Tensor) -> Tensor:
        if candidates.ndim != 1:
            raise ValueError("reference принимает одну candidate на history")
        x, valid = self.sequential_tokens(history)
        ns = self.nonsequential_tokens(users, candidates, valid.sum(dim=1).float())
        x = torch.cat([x, ns], dim=1)
        for raw, keep in zip(self.blocks, self.config.pyramid, strict=True):
            block = cast(OneTransBlock, raw)
            x, valid = block(x, valid, keep)
        return cast(
            Tensor, self.head(self.final_norm(x[:, -self.config.ns_tokens :]).flatten(1))
        ).squeeze(-1)
