"""Two-Tower retrieval: современная модель для «модной» части задания.

Почему two-tower, а не GNN/LLM-ранкер:
- обучается на ноутбуке (MPS) за минуты и честно измерим;
- отражает продовый паттерн retrieval + ANN, то есть объясняет, как модель
  живёт в сервисе (эмбеддинги считаются заранее, онлайн — только ANN-поиск);
- легко объяснить бизнесу: «пользовательский вектор встречается с товарным».

Архитектура: user tower (id + активность), item tower (id + контентные
признаки из TF-IDF-SVD + цена), обучение in-batch softmax с температурой.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from recsys.config import ARTIFACTS_DIR
from recsys.data.dataset import InteractionData, load_dataset
from recsys.models.base import BaseRecommender
from recsys.models.content import build_item_text


class TwoTowerNet(nn.Module):
    def __init__(self, n_users: int, n_items: int, n_content: int, dim: int = 64) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, dim)
        self.item_embedding = nn.Embedding(n_items, dim)
        self.content_projection = nn.Linear(n_content, dim)
        self.item_mlp = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.user_mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def user_tower(self, user_idx: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.user_mlp(self.user_embedding(user_idx)), dim=-1)

    def item_tower(self, item_idx: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
        z = torch.cat([self.item_embedding(item_idx), self.content_projection(content)], dim=-1)
        return F.normalize(self.item_mlp(z), dim=-1)


def build_content_features(data: InteractionData, n_components: int = 64) -> np.ndarray:
    """TF-IDF -> SVD: компактные контентные признаки для item tower."""
    vectorizer = TfidfVectorizer(
        max_features=50_000, min_df=3, ngram_range=(1, 2), sublinear_tf=True
    )
    tfidf = normalize(vectorizer.fit_transform(build_item_text(data)))
    n_components = min(n_components, tfidf.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    dense = svd.fit_transform(tfidf).astype(np.float32)

    price = data.item_meta["price"].to_numpy(dtype=np.float32)
    price = np.log1p(
        np.nan_to_num(price, nan=np.nanmedian(price) if np.isfinite(price).any() else 0.0)
    )
    price = (price - price.mean()) / (price.std() + 1e-6)
    dense = np.hstack([dense, price[:, None]])
    return dense.astype(np.float32)


class TwoTowerRecommender(BaseRecommender):
    name = "two_tower"

    def __init__(
        self,
        dim: int = 64,
        content_dim: int = 64,
        epochs: int = 12,
        batch_size: int = 4096,
        lr: float = 1e-3,
        temperature: float = 0.07,
        device: str | None = None,
        checkpoint_dir: Path | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.content_dim = content_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.temperature = temperature
        self.device = torch.device(
            device or ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        self.checkpoint_dir = checkpoint_dir
        self.item_embeddings: np.ndarray | None = None
        self.user_embeddings: np.ndarray | None = None

    def _train(
        self, data: InteractionData, content: np.ndarray
    ) -> tuple[TwoTowerNet, np.ndarray, np.ndarray]:
        torch.manual_seed(42)
        net = TwoTowerNet(data.n_users, data.n_items, content.shape[1], self.dim).to(self.device)
        content_t = torch.from_numpy(content).to(self.device)

        rows, cols = data.train_csr.nonzero()
        users = torch.from_numpy(rows.astype(np.int64))
        items = torch.from_numpy(cols.astype(np.int64))
        n = len(users)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=1e-6)

        for epoch in range(self.epochs):
            permutation = torch.randperm(n)
            total_loss = 0.0
            steps = 0
            net.train()
            for start in range(0, n, self.batch_size):
                batch = permutation[start : start + self.batch_size]
                if len(batch) < 16:
                    continue
                u = users[batch].to(self.device)
                i = items[batch].to(self.device)
                user_vec = net.user_tower(u)
                item_vec = net.item_tower(i, content_t[i])
                logits = user_vec @ item_vec.T / self.temperature
                # In-batch softmax: позитив для строки r — её собственный товар.
                # В батче могут встретиться дубликаты товаров — это шум в
                # негативах, но при batch=4096 доля дубликатов мала.
                target = torch.arange(len(batch), device=self.device)
                loss = F.cross_entropy(logits, target)
                optimizer.zero_grad()
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
                total_loss += float(loss.item())
                steps += 1
            print(f"epoch {epoch + 1}/{self.epochs}: loss={total_loss / max(steps, 1):.4f}")

        net.eval()
        with torch.no_grad():
            all_items = torch.arange(data.n_items, device=self.device)
            item_vec = net.item_tower(all_items, content_t)
            all_users = torch.arange(data.n_users, device=self.device)
            user_vec = net.user_tower(all_users)
        return net, user_vec.cpu().numpy(), item_vec.cpu().numpy()

    def fit(self, data: InteractionData) -> TwoTowerRecommender:
        super().fit(data)
        if self.checkpoint_dir is not None and self._load_checkpoint(data):
            return self

        content = build_content_features(data, self.content_dim)
        started = time.perf_counter()
        _, self.user_embeddings, self.item_embeddings = self._train(data, content)
        print(f"обучение two-tower: {time.perf_counter() - started:.1f} c")

        if self.checkpoint_dir is not None:
            self._save_checkpoint(data)
        return self

    def _save_checkpoint(self, data: InteractionData) -> None:
        assert self.user_embeddings is not None and self.item_embeddings is not None
        path = self._checkpoint_path(data)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            user_embeddings=self.user_embeddings,
            item_embeddings=self.item_embeddings,
        )

    def _load_checkpoint(self, data: InteractionData) -> bool:
        assert self.checkpoint_dir is not None
        path = self._checkpoint_path(data)
        if not path.exists():
            return False
        arrays = np.load(path)
        item_embeddings = np.asarray(arrays["item_embeddings"])
        if item_embeddings.shape[0] != data.n_items:
            return False
        self.user_embeddings = np.asarray(arrays["user_embeddings"])
        self.item_embeddings = item_embeddings
        print(f"two-tower: загружен чекпоинт {path.name}")
        return True

    @staticmethod
    def _checkpoint_path(data: InteractionData) -> Path:
        category = data.stats.get("category", "unknown")
        return ARTIFACTS_DIR / "two_tower" / f"{category}.npz"

    def score_users(self, user_idx: np.ndarray) -> np.ndarray:
        if self.user_embeddings is None or self.item_embeddings is None:
            raise RuntimeError("сначала вызови fit()")
        # Эмбеддинги L2-нормированы, скалярное произведение = косинус.
        scores = self.user_embeddings[user_idx] @ self.item_embeddings.T
        return np.asarray(scores, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="Video_Games")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--no-checkpoint", action="store_true")
    args = parser.parse_args()

    data = load_dataset(args.category)
    model = TwoTowerRecommender(
        dim=args.dim,
        epochs=args.epochs,
        checkpoint_dir=None if args.no_checkpoint else ARTIFACTS_DIR / "two_tower",
    )
    model.fit(data)
    assert model.user_embeddings is not None and model.item_embeddings is not None
    print(
        f"user_embeddings={model.user_embeddings.shape} "
        f"item_embeddings={model.item_embeddings.shape}"
    )


if __name__ == "__main__":
    main()
