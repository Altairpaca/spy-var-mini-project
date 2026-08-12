"""M3 —— 多分位数 MLP（joint pinball loss，结构非交叉）。

输出头：q_01 = z1, q_05 = z1 + softplus(z2), q_10 = q_05 + softplus(z3)，
softplus 恒正保证 q_01 <= q_05 <= q_10 由构造成立。

训练协议（全部限制在窗口内）：
- 标准化：仅用训练子集拟合 StandardScaler（窗口最后 10% 行作
  early-stopping 验证集，不参与 scaler 拟合与梯度更新）；
- early stopping：验证集 joint pinball，patience 个 epoch 无改善即停，
  恢复最优参数；
- 每日全量重训；种子固定；单线程 torch 保证可复现。
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..rolling import Model, QuantileForecast, WindowData

_ALPHA = torch.tensor([0.01, 0.05, 0.10])


def _pinball(y: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    diff = y[:, None] - q
    return torch.mean(torch.maximum(_ALPHA[None, :] * diff, (_ALPHA[None, :] - 1.0) * diff))


def _ordered_head(z: torch.Tensor) -> torch.Tensor:
    z1, g1, g2 = z[:, 0:1], z[:, 1:2], z[:, 2:3]
    q05 = z1 + torch.nn.functional.softplus(g1)
    q10 = q05 + torch.nn.functional.softplus(g2)
    return torch.cat([z1, q05, q10], dim=1)


class _MLP(nn.Module):
    def __init__(self, d_in: int, widths: list[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = d_in
        for w in widths:
            layers.append(nn.Linear(prev, w))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = w
        layers.append(nn.Linear(prev, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _ordered_head(self.net(x))


class MultiQuantileMLP(Model):
    model_id = "M3"

    def __init__(
        self,
        hidden: list[int] | None = None,
        dropout: float = 0.0,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 128,
        epochs: int = 300,
        patience: int = 30,
        val_fraction: float = 0.1,
        device: str = "cpu",
    ):
        self._hidden = hidden or [32, 32]
        self._dropout = float(dropout)
        self._lr = float(lr)
        self._wd = float(weight_decay)
        self._batch = int(batch_size)
        self._epochs = int(epochs)
        self._patience = int(patience)
        self._val_frac = float(val_fraction)
        self._device = "cuda:0" if device == "cuda" and torch.cuda.is_available() else "cpu"

    def fit(self, window: WindowData) -> QuantileForecast:
        if window.features is None:
            raise ValueError("M3 需要特征矩阵")
        torch.set_num_threads(1)
        torch.manual_seed(window.seed)
        feat = window.features.to_numpy(dtype=np.float32)
        locs = window.dates.get_indexer(window.features.index)
        valid = locs + 1 < len(window.returns)
        yv = window.returns[locs[valid] + 1].astype(np.float32)
        X = feat[valid]
        if len(yv) < 100:
            raise ValueError(f"训练样本过少: {len(yv)}")
        n_val = max(1, int(self._val_frac * len(yv)))
        X_tr, X_va = X[:-n_val], X[-n_val:]
        y_tr, y_va = yv[:-n_val], yv[-n_val:]
        mu = X_tr.mean(axis=0, keepdims=True)
        sd = X_tr.std(axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        X_tr = (X_tr - mu) / sd
        X_va = (X_va - mu) / sd
        x_origin = (feat[-1:] - mu) / sd

        model = _MLP(X_tr.shape[1], self._hidden, self._dropout).to(self._device)
        opt = torch.optim.Adam(model.parameters(), lr=self._lr, weight_decay=self._wd)
        Xt = torch.from_numpy(X_tr).to(self._device)
        yt = torch.from_numpy(y_tr).to(self._device)
        Xv = torch.from_numpy(X_va).to(self._device)
        yva = torch.from_numpy(y_va).to(self._device)
        n = len(Xt)
        best_val, best_state, best_epoch, no_improve = float("inf"), None, 0, 0
        for epoch in range(self._epochs):
            model.train()
            perm = torch.randperm(n)
            for i in range(0, n, self._batch):
                idx = perm[i : i + self._batch]
                opt.zero_grad()
                loss = _pinball(yt[idx], model(Xt[idx]))
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                val_loss = float(_pinball(yva, model(Xv)))
            if val_loss < best_val - 1e-9:
                best_val, best_epoch, no_improve = val_loss, epoch, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= self._patience:
                    break
        if best_state is None:
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            q = model(torch.from_numpy(x_origin).to(self._device))[0].cpu().numpy()
        return QuantileForecast(
            q_001=float(q[0]),
            q_005=float(q[1]),
            q_010=float(q[2]),
            fit_status="ok",
            meta={
                "epochs_used": best_epoch + 1,
                "val_pinball": best_val,
                "n_train": len(yv),
                "device": self._device,
            },
        )
