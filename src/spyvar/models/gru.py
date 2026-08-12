"""M5 —— 小型 GRU 分位数模型（sequence-model 稳健性扩展）。

输入：最近 L 个交易日的特征向量序列（与 MLP 相同的信息集，
F3 的逐日特征）；GRU(1 层) 末隐状态 -> 非交叉有序输出头。

回答：在结构化 HAR 特征已存在时，显式序列表示是否仍有增量。
序列窗口完全限制在训练窗口内；训练/验证/标准化协议与 M3 一致。
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..rolling import Model, QuantileForecast, WindowData
from .mlp import _ordered_head, _pinball


class _GRUModel(nn.Module):
    def __init__(self, d_in: int, hidden: int, layers: int):
        super().__init__()
        self.gru = nn.GRU(d_in, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return _ordered_head(self.head(out[:, -1, :]))


class GRUQuantile(Model):
    model_id = "M5"

    def __init__(
        self,
        hidden: int = 16,
        layers: int = 1,
        seq_len: int = 22,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 64,
        epochs: int = 300,
        patience: int = 30,
        val_fraction: float = 0.1,
        device: str = "cpu",
    ):
        self._hidden = int(hidden)
        self._layers = int(layers)
        self._seq_len = int(seq_len)
        self._lr = float(lr)
        self._wd = float(weight_decay)
        self._batch = int(batch_size)
        self._epochs = int(epochs)
        self._patience = int(patience)
        self._val_frac = float(val_fraction)
        self._device = "cuda:0" if device == "cuda" and torch.cuda.is_available() else "cpu"

    def fit(self, window: WindowData) -> QuantileForecast:
        if window.features is None:
            raise ValueError("M5 需要特征矩阵")
        torch.set_num_threads(1)
        torch.manual_seed(window.seed)
        L = self._seq_len
        feat = window.features.to_numpy(dtype=np.float32)
        locs = window.dates.get_indexer(window.features.index)
        valid = locs + 1 < len(window.returns)
        y_all = window.returns[locs[valid] + 1].astype(np.float32)
        valid_idx = np.where(valid)[0]
        if valid.sum() < L + 100:
            raise ValueError(f"训练样本过少: {int(valid.sum())}")
        seqs, yv = [], []
        for i in valid_idx:
            if i < L - 1:
                continue
            seqs.append(feat[i - L + 1 : i + 1])
            yv.append(y_all[i])
        X = np.stack(seqs)
        y = np.array(yv, dtype=np.float32)
        n_val = max(1, int(self._val_frac * len(y)))
        X_tr, X_va = X[:-n_val], X[-n_val:]
        y_tr, y_va = y[:-n_val], y[-n_val:]
        flat = X_tr.reshape(-1, X_tr.shape[-1])
        mu = flat.mean(axis=0, keepdims=True)
        sd = flat.std(axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        X_tr = (X_tr - mu) / sd
        X_va = (X_va - mu) / sd
        x_origin = (feat[-L:] - mu) / sd

        model = _GRUModel(X_tr.shape[-1], self._hidden, self._layers).to(self._device)
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
            q = model(torch.from_numpy(x_origin[None, ...]).to(self._device))[0].cpu().numpy()
        return QuantileForecast(
            q_001=float(q[0]),
            q_005=float(q[1]),
            q_010=float(q[2]),
            fit_status="ok",
            meta={
                "epochs_used": best_epoch + 1,
                "val_pinball": best_val,
                "n_train": len(y),
                "seq_len": L,
                "device": self._device,
            },
        )
