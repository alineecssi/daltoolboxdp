"""
Integrated adaptive-normalization LSTM forecaster (weights learned end-to-end).

Interface (compatible with other backends):
  - ts_lstm_create(n_neurons, look_back) -> torch.nn.Module
  - ts_lstm_fit(model, df_train, n_epochs, lr) -> model
  - ts_lstm_predict(model, df_test) -> np.ndarray of shape (n_samples,)

Key idea:
  - A learnable weighted moving average layer computes m(x) over lag columns
    using softmax-normalized weights. The network predicts residuals on
    centered inputs and outputs y_hat = residual + m(x), so gradients flow
    through the averaging weights.
  - No external scaling/min-max is used; the model learns end-to-end on raw scale.

Model attributes configurable via reticulate (R):
  - Weighted moving average
      model._aw_last_n: int   # 0 = all columns; N = last N columns
      model._aw_temp: float   # softmax temperature
  - Robust training
      model._huber_beta: float        # SmoothL1 beta
      model._grad_clip_enabled: bool  # enable gradient clipping
      model._grad_clip_norm: float    # clipping norm value

Example in R:
  model <- ts_lstm_anw(ts_norm_none(), input_size=4, epochs=10000)
  model$model$`_aw_last_n` <- 4
  model$model$`_aw_temp` <- 0.7
  model$model$`_huber_beta` <- 1.0
  model$model$`_grad_clip_enabled` <- TRUE
  model$model$`_grad_clip_norm` <- 1.0
  model <- fit(model, x=io_train$input, y=io_train$output)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class WeightedMeanLayer(nn.Module):
  def __init__(self, input_size: int):
    super().__init__()
    self.input_size = int(input_size)
    # Raw weights -> softmax to obtain convex combination
    self.w_raw = nn.Parameter(torch.zeros(self.input_size))

  def forward(self, x: torch.Tensor, last_n: int = 0, temp: float = 1.0) -> torch.Tensor:
    # x: (seq_len=1, batch, input_size)
    x0 = x[0]  # (batch, input_size)

    # Build mask to optionally restrict to last N columns
    if last_n is None or last_n <= 0 or last_n >= self.input_size:
      mask = torch.zeros_like(self.w_raw)
    else:
      mask = torch.full_like(self.w_raw, float('-inf'))
      start = self.input_size - int(last_n)
      mask[start:] = 0.0

    weights = torch.softmax((self.w_raw + mask) / max(float(temp), 1e-6), dim=0)  # (input_size,)
    # Weighted mean per batch
    m = (x0 * weights)  # broadcast (batch, input_size)
    m = m.sum(dim=1)    # (batch,)
    return m


class TsLSTMNetANW(nn.Module):
  def __init__(self, n_neurons: int, input_shape: int):
    super().__init__()
    self.input_shape = int(input_shape)
    self.norm = WeightedMeanLayer(self.input_shape)
    self.lstm = nn.LSTM(input_size=self.input_shape, hidden_size=int(n_neurons))
    self.fc = nn.Linear(int(n_neurons), 1)
    # Optional controls
    self._aw_last_n = 0
    self._aw_temp = 1.0
    # Robust training options
    self._huber_beta = 1.0           # SmoothL1 beta parameter
    self._grad_clip_enabled = True
    self._grad_clip_norm = 1.0

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Compute learnable weighted average m(x)
    m = self.norm(x, last_n=self._aw_last_n, temp=self._aw_temp)  # (batch,)
    # Center inputs by m(x) along feature dimension
    x_centered = x - m[None, :, None]
    out, _ = self.lstm(x_centered)
    res = self.fc(out)  # residual prediction
    # Recompose into original scale: y_hat = residual + m(x)
    y_hat = res + m[None, :, None]
    return y_hat


def ts_lstm_create_anw(n_neurons: int, look_back: int) -> TsLSTMNetANW:
  n_neurons = int(n_neurons)
  look_back = int(look_back)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = TsLSTMNetANW(n_neurons, look_back).to(device)
  return model


def _train_loop(epochs: int, lr: float, model: nn.Module, train_loader: DataLoader, opt_func=torch.optim.Adam):
  # Huber/SmoothL1 loss for robustness to outliers
  beta = getattr(model, "_huber_beta", 1.0)
  criterion = nn.SmoothL1Loss(beta=beta)
  optimizer = opt_func(model.parameters(), lr)
  last_error = float("inf")
  last_epoch = 0
  avg_train_losses = []
  convergency = max(epochs // 10, 100)

  for epoch in range(int(epochs)):
    model.train()
    train_losses = []
    for xb, yb in train_loader:
      model.zero_grad()
      device = next(model.parameters()).device
      out = model(xb.float().to(device))
      loss = criterion(out, yb.float().to(device))
      loss.backward()
      # Optional gradient clipping
      if getattr(model, "_grad_clip_enabled", False):
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=getattr(model, "_grad_clip_norm", 1.0))
      optimizer.step()
      train_losses.append(float(loss.item()))

    train_loss = float(np.average(train_losses)) if train_losses else 0.0
    avg_train_losses.append(train_loss)
    if (last_error - train_loss) > 1e-3:
      last_error = train_loss
      last_epoch = epoch
    if train_loss == 0.0:
      break
    if (epoch - last_epoch) > convergency:
      break

  return model, avg_train_losses


def ts_lstm_fit_anw(model: nn.Module, df_train, n_epochs: int = 10000, lr: float = 0.001):
  # Raw features/target; no external normalization
  X_train = df_train.drop('t0', axis=1).to_numpy()
  y_train = df_train['t0'].to_numpy()

  # Shape to LSTM input: (seq_len=1, batch, input_size)
  X = torch.from_numpy(X_train[:, :, None]).permute(2, 0, 1)
  y = torch.from_numpy(y_train[:, None]).permute(1, 0)[:, :, None]

  train_ds = TensorDataset(X, y)
  train_loader = DataLoader(train_ds, batch_size=8, shuffle=False)

  model = model.float()
  model, _ = _train_loop(n_epochs, lr, model, train_loader, opt_func=torch.optim.Adam)
  return model


def ts_lstm_predict_anw(model: nn.Module, df_test):
  X_test = df_test.drop('t0', axis=1, errors='ignore').to_numpy()

  Xt = torch.from_numpy(X_test[:, :, None]).permute(2, 0, 1)
  test_ds = TensorDataset(Xt, torch.zeros_like(Xt))  # labels unused
  test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

  outputs = []
  with torch.no_grad():
    for xb, _ in test_loader:
      device = next(model.parameters()).device
      out = model(xb.float().to(device))
      outputs.append(out.flatten().cpu())

  y_pred = torch.vstack(outputs).squeeze(1).numpy().reshape(-1)
  return y_pred.astype(float)
