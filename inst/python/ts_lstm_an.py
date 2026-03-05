"""
Adaptive-normalized LSTM forecaster for daltoolboxdp via reticulate.

Interface (compatible with ts_lstm.py):
  - ts_lstm_create(n_neurons, look_back) -> torch.nn.Module
  - ts_lstm_fit(model, df_train, n_epochs, lr) -> model
  - ts_lstm_predict(model, df_test) -> np.ndarray of shape (n_samples,)

What it does:
  - Performs adaptive normalization internally so callers may pass raw
    sliding-window data (i.e., use ts_norm_none() on the R side).
  - Adaptive step: subtract per-row moving average (over last `nw` columns
    or all columns if `nw == 0`) from X and y, then apply global scaling
    to [0,1] derived from the centered training data. Prediction reverses
    both steps using stored parameters.

Model attributes configurable via reticulate (R):
  - Normalization
      model._an_nw: int     # 0 = all columns; N = last N columns
      model._an_gmin/_an_gmax: floats learned on fit (read-only)
  - Robust handling of outliers (optional)
      model._robust_clip: bool    # clip centered data by quantiles before scaling
      model._clip_q_low: float    # e.g., 0.01
      model._clip_q_high: float   # e.g., 0.99
      model._robust_minmax: bool  # use quantile-based min/max for scaling
      model._mm_q_low: float      # e.g., 0.05
      model._mm_q_high: float     # e.g., 0.95

Example in R (reticulate):
  model <- ts_lstm(ts_norm_none(), input_size=4, epochs=10000)
  # set options before fit
  model$model$`_an_nw` <- 3
  model$model$`_robust_clip` <- TRUE
  model$model$`_robust_minmax` <- TRUE
  model <- fit(model, x=io_train$input, y=io_train$output)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple


class TsLSTMNet(nn.Module):
  def __init__(self, n_neurons: int, input_shape: int):
    super().__init__()
    self.lstm = nn.LSTM(input_size=input_shape, hidden_size=n_neurons)
    self.fc = nn.Linear(n_neurons, 1)
    # Adaptive normalization state (filled during fit)
    self._an_nw: int = 0      # 0 -> use all columns
    self._an_gmin: float | None = None
    self._an_gmax: float | None = None
    # Robust options for outliers (configurable from R via reticulate)
    self._robust_clip: bool = False
    self._clip_q_low: float = 0.01
    self._clip_q_high: float = 0.99
    self._robust_minmax: bool = False
    self._mm_q_low: float = 0.05
    self._mm_q_high: float = 0.95

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    out, _ = self.lstm(x)
    out = self.fc(out)
    return out


def _row_mean_last_n(X: np.ndarray, nw: int) -> np.ndarray:
  """Per-row mean over the last `nw` columns. If nw == 0 or >= n_features,
  use all columns. X shape: (n_samples, n_features). Returns shape: (n_samples,).
  Mirrors R logic: cols <- ncol(data) - ((nw-1):0) -> last `nw` columns.
  """
  n_features = X.shape[1]
  if nw is None or nw <= 0 or nw >= n_features:
    return X.mean(axis=1)
  cols = np.arange(n_features - nw, n_features)
  return X[:, cols].mean(axis=1)


def _center_and_scale_train(X: np.ndarray, y: np.ndarray, nw: int,
                            robust_clip: bool = False, clip_q_low: float = 0.01, clip_q_high: float = 0.99,
                            robust_minmax: bool = False, mm_q_low: float = 0.05, mm_q_high: float = 0.95
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
  """Center X and y by per-row moving average, then compute global min/max
  on the centered data and scale to [0, 1]. Returns (Xn, yn, an, gmin, gmax).
  """
  an = _row_mean_last_n(X, nw)  # shape: (n_samples,)
  Xc = X - an[:, None]
  yc = y - an

  # Optional robust clipping by global quantiles (on centered data)
  Xc_used = Xc
  yc_used = yc
  if robust_clip:
    flat = np.concatenate([Xc.reshape(-1), yc.reshape(-1)], axis=0)
    lo = np.quantile(flat, clip_q_low)
    hi = np.quantile(flat, clip_q_high)
    Xc_used = np.clip(Xc, lo, hi)
    yc_used = np.clip(yc, lo, hi)

  # Robust or plain min/max over the used centered data
  flat_used = np.concatenate([Xc_used.reshape(-1), yc_used.reshape(-1)], axis=0)
  if robust_minmax:
    gmin = np.quantile(flat_used, mm_q_low)
    gmax = np.quantile(flat_used, mm_q_high)
  else:
    gmin = np.min(flat_used)
    gmax = np.max(flat_used)
  rng = gmax - gmin
  if rng == 0:
    rng = 1.0
  Xn = (Xc_used - gmin) / rng
  yn = (yc_used - gmin) / rng
  return Xn, yn, an, float(gmin), float(gmax)


def _center_and_scale_infer(X: np.ndarray, gmin: float, gmax: float, nw: int) -> Tuple[np.ndarray, np.ndarray]:
  """Center X by per-row moving average and scale using training gmin/gmax.
  Returns (Xn, an)."""
  an = _row_mean_last_n(X, nw)
  Xc = X - an[:, None]
  rng = gmax - gmin
  if rng == 0:
    rng = 1.0
  Xn = (Xc - gmin) / rng
  return Xn, an


def _invert_scale(pred_scaled: np.ndarray, an: np.ndarray, gmin: float, gmax: float) -> np.ndarray:
  """Invert min–max scaling and add back the per-row moving average to recover
  predictions in the original data scale."""
  rng = gmax - gmin
  if rng == 0:
    rng = 1.0
  y = pred_scaled * rng + gmin
  y = y + an
  return y


def ts_lstm_create_an(n_neurons: int, look_back: int) -> TsLSTMNet:
  """Factory called from R to create the LSTM model for a given look-back window.
  Normalization parameters are stored on the returned module.
  """
  n_neurons = int(n_neurons)
  look_back = int(look_back)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = TsLSTMNet(n_neurons, look_back).to(device)
  # Default: adaptive mean over all columns (nw=0). Caller may override.
  model._an_nw = 0
  return model


def _train_loop(epochs: int, lr: float, model: TsLSTMNet, train_loader: DataLoader, opt_func=torch.optim.Adam):
  criterion = nn.MSELoss()
  optimizer = opt_func(model.parameters(), lr)
  last_error = float("inf")
  last_epoch = 0
  avg_train_losses = []
  convergency = max(epochs // 10, 100)

  for epoch in range(epochs):
    model.train()
    train_losses = []
    for xb, yb in train_loader:
      model.zero_grad()
      device = next(model.parameters()).device
      out = model(xb.float().to(device))
      loss = nn.functional.mse_loss(out, yb.float().to(device))
      loss.backward()
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


def ts_lstm_fit_an(model: TsLSTMNet, df_train, n_epochs: int = 10000, lr: float = 0.001) -> TsLSTMNet:
  """Fit LSTM with internal adaptive normalization.

  Expects df_train as a pandas.DataFrame with lag columns and a 't0' target.
  """
  n_epochs = int(n_epochs)

  # Extract raw features/target
  X_train = df_train.drop('t0', axis=1).to_numpy()
  y_train = df_train['t0'].to_numpy()

  # Adaptive normalization: per-row mean with optional robust handling
  Xn, yn, an, gmin, gmax = _center_and_scale_train(
    X_train, y_train,
    getattr(model, "_an_nw", 0),
    robust_clip=getattr(model, "_robust_clip", False),
    clip_q_low=getattr(model, "_clip_q_low", 0.01),
    clip_q_high=getattr(model, "_clip_q_high", 0.99),
    robust_minmax=getattr(model, "_robust_minmax", False),
    mm_q_low=getattr(model, "_mm_q_low", 0.05),
    mm_q_high=getattr(model, "_mm_q_high", 0.95),
  )

  # Persist normalization parameters on model for inference
  model._an_gmin = gmin
  model._an_gmax = gmax

  # Shape to LSTM expected input: (seq_len=1, batch, input_size=look_back)
  Xn = Xn[:, :, np.newaxis]              # (N, F, 1)
  yn = yn[:, np.newaxis]                 # (N, 1)
  tx = torch.from_numpy(Xn).permute(2, 0, 1)      # (1, N, F)
  ty = torch.from_numpy(yn).permute(1, 0)[:, :, None]  # (1, N, 1)

  train_ds = TensorDataset(tx, ty)
  train_loader = DataLoader(train_ds, batch_size=8, shuffle=False)

  model = model.float()
  model, _ = _train_loop(n_epochs, lr, model, train_loader, opt_func=torch.optim.Adam)
  return model


def ts_lstm_predict_an(model: TsLSTMNet, df_test) -> np.ndarray:
  """Predict with LSTM and invert the internal adaptive normalization.

  Expects df_test as a pandas.DataFrame with the same lag columns (and may
  include a dummy 't0' column, which is ignored).
  """
  X_test = df_test.drop('t0', axis=1, errors='ignore').to_numpy()

  if model._an_gmin is None or model._an_gmax is None:
    raise RuntimeError("Model normalization parameters are not set. Fit the model first.")
  # Apply the same centering and scaling used in training
  Xn, an = _center_and_scale_infer(X_test, model._an_gmin, model._an_gmax, getattr(model, "_an_nw", 0))

    # Shape to LSTM input and run forward
  Xn = Xn[:, :, np.newaxis]
  tx = torch.from_numpy(Xn).permute(2, 0, 1)  # (1, N, F)
  test_ds = TensorDataset(tx, torch.zeros_like(tx))  # labels unused
  test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

  outputs = []
  with torch.no_grad():
    for xb, _ in test_loader:
      device = next(model.parameters()).device
      out = model(xb.float().to(device))  # (1, B, 1)
      outputs.append(out.flatten().cpu())

  y_scaled = torch.vstack(outputs).squeeze(1).numpy().reshape(-1)
  y_pred = _invert_scale(y_scaled, an, model._an_gmin, model._an_gmax)
  return y_pred.astype(float)

  