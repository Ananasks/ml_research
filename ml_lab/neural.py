"""Real recurrent regression on windows of rows, not windows of feature columns."""
import copy
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class RecurrentRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, units=32, layers=1, dropout=0.1, dense_units=16,
                 window=12, epochs=30, batch_size=64, learning_rate=0.001,
                 optimizer='Adam', seed=42, cell='LSTM'):
        self.units = units
        self.layers = layers
        self.dropout = dropout
        self.dense_units = dense_units
        self.window = window
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.seed = seed
        self.cell = cell

    def _windows(self, values, context=None, breaks=None):
        import torch
        values = np.asarray(values, dtype=np.float32)
        if len(values) * self.window * values.shape[1] > 40_000_000:
            raise ValueError('Слишком большой объём последовательностей. Уменьшите окно или число признаков.')
        prefix = np.repeat(values[:1], self.window - 1, axis=0) if context is None else context
        joined = np.concatenate([prefix, values])
        windows=np.stack([joined[i:i+self.window] for i in range(len(values))])
        if breaks is not None:
            if len(breaks)!=len(values): raise ValueError('Некорректные границы временных окон.')
            for start in np.flatnonzero(breaks):
                for i in range(start,min(start+self.window-1,len(values))):
                    chunk=values[start:i+1]
                    windows[i]=np.concatenate([np.repeat(values[start:start+1],self.window-len(chunk),axis=0),chunk])
        return torch.from_numpy(windows)

    def fit(self, X, y):
        import torch
        from torch import nn
        torch.manual_seed(self.seed)
        torch.set_num_threads(2)
        X = np.asarray(X, dtype=np.float32)
        if len(X) < self.window + 8:
            raise ValueError('Слишком мало обучающих строк для выбранного окна LSTM.')
        self.y_mean_, self.y_scale_ = float(np.mean(y)), max(float(np.std(y)), 1e-8)
        labels = torch.tensor((np.asarray(y, dtype=np.float32) - self.y_mean_) / self.y_scale_).reshape(-1, 1)
        self.rnn_ = getattr(nn, self.cell)(X.shape[1], self.units, self.layers, batch_first=True,
                            dropout=self.dropout if self.layers > 1 else 0.)
        self.head_ = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(self.units, self.dense_units),
                                   nn.ReLU(), nn.Linear(self.dense_units, 1))
        parameters = list(self.rnn_.parameters()) + list(self.head_.parameters())
        optim = getattr(torch.optim, self.optimizer)(parameters, lr=self.learning_rate)
        criterion = nn.MSELoss()
        windows = self._windows(X,breaks=getattr(self,'fit_breaks_',None))
        # Internal chronological validation for early stopping; outer validation and test are untouched.
        boundary = int(len(X) * .85)
        if boundary <= self.window:
            raise ValueError('Уменьшите окно LSTM или увеличьте набор данных.')
        self.loss_curve_, self.validation_curve_ = [], []
        best, best_state, stale = float('inf'), None, 0
        for _ in range(self.epochs):
            self.rnn_.train(); self.head_.train()
            losses = []
            for start in range(0, boundary, self.batch_size):
                end = min(start + self.batch_size, boundary)
                optim.zero_grad()
                output, _ = self.rnn_(windows[start:end])
                loss = criterion(self.head_(output[:, -1]), labels[start:end])
                if not torch.isfinite(loss):
                    raise ValueError('LSTM получила нечисловую ошибку. Уменьшите learning rate.')
                loss.backward()
                nn.utils.clip_grad_norm_(parameters, 1.)
                optim.step()
                losses.append(float(loss.detach()))
            self.rnn_.eval(); self.head_.eval()
            with torch.no_grad():
                output, _ = self.rnn_(windows[boundary:])
                val_loss = float(criterion(self.head_(output[:, -1]), labels[boundary:]))
            self.loss_curve_.append(float(np.mean(losses)))
            self.validation_curve_.append(val_loss)
            if val_loss < best:
                best, stale = val_loss, 0
                best_state = (copy.deepcopy(self.rnn_.state_dict()), copy.deepcopy(self.head_.state_dict()))
            else:
                stale += 1
            if stale >= 6:
                break
        self.rnn_.load_state_dict(best_state[0]); self.head_.load_state_dict(best_state[1])
        self.context_ = X[-(self.window - 1):] if self.window > 1 else X[:0]
        return self

    def predict(self, X):
        import torch
        breaks=getattr(self,'prediction_breaks_',None)
        self.prediction_breaks_=None
        windows = self._windows(X, self.context_,breaks=breaks)
        self.rnn_.eval(); self.head_.eval()
        predictions = []
        with torch.no_grad():
            for start in range(0, len(windows), self.batch_size):
                output, _ = self.rnn_(windows[start:start+self.batch_size])
                predictions.extend(self.head_(output[:, -1]).flatten().tolist())
        return np.asarray(predictions) * self.y_scale_ + self.y_mean_
