"""A tiny hand-rolled 2-hidden-layer MLP with manual backprop and Adam.

Kept dependency-free (NumPy only) and fully transparent on purpose: the
point of this demo is to *see* learning happen, not to hide it behind an
autograd framework.
"""

import numpy as np


class TinyMLP:
    def __init__(self, input_dim=1, hidden=64, rng=None, lr=1e-3):
        rng = rng or np.random.default_rng()
        self.W1 = rng.normal(scale=1.0 / np.sqrt(input_dim), size=(input_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(scale=1.0 / np.sqrt(hidden), size=(hidden, hidden))
        self.b2 = np.zeros(hidden)
        self.W3 = rng.normal(scale=1.0 / np.sqrt(hidden), size=(hidden, 1))
        self.b3 = np.zeros(1)
        self.lr = lr

        self._params = ["W1", "b1", "W2", "b2", "W3", "b3"]
        self._m = {p: np.zeros_like(getattr(self, p)) for p in self._params}
        self._v = {p: np.zeros_like(getattr(self, p)) for p in self._params}
        self._t = 0

    def forward(self, x):
        """x: (N, input_dim) -> (N, 1), linear output."""
        self._x = x
        self._z1 = x @ self.W1 + self.b1
        self._a1 = np.tanh(self._z1)
        self._z2 = self._a1 @ self.W2 + self.b2
        self._a2 = np.tanh(self._z2)
        self._z3 = self._a2 @ self.W3 + self.b3
        return self._z3

    def backward(self, grad_out):
        """grad_out = dLoss/d(output), already averaged over the batch (i.e.
        includes the 1/N factor from a mean-reduced loss)."""
        dW3 = self._a2.T @ grad_out
        db3 = grad_out.sum(axis=0)

        da2 = grad_out @ self.W3.T
        dz2 = da2 * (1 - self._a2**2)
        dW2 = self._a1.T @ dz2
        db2 = dz2.sum(axis=0)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * (1 - self._a1**2)
        dW1 = self._x.T @ dz1
        db1 = dz1.sum(axis=0)

        grads = dict(W1=dW1, b1=db1, W2=dW2, b2=db2, W3=dW3, b3=db3)
        self._adam_step(grads)

    def _adam_step(self, grads, beta1=0.9, beta2=0.999, eps=1e-8):
        self._t += 1
        for p in self._params:
            g = grads[p]
            self._m[p] = beta1 * self._m[p] + (1 - beta1) * g
            self._v[p] = beta2 * self._v[p] + (1 - beta2) * g**2
            m_hat = self._m[p] / (1 - beta1**self._t)
            v_hat = self._v[p] / (1 - beta2**self._t)
            update = self.lr * m_hat / (np.sqrt(v_hat) + eps)
            setattr(self, p, getattr(self, p) - update)

    def predict(self, x):
        return self.forward(x)

    def save(self, path):
        """Weights only -- enough to reconstruct a working predict(), not to
        resume training (the Adam moment estimates aren't saved)."""
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, W3=self.W3, b3=self.b3)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        input_dim, hidden = data["W1"].shape
        model = cls(input_dim=input_dim, hidden=hidden)
        for p in model._params:
            setattr(model, p, data[p])
        return model
