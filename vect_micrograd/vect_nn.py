"""Vectorized neural-network helpers for micrograd.

This module deliberately stays small: it only defines model building blocks.
Training loops, optimizers, and checkpointing live in the demo code.
"""

import numpy as np

from vect_micrograd.vect_engine import DEFAULT_DTYPE, Value


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def parameters(self):
        return []

class Layer(Module):
    def __init__(self, nin, nout, bias=True, dtype=DEFAULT_DTYPE):
        scale = np.sqrt(2.0 / nin)
        self.W = Value((np.random.randn(nin, nout) * scale).astype(dtype, copy=False))
        self.b = Value(np.zeros(nout, dtype=dtype)) if bias else None

    def __call__(self, x):
        x = x if isinstance(x, Value) else Value(x, requires_grad=False)
        out = x @ self.W
        return out + self.b if self.b is not None else out

    def parameters(self):
        return [self.W] + ([] if self.b is None else [self.b])

    def __repr__(self):
        return f"Layer({self.W.data.shape[0]}, {self.W.data.shape[1]})"


class MLP(Module):
    def __init__(self, nin, nouts, activation=Value.relu, dtype=DEFAULT_DTYPE):
        sz = [nin] + list(nouts)
        self.layers = [Layer(sz[i], sz[i + 1], dtype=dtype) for i in range(len(nouts))]
        self.activation = activation

    def __call__(self, x):
        x = x if isinstance(x, Value) else Value(x, requires_grad=False)
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
