"""A vectorized NumPy autograd engine for micrograd.

This keeps the micrograd idea (dynamic DAG + reverse-mode autodiff), but each
Value stores a whole NumPy array instead of one Python scalar. That changes the
cost of a dense layer from thousands of tiny scalar nodes to a handful of array
ops: matmul, add, ReLU, sum/mean.
"""

from __future__ import annotations

import numpy as np


DEFAULT_DTYPE = np.float32


def _as_array(data, dtype=None):
    """Convert numbers/lists/arrays to a floating ndarray.

    Floating NumPy arrays/scalars keep their dtype by default, so float32
    training data stays float32 and float64 arrays remain usable for gradient
    checks. Plain Python numbers and lists are converted to DEFAULT_DTYPE unless
    an explicit dtype is requested.
    """
    arr = np.asarray(data)
    if dtype is not None:
        return arr.astype(dtype, copy=False)
    if np.issubdtype(arr.dtype, np.floating) and isinstance(data, (np.ndarray, np.generic)):
        return arr.astype(arr.dtype, copy=False)
    return arr.astype(DEFAULT_DTYPE)


def _ensure_value(x, dtype=None):
    """Wrap constants as non-differentiable Values."""
    return x if isinstance(x, Value) else Value(x, requires_grad=False, dtype=dtype)


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum a broadcasted gradient back down to the original operand shape.

    Example: if b has shape (32,) and out = X + b has shape (200, 32), then
    dL/db is the row-sum of dL/dout, not the full (200, 32) array.
    """
    grad = np.asarray(grad)

    # Scalars receive the sum of all gradient contributions.
    if shape == ():
        return np.asarray(grad.sum(), dtype=grad.dtype)

    # Remove leading dimensions that were added by NumPy broadcasting.
    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)

    # Dimensions of size 1 were stretched; sum them back with keepdims=True.
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad.reshape(shape)


class Value:
    """Stores a NumPy array and its gradient."""

    __array_priority__ = 1000

    def __init__(self, data, _children=(), _op: str = "", requires_grad: bool = True, dtype=None):
        self.data = _as_array(data, dtype=dtype)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None

        # Internal variables used for autograd graph construction.
        self._backward = lambda: None
        self._prev = set(v for v in _children if getattr(v, "requires_grad", True))
        self._op = _op

    @property
    def shape(self):
        """Shape of the underlying NumPy array."""
        return self.data.shape

    def detach(self):
        """Return a non-differentiable Value with a copy of this Value's data."""
        return Value(self.data.copy(), requires_grad=False)

    # ---------------------------------------------------------------------
    # Small tensor plumbing used by vectorized models and normalizing flows.
    # These keep the engine NumPy-like without changing the micrograd design:
    # each operation still creates a single DAG node whose backward pass is a
    # simple vector-Jacobian product.
    # ---------------------------------------------------------------------
    def __getitem__(self, idx):
        """Differentiable NumPy-style slicing/indexing.

        The backward pass scatters the upstream gradient back into the original
        array shape. np.add.at handles repeated advanced indices correctly.
        """
        out = Value(self.data[idx], (self,), "slice", self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad = np.zeros_like(self.data)
                np.add.at(grad, idx, out.grad)
                self.grad += grad

        out._backward = _backward
        return out

    def reshape(self, *shape):
        """Differentiable reshape. Accepts reshape(2, 3) or reshape((2, 3))."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = Value(self.data.reshape(*shape), (self,), "reshape", self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def transpose(self, axes=None):
        """Differentiable transpose."""
        out = Value(np.transpose(self.data, axes=axes), (self,), "transpose", self.requires_grad)

        def _backward():
            if self.requires_grad:
                if axes is None:
                    self.grad += np.transpose(out.grad)
                else:
                    inverse_axes = np.argsort(axes)
                    self.grad += np.transpose(out.grad, axes=inverse_axes)

        out._backward = _backward
        return out

    @property
    def T(self):
        return self.transpose()

    @staticmethod
    def concatenate(values, axis=0):
        """Differentiable np.concatenate over a list/tuple of Values."""
        if not values:
            raise ValueError("Value.concatenate requires at least one value")

        first = values[0]
        dtype = first.data.dtype if isinstance(first, Value) else None
        vals = [_ensure_value(v, dtype=dtype) for v in values]
        data = np.concatenate([v.data for v in vals], axis=axis)
        requires_grad = any(v.requires_grad for v in vals)
        out = Value(data, tuple(vals), "concat", requires_grad)

        ndim = out.data.ndim
        axis_norm = axis if axis >= 0 else axis + ndim
        sizes = [v.data.shape[axis_norm] for v in vals]

        def _backward():
            start = 0
            for v, size in zip(vals, sizes):
                end = start + size
                if v.requires_grad:
                    slices = [slice(None)] * ndim
                    slices[axis_norm] = slice(start, end)
                    v.grad += out.grad[tuple(slices)]
                start = end

        out._backward = _backward
        return out

    def zero_grad(self):
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    def __add__(self, other):
        other = _ensure_value(other, dtype=self.data.dtype)
        out = Value(self.data + other.data, (self, other), "+", self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = _ensure_value(other, dtype=self.data.dtype)
        out = Value(self.data * other.data, (self, other), "*", self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(other.data * out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    # -------------------------------------------------------------------------
    # Matrix multiplication
    # -------------------------------------------------------------------------
    #
    # This implementation intentionally supports only the cases needed for this
    # educational vectorized micrograd:
    #
    #   1. Matrix @ Matrix
    #        (batch, in_features) @ (in_features, out_features)
    #        -> (batch, out_features)
    #
    #      This is the standard dense-layer case:
    #
    #        y = x @ W
    #
    #      Backward:
    #
    #        dL/dx = dL/dy @ W.T
    #        dL/dW = x.T @ dL/dy
    #
    #
    #   2. Vector @ Matrix
    #        (in_features,) @ (in_features, out_features)
    #        -> (out_features,)
    #
    #
    #   3. Matrix @ Vector
    #        (batch, in_features) @ (in_features,)
    #        -> (batch,)
    #
    #
    #   4. Vector @ Vector
    #        (n,) @ (n,)
    #        -> scalar dot product
    #
    #
    # Important:
    # ----------
    # NumPy's np.matmul supports more advanced behavior, including batched
    # matrix multiplication with arrays of dimension 3 or higher. This small
    # autograd engine does NOT implement the backward pass for those cases.
    #
    # That is intentional. Supporting full NumPy matmul broadcasting would make
    # the backward pass much more complicated and would distract from the main
    # purpose of this project: understanding reverse-mode autodiff with
    # vectorized neural-network operations.
    #
    # If operands with ndim > 2 are passed here, the forward NumPy operation may
    # work, but the backward pass is not implemented, so we raise
    # NotImplementedError.
    #
    # For this project, the most important supported case is:
    #
    #        X @ W
    #
    # where:
    #
    #        X.shape == (batch_size, input_size)
    #        W.shape == (input_size, output_size)
    #
    # This is enough to build MLP layers efficiently.
    def __matmul__(self, other):
        other = _ensure_value(other, dtype=self.data.dtype)
        if self.data.ndim > 2 or other.data.ndim > 2:
            raise NotImplementedError(
                "matmul currently supports only 1D/2D operands; "
                "batched matmul with ndim > 2 is not implemented"
            )
        out = Value(self.data @ other.data, (self, other), "@", self.requires_grad or other.requires_grad)

        def _backward():
            # Common dense-layer case: (batch, in) @ (in, out).
            if self.data.ndim == 2 and other.data.ndim == 2:
                if self.requires_grad:
                    self.grad += out.grad @ other.data.T
                if other.requires_grad:
                    other.grad += self.data.T @ out.grad
            # Convenience cases for vectors. They are less important for MLPs,
            # but make the operator usable in small experiments.
            elif self.data.ndim == 1 and other.data.ndim == 2:
                if self.requires_grad:
                    self.grad += out.grad @ other.data.T
                if other.requires_grad:
                    other.grad += np.outer(self.data, out.grad)
            elif self.data.ndim == 2 and other.data.ndim == 1:
                if self.requires_grad:
                    self.grad += np.outer(out.grad, other.data)
                if other.requires_grad:
                    other.grad += self.data.T @ out.grad
            elif self.data.ndim == 1 and other.data.ndim == 1:
                if self.requires_grad:
                    self.grad += other.data * out.grad
                if other.requires_grad:
                    other.grad += self.data * out.grad
            else:
                raise NotImplementedError(
                    "matmul backward currently supports only 1D/2D operands; "
                    "batched matmul with ndim > 2 is not implemented"
                )

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f"**{other}", self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(np.maximum(self.data, 0), (self,), "ReLU", self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Value(t, (self,), "tanh", self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def softmax_ce(self, targets):
        targets = np.asarray(targets, dtype=self.data.dtype)

        if self.data.ndim != 2:
            raise ValueError("softmax_ce expects logits with shape (batch, classes)")
        if targets.shape != self.data.shape:
            raise ValueError("targets must have the same shape as logits")

        shifted = self.data - self.data.max(axis=1, keepdims=True)
        exps = np.exp(shifted)
        probs = exps / exps.sum(axis=1, keepdims=True)

        n = targets.shape[0]
        loss = -(np.log(probs.clip(1e-7)) * targets).sum() / n
        out = Value(loss, (self,), "softmax_ce", self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += ((probs - targets) / n) * out.grad

        out._backward = _backward
        return out, probs

    def softmax_ce_sparse(self, targets):
        """Mean softmax cross entropy for integer class labels.

        Args:
            targets: 1-D integer array of class indices with shape (batch,).

        Returns:
            loss: scalar Value
            probs: NumPy array of class probabilities with same shape as logits
        """
        targets = np.asarray(targets, dtype=np.int64)

        if self.data.ndim != 2:
            raise ValueError("softmax_ce_sparse expects logits with shape (batch, classes)")
        if targets.ndim != 1:
            raise ValueError("targets must be a 1-D integer array of class indices")
        if targets.shape[0] != self.data.shape[0]:
            raise ValueError("targets length must match logits batch size")

        batch_size, num_classes = self.data.shape
        if np.any((targets < 0) | (targets >= num_classes)):
            raise ValueError("targets contain class indices outside the logits range")

        shifted = self.data - self.data.max(axis=1, keepdims=True)
        exps = np.exp(shifted)
        probs = exps / exps.sum(axis=1, keepdims=True)

        rows = np.arange(batch_size)
        loss = -np.log(probs[rows, targets].clip(1e-7)).mean()
        out = Value(loss, (self,), "softmax_ce_sparse", self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad = probs.copy()
                grad[rows, targets] -= 1.0
                self.grad += (grad / batch_size) * out.grad

        out._backward = _backward
        return out, probs

    def exp(self):
        e = np.exp(self.data)
        out = Value(e, (self,), "exp", self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += e * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Value(np.log(self.data), (self,), "log", self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += (1 / self.data) * out.grad

        out._backward = _backward
        return out

    def softplus(self):
        """Numerically stable log(1 + exp(x))."""
        data = np.logaddexp(0, self.data)
        out = Value(data, (self,), "softplus", self.requires_grad)

        def _backward():
            if self.requires_grad:
                # Stable sigmoid derivative of softplus.
                pos = self.data >= 0
                sigmoid = np.empty_like(self.data)
                sigmoid[pos] = 1.0 / (1.0 + np.exp(-self.data[pos]))
                exp_x = np.exp(self.data[~pos])
                sigmoid[~pos] = exp_x / (1.0 + exp_x)
                self.grad += sigmoid * out.grad

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims: bool = False):
        out = Value(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum", self.requires_grad)

        def _backward():
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis)
            if self.requires_grad:
                self.grad += np.ones_like(self.data) * grad

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims: bool = False):
        if axis is None:
            denom = self.data.size
        else:
            axes = axis if isinstance(axis, tuple) else (axis,)
            denom = 1
            for ax in axes:
                denom *= self.data.shape[ax]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / denom)

    def backward(self, grad=None):
        """Backpropagate from this Value through the dynamically built graph.

        Scalar outputs default to a seed gradient of 1.0. Non-scalar outputs
        require an explicit seed gradient, mirroring the vector-Jacobian product
        interface used by larger autograd systems.
        """
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        if grad is None:
            if self.data.shape != ():
                raise RuntimeError("grad must be specified for non-scalar outputs")
            grad = np.ones_like(self.data)
        else:
            grad = np.asarray(grad, dtype=self.data.dtype)
            if grad.shape != self.data.shape:
                raise ValueError(f"grad shape {grad.shape} does not match Value shape {self.data.shape}")

        self.grad = grad
        for v in reversed(topo):
            v._backward()

    def item(self):
        return self.data.item()

    def __float__(self):
        return float(self.data)

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __rmatmul__(self, other):
        other = _ensure_value(other, dtype=self.data.dtype)
        return other @ self

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def __repr__(self):
        if self.data.shape == ():
            data = self.data.item()
            grad = None if self.grad is None else self.grad.item()
            return f"Value(data={data}, grad={grad})"
        return f"Value(shape={self.data.shape}, data={self.data}, grad={self.grad})"
