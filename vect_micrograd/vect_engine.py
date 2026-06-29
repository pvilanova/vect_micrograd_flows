"""A vectorized NumPy autograd engine for micrograd.

This keeps the micrograd idea (dynamic DAG + reverse-mode autodiff), but each
Value stores a whole NumPy array instead of one Python scalar. That changes the
cost of a dense layer from thousands of tiny scalar nodes to a handful of array
ops: matmul, add, ReLU, sum/mean.

Performance-oriented version:
- lazy gradient allocation: Value.grad starts as None and is allocated only when
  a gradient contribution actually arrives;
- in-place zero_grad for already-allocated gradients;
- fast backward path for ordinary slicing;
- sum backward uses broadcasting instead of np.ones_like(...) temporaries.
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


def _add_grad(v: "Value", grad) -> None:
    """Accumulate a gradient contribution into v with lazy allocation.

    The original engine eagerly allocated ``np.zeros_like(data)`` for every
    differentiable intermediate. That is simple, but expensive for deep dynamic
    graphs. This helper allocates only when a contribution arrives.
    """
    if not v.requires_grad:
        return
    if grad is None:
        return

    grad_arr = np.asarray(grad, dtype=v.data.dtype)
    if v.grad is None:
        # Copy because grad may be a view/broadcasted view or may be reused by
        # the caller. After this point v owns its gradient buffer.
        v.grad = np.array(grad_arr, dtype=v.data.dtype, copy=True)
    else:
        v.grad += grad_arr


def _is_basic_index(idx) -> bool:
    """Return True for ordinary NumPy indexing where ``grad[idx] += ...`` is safe.

    Advanced integer/boolean-array indexing can contain repeated indices. In
    those cases ``grad[idx] += out.grad`` is not a correct scatter-add, so we
    fall back to np.add.at.
    """
    if not isinstance(idx, tuple):
        idx = (idx,)

    basic_types = (slice, int, np.integer, type(Ellipsis), type(None))
    return all(isinstance(part, basic_types) for part in idx)


class Value:
    """Stores a NumPy array and its gradient."""

    __array_priority__ = 1000

    def __init__(self, data, _children=(), _op: str = "", requires_grad: bool = True, dtype=None):
        self.data = _as_array(data, dtype=dtype)
        self.requires_grad = requires_grad

        # Lazy gradient allocation: no gradient buffer is created until a
        # backward pass actually contributes to this Value.
        self.grad = None

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

        Fast path:
            ordinary slicing writes directly into ``self.grad[idx]`` and avoids
            a full temporary plus np.add.at.

        Fallback:
            advanced indexing uses np.add.at so repeated indices are handled
            correctly.
        """
        out = Value(self.data[idx], (self,), "slice", self.requires_grad)
        basic_index = _is_basic_index(idx)

        def _backward():
            if self.requires_grad and out.grad is not None:
                if basic_index:
                    if self.grad is None:
                        self.grad = np.zeros_like(self.data)
                    self.grad[idx] += out.grad
                else:
                    grad = np.zeros_like(self.data)
                    np.add.at(grad, idx, out.grad)
                    _add_grad(self, grad)

        out._backward = _backward
        return out

    def reshape(self, *shape):
        """Differentiable reshape. Accepts reshape(2, 3) or reshape((2, 3))."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = Value(self.data.reshape(*shape), (self,), "reshape", self.requires_grad)

        def _backward():
            if self.requires_grad and out.grad is not None:
                _add_grad(self, out.grad.reshape(self.data.shape))

        out._backward = _backward
        return out

    def transpose(self, axes=None):
        """Differentiable transpose."""
        out = Value(np.transpose(self.data, axes=axes), (self,), "transpose", self.requires_grad)

        def _backward():
            if self.requires_grad and out.grad is not None:
                if axes is None:
                    _add_grad(self, np.transpose(out.grad))
                else:
                    inverse_axes = np.argsort(axes)
                    _add_grad(self, np.transpose(out.grad, axes=inverse_axes))

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
            if out.grad is None:
                return
            start = 0
            for v, size in zip(vals, sizes):
                end = start + size
                if v.requires_grad:
                    slices = [slice(None)] * ndim
                    slices[axis_norm] = slice(start, end)
                    _add_grad(v, out.grad[tuple(slices)])
                start = end

        out._backward = _backward
        return out

    def zero_grad(self):
        """Clear this Value's gradient.

        If a gradient buffer already exists, zero it in-place. If no gradient
        has ever been allocated, keep it as None. This preserves the lazy
        allocation benefit while avoiding repeated allocation in training loops.
        """
        if self.requires_grad and self.grad is not None:
            self.grad.fill(0.0)

    def __add__(self, other):
        other = _ensure_value(other, dtype=self.data.dtype)
        out = Value(self.data + other.data, (self, other), "+", self.requires_grad or other.requires_grad)

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                _add_grad(self, _unbroadcast(out.grad, self.data.shape))
            if other.requires_grad:
                _add_grad(other, _unbroadcast(out.grad, other.data.shape))

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = _ensure_value(other, dtype=self.data.dtype)
        out = Value(self.data * other.data, (self, other), "*", self.requires_grad or other.requires_grad)

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                _add_grad(self, _unbroadcast(other.data * out.grad, self.data.shape))
            if other.requires_grad:
                _add_grad(other, _unbroadcast(self.data * out.grad, other.data.shape))

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
            if out.grad is None:
                return
            # Common dense-layer case: (batch, in) @ (in, out).
            if self.data.ndim == 2 and other.data.ndim == 2:
                if self.requires_grad:
                    _add_grad(self, out.grad @ other.data.T)
                if other.requires_grad:
                    _add_grad(other, self.data.T @ out.grad)
            # Convenience cases for vectors. They are less important for MLPs,
            # but make the operator usable in small experiments.
            elif self.data.ndim == 1 and other.data.ndim == 2:
                if self.requires_grad:
                    _add_grad(self, out.grad @ other.data.T)
                if other.requires_grad:
                    _add_grad(other, np.outer(self.data, out.grad))
            elif self.data.ndim == 2 and other.data.ndim == 1:
                if self.requires_grad:
                    _add_grad(self, np.outer(out.grad, other.data))
                if other.requires_grad:
                    _add_grad(other, self.data.T @ out.grad)
            elif self.data.ndim == 1 and other.data.ndim == 1:
                if self.requires_grad:
                    _add_grad(self, other.data * out.grad)
                if other.requires_grad:
                    _add_grad(other, self.data * out.grad)
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
            if self.requires_grad and out.grad is not None:
                _add_grad(self, other * (self.data ** (other - 1)) * out.grad)

        out._backward = _backward
        return out

    def relu(self):
        out = Value(np.maximum(self.data, 0), (self,), "ReLU", self.requires_grad)

        def _backward():
            if self.requires_grad and out.grad is not None:
                _add_grad(self, (self.data > 0) * out.grad)

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Value(t, (self,), "tanh", self.requires_grad)

        def _backward():
            if self.requires_grad and out.grad is not None:
                _add_grad(self, (1 - t**2) * out.grad)

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
            if self.requires_grad and out.grad is not None:
                _add_grad(self, ((probs - targets) / n) * out.grad)

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
            if self.requires_grad and out.grad is not None:
                grad = probs.copy()
                grad[rows, targets] -= 1.0
                _add_grad(self, (grad / batch_size) * out.grad)

        out._backward = _backward
        return out, probs

    def exp(self):
        e = np.exp(self.data)
        out = Value(e, (self,), "exp", self.requires_grad)

        def _backward():
            if self.requires_grad and out.grad is not None:
                _add_grad(self, e * out.grad)

        out._backward = _backward
        return out

    def log(self):
        out = Value(np.log(self.data), (self,), "log", self.requires_grad)

        def _backward():
            if self.requires_grad and out.grad is not None:
                _add_grad(self, (1 / self.data) * out.grad)

        out._backward = _backward
        return out

    def softplus(self):
        """Numerically stable log(1 + exp(x))."""
        data = np.logaddexp(0, self.data)
        out = Value(data, (self,), "softplus", self.requires_grad)

        def _backward():
            if self.requires_grad and out.grad is not None:
                # Stable sigmoid derivative of softplus.
                pos = self.data >= 0
                sigmoid = np.empty_like(self.data)
                sigmoid[pos] = 1.0 / (1.0 + np.exp(-self.data[pos]))
                exp_x = np.exp(self.data[~pos])
                sigmoid[~pos] = exp_x / (1.0 + exp_x)
                _add_grad(self, sigmoid * out.grad)

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims: bool = False):
        out = Value(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum", self.requires_grad)

        def _backward():
            if self.requires_grad and out.grad is not None:
                grad = out.grad

                if axis is not None and not keepdims:
                    axes = axis if isinstance(axis, tuple) else (axis,)
                    axes = tuple(ax if ax >= 0 else ax + self.data.ndim for ax in axes)
                    for ax in sorted(axes):
                        grad = np.expand_dims(grad, ax)

                # Avoid np.ones_like(self.data) * grad. broadcast_to creates a
                # view when possible; _add_grad copies only if this is the first
                # gradient contribution to self.
                _add_grad(self, np.broadcast_to(grad, self.data.shape))

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

        # Seed output gradient. If backward is called repeatedly without
        # zero_grad, accumulate in micrograd style instead of silently replacing
        # existing gradients on the output node.
        _add_grad(self, grad)

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
