"""Normalizing-flow components built on vect_micrograd.

This module intentionally separates the three pieces that make up a flow model:

1. invertible transforms, e.g. additive/affine coupling layers;
2. a base distribution, e.g. a standard normal or logistic distribution;
3. a generic NormalizingFlow wrapper that handles log_prob, nll, and sampling.

The original NICE model is now just one builder:

    make_nice_flow(...)

which composes additive coupling layers with a final diagonal scaling layer.
The Real NVP-style extension is another builder:

    make_realnvp_flow(...)

which composes affine coupling layers and optional permutations/scaling.

There is no monolithic NICE model class here: named builders return a generic
NormalizingFlow assembled from composable transforms and base distributions.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from vect_micrograd.vect_engine import DEFAULT_DTYPE, Value
from vect_micrograd.vect_nn import MLP, Module

LOG_2PI = float(np.log(2.0 * np.pi))


def _as_value(x) -> Value:
    """Return x as a Value, treating plain arrays as non-differentiable inputs."""
    return x if isinstance(x, Value) else Value(x, requires_grad=False)


def _zero_logdet(dtype=DEFAULT_DTYPE) -> Value:
    return Value(np.array(0.0, dtype=dtype), requires_grad=False)


def _check_2d_dim(x: Value, dim: int, name: str):
    if x.data.ndim != 2 or x.data.shape[1] != dim:
        raise ValueError(f"expected {name} with shape (batch, {dim})")


def normal_logpdf(z: Value) -> Value:
    """Elementwise standard normal log-density."""
    z = _as_value(z)
    return -0.5 * (z * z + LOG_2PI)


def logistic_logpdf(z: Value) -> Value:
    """Elementwise standard logistic log-density.

    Mathematically this is

        -log(1 + exp(z)) - log(1 + exp(-z))

    implemented with the engine's stable softplus operation.
    """
    z = _as_value(z)
    return -z.softplus() - (-z).softplus()


class StandardNormal:
    """Factorized standard normal base distribution."""

    def __init__(self, dim: int):
        if dim < 1:
            raise ValueError("dim must be >= 1")
        self.dim = int(dim)

    def log_prob(self, z) -> Value:
        z = _as_value(z)
        _check_2d_dim(z, self.dim, "z")
        return normal_logpdf(z).sum(axis=1)

    def sample(self, n: int, dtype=DEFAULT_DTYPE) -> np.ndarray:
        return np.random.randn(int(n), self.dim).astype(dtype, copy=False)

    def __repr__(self):
        return f"StandardNormal(dim={self.dim})"


class StandardLogistic:
    """Factorized standard logistic base distribution."""

    def __init__(self, dim: int):
        if dim < 1:
            raise ValueError("dim must be >= 1")
        self.dim = int(dim)

    def log_prob(self, z) -> Value:
        z = _as_value(z)
        _check_2d_dim(z, self.dim, "z")
        return logistic_logpdf(z).sum(axis=1)

    def sample(self, n: int, dtype=DEFAULT_DTYPE) -> np.ndarray:
        # Inverse CDF of the standard logistic distribution.
        u = np.random.rand(int(n), self.dim).astype(dtype, copy=False)
        eps = np.finfo(np.dtype(dtype)).eps
        u = np.clip(u, eps, 1.0 - eps)
        return np.log(u) - np.log1p(-u)

    def __repr__(self):
        return f"StandardLogistic(dim={self.dim})"


def make_prior(prior, dim: int):
    """Create a base distribution from a string or return an existing object."""
    if hasattr(prior, "log_prob") and hasattr(prior, "sample"):
        if getattr(prior, "dim", dim) != dim:
            raise ValueError(f"prior dim {prior.dim} does not match model dim {dim}")
        return prior
    if prior == "normal":
        return StandardNormal(dim)
    if prior == "logistic":
        return StandardLogistic(dim)
    raise ValueError("prior must be 'normal', 'logistic', or a distribution object")


class AdditiveCoupling(Module):
    """NICE-style additive coupling layer.

    Args:
        dim: Input/output dimensionality.
        hidden_sizes: Hidden layer sizes for the conditioner MLP t(.).
        flip: If False, the left block conditions the right block. If True,
            the right block conditions the left block.

    Forward transform:

        y_id     = x_id
        y_change = x_change + t(x_id)

    The log-determinant is exactly zero, because the transform is volume
    preserving.
    """

    def __init__(self, dim: int, hidden_sizes=(64, 64), flip: bool = False, dtype=DEFAULT_DTYPE):
        if dim < 2:
            raise ValueError("AdditiveCoupling requires dim >= 2")

        self.dim = int(dim)
        self.split = self.dim // 2
        self.flip = bool(flip)

        if self.flip:
            conditioner_dim = self.dim - self.split
            transformed_dim = self.split
        else:
            conditioner_dim = self.split
            transformed_dim = self.dim - self.split

        self.transformed_dim = transformed_dim
        self.net = MLP(conditioner_dim, list(hidden_sizes) + [transformed_dim], dtype=dtype)

    def _split(self, x: Value):
        if not self.flip:
            return x[:, : self.split], x[:, self.split :]
        return x[:, self.split :], x[:, : self.split]

    def _merge(self, x_id: Value, x_change: Value):
        if not self.flip:
            return Value.concatenate([x_id, x_change], axis=1)
        return Value.concatenate([x_change, x_id], axis=1)

    def forward(self, x):
        x = _as_value(x)
        _check_2d_dim(x, self.dim, "x")

        x_id, x_change = self._split(x)
        y_change = x_change + self.net(x_id)
        y = self._merge(x_id, y_change)
        return y, _zero_logdet(x.data.dtype)

    def inverse(self, y):
        y = _as_value(y)
        _check_2d_dim(y, self.dim, "y")

        y_id, y_change = self._split(y)
        x_change = y_change - self.net(y_id)
        return self._merge(y_id, x_change)

    def parameters(self):
        return self.net.parameters()

    def __repr__(self):
        direction = "right->left" if self.flip else "left->right"
        return f"AdditiveCoupling(dim={self.dim}, {direction})"


class AffineCoupling(Module):
    """Real NVP-style affine coupling layer.

    Forward transform:

        y_id     = x_id
        y_change = x_change * exp(s(x_id)) + t(x_id)
        logdet   = sum_j s_j(x_id)

    The scale is bounded as max_log_scale * tanh(raw_scale) for numerical
    stability in this small educational engine.
    """

    def __init__(
        self,
        dim: int,
        hidden_sizes=(64, 64),
        flip: bool = False,
        max_log_scale: float = 2.0,
        dtype=DEFAULT_DTYPE,
    ):
        if dim < 2:
            raise ValueError("AffineCoupling requires dim >= 2")

        self.dim = int(dim)
        self.split = self.dim // 2
        self.flip = bool(flip)
        self.max_log_scale = float(max_log_scale)

        if self.flip:
            conditioner_dim = self.dim - self.split
            transformed_dim = self.split
        else:
            conditioner_dim = self.split
            transformed_dim = self.dim - self.split

        self.transformed_dim = transformed_dim
        self.net = MLP(conditioner_dim, list(hidden_sizes) + [2 * transformed_dim], dtype=dtype)

    def _split(self, x: Value):
        if not self.flip:
            return x[:, : self.split], x[:, self.split :]
        return x[:, self.split :], x[:, : self.split]

    def _merge(self, x_id: Value, x_change: Value):
        if not self.flip:
            return Value.concatenate([x_id, x_change], axis=1)
        return Value.concatenate([x_change, x_id], axis=1)

    def _shift_and_log_scale(self, x_id):
        out = self.net(x_id)
        shift = out[:, : self.transformed_dim]
        raw_log_scale = out[:, self.transformed_dim :]
        log_scale = self.max_log_scale * raw_log_scale.tanh()
        return shift, log_scale

    def forward(self, x):
        x = _as_value(x)
        _check_2d_dim(x, self.dim, "x")

        x_id, x_change = self._split(x)
        shift, log_scale = self._shift_and_log_scale(x_id)
        y_change = x_change * log_scale.exp() + shift
        y = self._merge(x_id, y_change)
        return y, log_scale.sum(axis=1)

    def inverse(self, y):
        y = _as_value(y)
        _check_2d_dim(y, self.dim, "y")

        y_id, y_change = self._split(y)
        shift, log_scale = self._shift_and_log_scale(y_id)
        x_change = (y_change - shift) * (-log_scale).exp()
        return self._merge(y_id, x_change)

    def parameters(self):
        return self.net.parameters()

    def __repr__(self):
        direction = "right->left" if self.flip else "left->right"
        return f"AffineCoupling(dim={self.dim}, {direction}, max_log_scale={self.max_log_scale})"


class DiagonalScaling(Module):
    """Trainable elementwise scaling transform.

    This is the final non-volume-preserving layer used by the original NICE
    model. In Real NVP-style models, affine coupling already provides local
    non-volume-preserving behavior, so this layer is optional.
    """

    def __init__(self, dim: int, dtype=DEFAULT_DTYPE):
        if dim < 1:
            raise ValueError("DiagonalScaling requires dim >= 1")
        self.dim = int(dim)
        self.log_s = Value(np.zeros(self.dim, dtype=dtype))

    def forward(self, x):
        x = _as_value(x)
        _check_2d_dim(x, self.dim, "x")
        z = x * self.log_s.exp()
        return z, self.log_s.sum()

    def inverse(self, z):
        z = _as_value(z)
        _check_2d_dim(z, self.dim, "z")
        return z * (-self.log_s).exp()

    def parameters(self):
        return [self.log_s]

    def __repr__(self):
        return f"DiagonalScaling(dim={self.dim})"


class Reverse(Module):
    """Dimension-reversing permutation transform with zero log-determinant."""

    def __init__(self, dim: int):
        if dim < 1:
            raise ValueError("Reverse requires dim >= 1")
        self.dim = int(dim)
        self.perm = np.arange(self.dim - 1, -1, -1)
        self.inv_perm = np.argsort(self.perm)

    def forward(self, x):
        x = _as_value(x)
        _check_2d_dim(x, self.dim, "x")
        return x[:, self.perm], _zero_logdet(x.data.dtype)

    def inverse(self, z):
        z = _as_value(z)
        _check_2d_dim(z, self.dim, "z")
        return z[:, self.inv_perm]

    def parameters(self):
        return []

    def __repr__(self):
        return f"Reverse(dim={self.dim})"


class Permute(Module):
    """Fixed permutation transform with zero log-determinant."""

    def __init__(self, perm: Sequence[int]):
        perm = np.asarray(perm, dtype=int)
        if perm.ndim != 1:
            raise ValueError("perm must be a 1D sequence")
        if sorted(perm.tolist()) != list(range(len(perm))):
            raise ValueError("perm must be a permutation of range(dim)")
        self.perm = perm
        self.inv_perm = np.argsort(perm)
        self.dim = int(len(perm))

    def forward(self, x):
        x = _as_value(x)
        _check_2d_dim(x, self.dim, "x")
        return x[:, self.perm], _zero_logdet(x.data.dtype)

    def inverse(self, z):
        z = _as_value(z)
        _check_2d_dim(z, self.dim, "z")
        return z[:, self.inv_perm]

    def parameters(self):
        return []

    def __repr__(self):
        return f"Permute(dim={self.dim}, perm={self.perm.tolist()})"


class FlowSequential(Module):
    """A sequential composition of invertible transforms.

    Each layer must implement:

        forward(x) -> (z, logdet)
        inverse(z) -> x
        parameters() -> list[Value]
    """

    def __init__(self, layers: Iterable[Module]):
        self.layers = list(layers)
        if not self.layers:
            raise ValueError("FlowSequential requires at least one layer")

    def forward(self, x):
        h = _as_value(x)
        logdet = _zero_logdet(h.data.dtype)
        for layer in self.layers:
            h, layer_logdet = layer.forward(h)
            logdet = logdet + layer_logdet
        return h, logdet

    def inverse(self, z):
        x = _as_value(z)
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def __iter__(self):
        return iter(self.layers)

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, idx):
        return self.layers[idx]

    def __repr__(self):
        return "FlowSequential([" + ", ".join(repr(layer) for layer in self.layers) + "])"


class NormalizingFlow(Module):
    """Generic normalizing-flow density model."""

    def __init__(self, flow: FlowSequential, prior, dim: int | None = None, name: str = "NormalizingFlow"):
        self.flow = flow
        self.prior = prior
        self.name = name
        self.dim = int(dim if dim is not None else prior.dim)
        if getattr(prior, "dim", self.dim) != self.dim:
            raise ValueError(f"prior dim {prior.dim} does not match model dim {self.dim}")

    def forward(self, x):
        return self.flow.forward(x)

    def inverse(self, z):
        return self.flow.inverse(z)

    def log_prob(self, x):
        """Return per-example log p(x), shape (batch,)."""
        z, logdet = self.forward(x)
        return self.prior.log_prob(z) + logdet

    def nll(self, x):
        """Mean negative log-likelihood for a minibatch."""
        return -self.log_prob(x).mean()

    def sample(self, n: int, dtype=DEFAULT_DTYPE):
        """Draw samples from the model as a NumPy array."""
        z = self.prior.sample(n, dtype=dtype)
        return self.inverse(Value(z, requires_grad=False)).data

    def parameters(self):
        return self.flow.parameters()

    def __repr__(self):
        return f"{self.name}(dim={self.dim}, layers={len(self.flow)}, prior={self.prior})"


def _maybe_add_reverse(layers: list[Module], dim: int, enabled: bool):
    if enabled and dim > 1:
        layers.append(Reverse(dim))


def make_nice_flow(
    dim: int,
    hidden_sizes=(64, 64),
    num_coupling_layers: int = 4,
    prior: str | object = "normal",
    dtype=DEFAULT_DTYPE,
    use_reverse: bool = False,
) -> NormalizingFlow:
    """Build the original NICE-style flow.

    Architecture:

        additive coupling layers -> optional reverses -> diagonal scaling -> prior

    The coupling layers have zero log-determinant; DiagonalScaling contributes
    the global non-volume-preserving term.
    """
    if dim < 2:
        raise ValueError("make_nice_flow requires dim >= 2")
    if num_coupling_layers < 1:
        raise ValueError("num_coupling_layers must be >= 1")

    layers: list[Module] = []
    for i in range(num_coupling_layers):
        layers.append(AdditiveCoupling(dim, hidden_sizes=hidden_sizes, flip=bool(i % 2), dtype=dtype))
        if i != num_coupling_layers - 1:
            _maybe_add_reverse(layers, dim, use_reverse)
    scaling = DiagonalScaling(dim, dtype=dtype)
    layers.append(scaling)

    model = NormalizingFlow(FlowSequential(layers), make_prior(prior, dim), dim=dim, name="NICEFlow")
    model.kind = "nice"
    model.coupling = "additive"
    model.scaling = scaling
    return model


def make_realnvp_flow(
    dim: int,
    hidden_sizes=(64, 64),
    num_coupling_layers: int = 4,
    prior: str | object = "normal",
    max_log_scale: float = 2.0,
    dtype=DEFAULT_DTYPE,
    use_reverse: bool = False,
    use_final_scaling: bool = False,
) -> NormalizingFlow:
    """Build a Real NVP-style affine coupling flow.

    Architecture:

        affine coupling layers -> optional reverses -> optional diagonal scaling -> prior

    Affine coupling layers already provide input-dependent log-determinants, so
    the final diagonal scaling layer is optional.
    """
    if dim < 2:
        raise ValueError("make_realnvp_flow requires dim >= 2")
    if num_coupling_layers < 1:
        raise ValueError("num_coupling_layers must be >= 1")

    layers: list[Module] = []
    for i in range(num_coupling_layers):
        layers.append(
            AffineCoupling(
                dim,
                hidden_sizes=hidden_sizes,
                flip=bool(i % 2),
                max_log_scale=max_log_scale,
                dtype=dtype,
            )
        )
        if i != num_coupling_layers - 1:
            _maybe_add_reverse(layers, dim, use_reverse)

    scaling = None
    if use_final_scaling:
        scaling = DiagonalScaling(dim, dtype=dtype)
        layers.append(scaling)

    model = NormalizingFlow(FlowSequential(layers), make_prior(prior, dim), dim=dim, name="RealNVPFlow")
    model.kind = "realnvp"
    model.coupling = "affine"
    if scaling is not None:
        model.scaling = scaling
    return model


def make_flow(
    kind: str,
    dim: int,
    hidden_sizes=(64, 64),
    num_coupling_layers: int = 4,
    prior: str | object = "normal",
    dtype=DEFAULT_DTYPE,
    **kwargs,
) -> NormalizingFlow:
    """Generic builder for named flow families."""
    kind = kind.lower()
    if kind in {"nice", "additive"}:
        return make_nice_flow(
            dim=dim,
            hidden_sizes=hidden_sizes,
            num_coupling_layers=num_coupling_layers,
            prior=prior,
            dtype=dtype,
            **kwargs,
        )
    if kind in {"realnvp", "affine"}:
        return make_realnvp_flow(
            dim=dim,
            hidden_sizes=hidden_sizes,
            num_coupling_layers=num_coupling_layers,
            prior=prior,
            dtype=dtype,
            **kwargs,
        )
    raise ValueError("kind must be 'nice'/'additive' or 'realnvp'/'affine'")
