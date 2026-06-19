import numpy as np

from vect_micrograd.flows import (
    AdditiveCoupling,
    AffineCoupling,
    DiagonalScaling,
    FlowSequential,
    NormalizingFlow,
    Permute,
    Reverse,
    StandardNormal,
    make_nice_flow,
    make_realnvp_flow,
)
from vect_micrograd.vect_engine import Value


def _make_eight_gaussians(n, radius=2.0, noise=0.08, seed=0, dtype=np.float64):
    rng = np.random.default_rng(seed)
    centers = []
    for k in range(8):
        angle = 2.0 * np.pi * k / 8.0
        centers.append([radius * np.cos(angle), radius * np.sin(angle)])
    centers = np.asarray(centers, dtype=dtype)

    labels = rng.integers(0, 8, size=n)
    X = centers[labels] + noise * rng.standard_normal((n, 2)).astype(dtype)
    X = X.astype(dtype, copy=False)
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)
    return X, labels


def test_slice_and_concatenate_backward():
    x_data = np.arange(6.0, dtype=np.float64).reshape(3, 2)
    x = Value(x_data)

    left = x[:, :1]
    right = x[:, 1:]
    y = Value.concatenate([left, right], axis=1)
    loss = (y * y).sum()
    loss.backward()

    assert y.data.shape == x_data.shape
    assert np.allclose(y.data, x_data)
    assert np.allclose(x.grad, 2.0 * x_data)


def test_reshape_transpose_backward():
    x_data = np.arange(6.0, dtype=np.float64)
    x = Value(x_data)
    y = x.reshape(2, 3).T
    loss = y.sum()
    loss.backward()

    assert y.data.shape == (3, 2)
    assert np.allclose(x.grad, np.ones_like(x_data))


def test_softplus_backward_against_sigmoid():
    x_data = np.array([-5.0, -1.0, 0.0, 1.0, 5.0], dtype=np.float64)
    x = Value(x_data)
    y = x.softplus().sum()
    y.backward()
    expected = 1.0 / (1.0 + np.exp(-x_data))
    assert np.allclose(x.grad, expected)


def test_additive_coupling_round_trip_and_zero_logdet():
    np.random.seed(0)
    layer = AdditiveCoupling(dim=4, hidden_sizes=(8,), flip=False, dtype=np.float64)
    x_data = np.random.randn(16, 4)

    y, logdet = layer.forward(Value(x_data, requires_grad=False))
    x_rec = layer.inverse(y)

    assert y.data.shape == x_data.shape
    assert logdet.data.shape == ()
    assert np.allclose(logdet.data, 0.0)
    assert np.allclose(x_rec.data, x_data, atol=1e-10)


def test_affine_coupling_round_trip_and_vector_logdet():
    np.random.seed(0)
    layer = AffineCoupling(dim=4, hidden_sizes=(8,), flip=True, dtype=np.float64)
    x_data = np.random.randn(16, 4)

    y, logdet = layer.forward(Value(x_data, requires_grad=False))
    x_rec = layer.inverse(y)

    assert y.data.shape == x_data.shape
    assert logdet.data.shape == (16,)
    assert np.all(np.isfinite(logdet.data))
    assert np.allclose(x_rec.data, x_data, atol=1e-10)


def test_diagonal_scaling_round_trip_and_logdet():
    layer = DiagonalScaling(dim=3, dtype=np.float64)
    layer.log_s.data[...] = np.array([0.1, -0.2, 0.3], dtype=np.float64)
    x_data = np.random.randn(5, 3)

    z, logdet = layer.forward(Value(x_data, requires_grad=False))
    x_rec = layer.inverse(z)

    assert z.data.shape == x_data.shape
    assert logdet.data.shape == ()
    assert np.allclose(logdet.data, layer.log_s.data.sum())
    assert np.allclose(x_rec.data, x_data, atol=1e-10)


def test_reverse_and_permute_round_trip_and_backward():
    x_data = np.arange(12.0, dtype=np.float64).reshape(3, 4)
    for layer in [Reverse(4), Permute([2, 0, 3, 1])]:
        x = Value(x_data)
        y, logdet = layer.forward(x)
        x_rec = layer.inverse(y)
        loss = (y * y).sum() + logdet
        loss.backward()

        assert y.data.shape == x_data.shape
        assert logdet.data.shape == ()
        assert np.allclose(x_rec.data, x_data)
        assert np.allclose(x.grad, 2.0 * x_data)


def test_flow_sequential_round_trip():
    np.random.seed(0)
    flow = FlowSequential(
        [
            AdditiveCoupling(4, hidden_sizes=(8,), flip=False, dtype=np.float64),
            Reverse(4),
            AdditiveCoupling(4, hidden_sizes=(8,), flip=True, dtype=np.float64),
            DiagonalScaling(4, dtype=np.float64),
        ]
    )
    x_data = np.random.randn(6, 4)
    z, logdet = flow.forward(Value(x_data, requires_grad=False))
    x_rec = flow.inverse(z)

    assert z.data.shape == x_data.shape
    assert logdet.data.shape == ()
    assert np.allclose(x_rec.data, x_data, atol=1e-10)


def test_normalizing_flow_log_prob_and_sample_shape():
    np.random.seed(1)
    flow = FlowSequential([AdditiveCoupling(2, hidden_sizes=(8,), dtype=np.float64), DiagonalScaling(2, dtype=np.float64)])
    model = NormalizingFlow(flow, StandardNormal(2), dim=2)
    x_data = np.random.randn(7, 2)

    logp = model.log_prob(x_data)
    samples = model.sample(11, dtype=np.float64)

    assert logp.data.shape == (7,)
    assert samples.shape == (11, 2)
    assert np.all(np.isfinite(logp.data))
    assert np.all(np.isfinite(samples))


def test_make_nice_flow_default_is_additive_round_trip():
    np.random.seed(0)
    model = make_nice_flow(dim=2, hidden_sizes=(8,), num_coupling_layers=4, dtype=np.float64)
    x_data = np.random.randn(16, 2)
    z, logdet = model.forward(Value(x_data, requires_grad=False))
    x_rec = model.inverse(z)

    assert model.kind == "nice"
    assert model.coupling == "additive"
    assert z.data.shape == x_data.shape
    assert logdet.data.shape == ()
    assert np.allclose(x_rec.data, x_data, atol=1e-10)


def test_make_realnvp_flow_is_affine_and_logdet_is_vector():
    np.random.seed(0)
    model = make_realnvp_flow(dim=2, hidden_sizes=(8,), num_coupling_layers=3, dtype=np.float64)
    x_data = np.random.randn(16, 2)
    z, logdet = model.forward(Value(x_data, requires_grad=False))
    x_rec = model.inverse(z)

    assert model.kind == "realnvp"
    assert model.coupling == "affine"
    assert z.data.shape == x_data.shape
    assert logdet.data.shape == (16,)
    assert np.allclose(x_rec.data, x_data, atol=1e-10)


def test_no_monolithic_nice_factory_is_exported():
    import vect_micrograd.flows as flows

    assert not hasattr(flows, "NICE")


def test_normalizing_flow_nll_backward_has_finite_parameter_grads():
    np.random.seed(1)
    X, _ = _make_eight_gaussians(32, seed=1, dtype=np.float64)
    model = make_realnvp_flow(dim=2, hidden_sizes=(8,), num_coupling_layers=2, dtype=np.float64)
    loss = model.nll(X)
    loss.backward()

    assert loss.data.shape == ()
    for p in model.parameters():
        assert p.grad.shape == p.data.shape
        assert np.all(np.isfinite(p.grad))
