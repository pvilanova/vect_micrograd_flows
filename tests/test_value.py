import numpy as np
from vect_micrograd.vect_engine import Value

def test_scalar_square():
    x = Value(3.0)
    y = x * x
    y.backward()
    assert np.allclose(x.grad, 6.0)

def test_repeated_use():
    x = Value(2.0)
    y = x + x
    y.backward()
    assert np.allclose(x.grad, 2.0)

def test_broadcast_add_bias():
    x = Value(np.ones((5, 3)))
    b = Value(np.zeros(3))
    y = x + b
    loss = y.sum()
    loss.backward()
    assert b.grad.shape == (3,)
    assert np.allclose(b.grad, np.ones(3) * 5)

def test_mul_broadcast():
    x = Value(np.ones((4, 3)))
    w = Value(np.array([1.0, 2.0, 3.0]))
    y = x * w
    loss = y.sum()
    loss.backward()
    assert np.allclose(w.grad, np.ones(3) * 4)

def test_matmul_shapes():
    x = Value(np.random.randn(5, 4))
    W = Value(np.random.randn(4, 3))
    b = Value(np.random.randn(3))

    y = x @ W + b
    loss = y.sum()
    loss.backward()

    assert x.grad.shape == x.data.shape
    assert W.grad.shape == W.data.shape
    assert b.grad.shape == b.data.shape

def test_numerical_gradient_simple_mlp_like():
    np.random.seed(0)

    x = Value(np.random.randn(4, 3))
    W = Value(np.random.randn(3, 2))
    b = Value(np.random.randn(2))

    y = x @ W + b
    loss = (y * y).mean()
    loss.backward()

    eps = 1e-6
    original = W.data[0, 0]

    W.data[0, 0] = original + eps
    loss_plus = (((x.data @ W.data + b.data) ** 2).mean())

    W.data[0, 0] = original - eps
    loss_minus = (((x.data @ W.data + b.data) ** 2).mean())

    W.data[0, 0] = original

    numerical_grad = (loss_plus - loss_minus) / (2 * eps)

    assert np.allclose(W.grad[0, 0], numerical_grad, atol=1e-5)

def test_relu():
    x = Value(np.array([[-1.0, 0.0, 2.0],
                        [3.0, -4.0, 5.0]]))

    y = x.relu().sum()
    y.backward()

    expected_grad = np.array([[0.0, 0.0, 1.0],
                              [1.0, 0.0, 1.0]])

    assert np.allclose(x.grad, expected_grad)


def test_tanh():
    x_data = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    x = Value(x_data)

    y = x.tanh().sum()
    y.backward()

    expected_grad = 1 - np.tanh(x_data) ** 2

    assert np.allclose(x.grad, expected_grad)


def test_exp():
    x_data = np.array([-1.0, 0.0, 1.0, 2.0])
    x = Value(x_data)

    y = x.exp().sum()
    y.backward()

    expected_grad = np.exp(x_data)

    assert np.allclose(x.grad, expected_grad)


def test_log():
    x_data = np.array([0.5, 1.0, 2.0, 4.0])
    x = Value(x_data)

    y = x.log().sum()
    y.backward()

    expected_grad = 1 / x_data

    assert np.allclose(x.grad, expected_grad)


def test_sum_axis_0():
    x = Value(np.arange(6.0).reshape(2, 3))

    y = x.sum(axis=0)
    loss = y.sum()
    loss.backward()

    expected_grad = np.ones_like(x.data)

    assert y.data.shape == (3,)
    assert np.allclose(x.grad, expected_grad)


def test_sum_axis_1():
    x = Value(np.arange(6.0).reshape(2, 3))

    y = x.sum(axis=1)
    loss = y.sum()
    loss.backward()

    expected_grad = np.ones_like(x.data)

    assert y.data.shape == (2,)
    assert np.allclose(x.grad, expected_grad)


def test_sum_axis_keepdims():
    x = Value(np.arange(6.0).reshape(2, 3))

    y = x.sum(axis=1, keepdims=True)
    loss = y.sum()
    loss.backward()

    expected_grad = np.ones_like(x.data)

    assert y.data.shape == (2, 1)
    assert np.allclose(x.grad, expected_grad)


def test_mean_axis_0():
    x = Value(np.arange(6.0).reshape(2, 3))

    y = x.mean(axis=0)
    loss = y.sum()
    loss.backward()

    expected_grad = np.ones_like(x.data) / 2

    assert y.data.shape == (3,)
    assert np.allclose(x.grad, expected_grad)


def test_mean_axis_1():
    x = Value(np.arange(6.0).reshape(2, 3))

    y = x.mean(axis=1)
    loss = y.sum()
    loss.backward()

    expected_grad = np.ones_like(x.data) / 3

    assert y.data.shape == (2,)
    assert np.allclose(x.grad, expected_grad)


def test_softmax_ce_probabilities_sum_to_one():
    logits = Value(np.array([[1.0, 2.0, 3.0],
                             [1.0, 0.0, -1.0]]))

    targets = np.array([[0.0, 0.0, 1.0],
                        [1.0, 0.0, 0.0]])

    loss, probs = logits.softmax_ce(targets)

    assert loss.data.shape == ()
    assert probs.shape == logits.data.shape
    assert np.allclose(probs.sum(axis=1), np.ones(2))

def test_softmax_ce_numerical_gradient():
    logits_data = np.array([[1.0, 2.0, 3.0],
                            [1.0, 0.0, -1.0]])

    targets = np.array([[0.0, 0.0, 1.0],
                        [1.0, 0.0, 0.0]])

    logits = Value(logits_data.copy())
    loss, probs = logits.softmax_ce(targets)
    loss.backward()

    eps = 1e-6
    i, j = 0, 1

    plus = logits_data.copy()
    plus[i, j] += eps

    shifted_plus = plus - plus.max(axis=1, keepdims=True)
    exps_plus = np.exp(shifted_plus)
    probs_plus = exps_plus / exps_plus.sum(axis=1, keepdims=True)
    loss_plus = -(np.log(probs_plus.clip(1e-7)) * targets).sum() / len(targets)

    minus = logits_data.copy()
    minus[i, j] -= eps

    shifted_minus = minus - minus.max(axis=1, keepdims=True)
    exps_minus = np.exp(shifted_minus)
    probs_minus = exps_minus / exps_minus.sum(axis=1, keepdims=True)
    loss_minus = -(np.log(probs_minus.clip(1e-7)) * targets).sum() / len(targets)

    numerical_grad = (loss_plus - loss_minus) / (2 * eps)

    assert np.allclose(logits.grad[i, j], numerical_grad, atol=1e-5)


def test_softmax_ce_loss_is_mean():
    logits = Value(np.array([[1.0, 2.0, 3.0],
                             [1.0, 0.0, -1.0]]))

    targets = np.array([[0.0, 0.0, 1.0],
                        [1.0, 0.0, 0.0]])

    loss, probs = logits.softmax_ce(targets)

    expected = -(np.log(probs.clip(1e-7)) * targets).sum() / len(targets)

    assert np.allclose(loss.data, expected)

def test_softmax_ce_respects_upstream_grad():
    logits = Value(np.array([[1.0, 2.0, 3.0],
                             [1.0, 0.0, -1.0]]))

    targets = np.array([[0.0, 0.0, 1.0],
                        [1.0, 0.0, 0.0]])

    loss, probs = logits.softmax_ce(targets)
    scaled_loss = 2.0 * loss
    scaled_loss.backward()

    expected_grad = 2.0 * (probs - targets) / len(targets)

    assert np.allclose(logits.grad, expected_grad)