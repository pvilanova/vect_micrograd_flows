import numpy as np

from vect_micrograd.optim import RMSProp
from vect_micrograd.vect_engine import Value


def test_rmsprop_first_step_matches_formula():
    p = Value(np.array([1.0, -2.0], dtype=np.float64))
    p.grad = np.array([0.5, -0.25], dtype=np.float64)
    opt = RMSProp([p], lr=0.1, rho=0.95, eps=1e-8, weight_decay=0.0)

    old_data = p.data.copy()
    grad = p.grad.copy()
    square_avg = 0.95 * np.zeros_like(grad) + 0.05 * grad ** 2
    expected = old_data - 0.1 * grad / (np.sqrt(square_avg) + 1e-8)

    opt.step(0)

    assert np.allclose(opt.square_avg[0], square_avg)
    assert np.allclose(p.data, expected)


def test_rmsprop_accumulates_square_average():
    p = Value(np.array([1.0], dtype=np.float64))
    opt = RMSProp([p], lr=0.1, rho=0.5, eps=1e-8)

    p.grad = np.array([1.0], dtype=np.float64)
    opt.step(0)

    p.grad = np.array([1.0], dtype=np.float64)
    old_data = p.data.copy()
    expected_square_avg = 0.5 * opt.square_avg[0] + 0.5 * p.grad ** 2
    expected = old_data - 0.1 * p.grad / (np.sqrt(expected_square_avg) + 1e-8)

    opt.step(1)

    assert np.allclose(opt.square_avg[0], expected_square_avg)
    assert np.allclose(p.data, expected)


def test_rmsprop_weight_decay_is_added_to_gradient():
    p = Value(np.array([2.0], dtype=np.float64))
    p.grad = np.array([0.5], dtype=np.float64)
    opt = RMSProp([p], lr=0.1, rho=0.5, eps=1e-8, weight_decay=0.1)

    grad = p.grad + 0.1 * p.data
    expected_square_avg = 0.5 * grad ** 2
    expected = p.data - 0.1 * grad / (np.sqrt(expected_square_avg) + 1e-8)

    opt.step(0)

    assert np.allclose(opt.square_avg[0], expected_square_avg)
    assert np.allclose(p.data, expected)


def test_rmsprop_tracks_current_learning_rate_schedule():
    p = Value(np.array([1.0], dtype=np.float64))
    opt = RMSProp([p], lr=1e-3, total_steps=100, rho=0.95, eps=1e-8)

    assert np.isclose(opt._current_lr(0), 1e-3)
    assert np.isclose(opt._current_lr(50), 1e-3 * (1.0 - 0.9 * 50 / 100))

    p.grad = np.array([1.0], dtype=np.float64)
    opt.step(50)

    assert np.isclose(opt.current_lr, opt._current_lr(50))
