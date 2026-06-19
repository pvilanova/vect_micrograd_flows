import numpy as np

from vect_micrograd.optim import RMSProp, RMSPropMomentum, RMSpropMomentum
from vect_micrograd.vect_engine import Value


def test_standard_rmsprop_first_step_matches_formula():
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


def test_standard_rmsprop_optional_momentum_uses_scaled_gradients():
    p = Value(np.array([1.0], dtype=np.float64))
    opt = RMSProp([p], lr=0.1, rho=0.5, eps=1e-8, momentum=0.9)

    p.grad = np.array([1.0], dtype=np.float64)
    opt.step(0)
    first_velocity = opt.velocity[0].copy()

    p.grad = np.array([1.0], dtype=np.float64)
    old_data = p.data.copy()
    expected_square_avg = 0.5 * opt.square_avg[0] + 0.5 * p.grad ** 2
    scaled_grad = p.grad / (np.sqrt(expected_square_avg) + 1e-8)
    expected_velocity = 0.9 * first_velocity + scaled_grad
    expected = old_data - 0.1 * expected_velocity

    opt.step(1)

    assert np.allclose(opt.square_avg[0], expected_square_avg)
    assert np.allclose(opt.velocity[0], expected_velocity)
    assert np.allclose(p.data, expected)


def test_matlab_rmsprop_momentum_first_step_matches_matlab_formula():
    p = Value(np.array([1.0, -2.0], dtype=np.float64))
    p.grad = np.array([0.5, -0.25], dtype=np.float64)
    opt = RMSPropMomentum(
        [p],
        lr=0.1,
        averaging_coeff=0.95,
        momentum=0.5,
        stabilizer=1e-2,
        weight_decay=0.0,
    )

    old_data = p.data.copy()
    grad = p.grad.copy()
    avg_grad_sq = 0.95 * np.zeros_like(grad) + 0.05 * grad ** 2
    rms_grad = np.maximum(np.sqrt(avg_grad_sq), 1e-2)
    mom = -0.1 * grad / rms_grad

    opt.step(0)

    assert np.allclose(opt.avg_grad_sq[0], avg_grad_sq)
    assert np.allclose(opt.mom[0], mom)
    assert np.allclose(p.data, old_data + mom)


def test_matlab_rmsprop_momentum_accumulates_momentum_with_floor():
    p = Value(np.array([1.0], dtype=np.float64))
    opt = RMSPropMomentum([p], lr=0.1, averaging_coeff=0.5, momentum=0.9, stabilizer=1e-2)

    p.grad = np.array([1.0], dtype=np.float64)
    opt.step(0)
    first_mom = opt.mom[0].copy()

    p.grad = np.array([1.0], dtype=np.float64)
    old_data = p.data.copy()
    old_avg_grad_sq = opt.avg_grad_sq[0].copy()
    expected_avg_grad_sq = 0.5 * old_avg_grad_sq + 0.5 * p.grad ** 2
    rms_grad = np.maximum(np.sqrt(expected_avg_grad_sq), 1e-2)
    expected_mom = 0.9 * first_mom - 0.1 * p.grad / rms_grad

    opt.step(1)

    assert np.allclose(opt.avg_grad_sq[0], expected_avg_grad_sq)
    assert np.allclose(opt.mom[0], expected_mom)
    assert np.allclose(p.data, old_data + expected_mom)


def test_matlab_rmsprop_pylearn2_style_lr_and_momentum_schedules():
    p = Value(np.array([1.0], dtype=np.float64))
    opt = RMSPropMomentum(
        [p],
        lr=1e-3,
        min_lr=1e-4,
        lr_decay_factor=1.0005,
        init_momentum=0.0,
        final_momentum=0.5,
        momentum_start_epoch=5,
        momentum_saturate_epoch=6,
        steps_per_epoch=10,
    )

    # First update: no decay yet, epoch 1 so initial momentum.
    assert np.isclose(opt._current_lr(0), 1e-3)
    assert np.isclose(opt._current_momentum(0), 0.0)

    # Epoch 5 is still initial momentum, epoch 6 and later uses final momentum.
    assert np.isclose(opt._current_momentum(40), 0.0)
    assert np.isclose(opt._current_momentum(50), 0.5)

    # Learning rate is divided by the decay factor per update and clipped by min_lr.
    assert np.isclose(opt._current_lr(10), 1e-3 / (1.0005 ** 10))
    assert np.isclose(opt._current_lr(1_000_000), 1e-4)


def test_rmsprop_momentum_alias_spelling():
    assert RMSpropMomentum is RMSPropMomentum
