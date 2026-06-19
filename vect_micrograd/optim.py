import numpy as np


class Optimizer:
    def __init__(self, parameters, lr, total_steps=None):
        self.parameters = list(parameters)
        self.lr = lr
        self.total_steps = total_steps

    def _current_lr(self, k):
        if self.total_steps is None:
            return self.lr
        return self.lr * (1.0 - 0.9 * k / self.total_steps)

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()

    def step(self, k):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, parameters, lr=1e-2, total_steps=None, momentum=0.0):
        super().__init__(parameters, lr, total_steps)
        self.momentum = momentum
        self.velocity = [np.zeros_like(p.data) for p in self.parameters] if momentum else None

    def step(self, k):
        lr = self._current_lr(k)
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue
            if self.momentum:
                self.velocity[i] = self.momentum * self.velocity[i] + p.grad
                p.data -= lr * self.velocity[i]
            else:
                p.data -= lr * p.grad


class RMSProp(Optimizer):
    def __init__(
        self,
        parameters,
        lr=1e-3,
        total_steps=None,
        rho=0.99,
        eps=1e-8,
        momentum=0.0,
        weight_decay=0.0,
    ):
        super().__init__(parameters, lr, total_steps)

        if not 0.0 <= rho < 1.0:
            raise ValueError("rho must satisfy 0 <= rho < 1")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must satisfy 0 <= momentum < 1")

        self.rho = float(rho)
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)

        self.square_avg = [np.zeros_like(p.data) for p in self.parameters]
        self.velocity = [np.zeros_like(p.data) for p in self.parameters] if self.momentum else None
        self.current_lr = float(lr)

    def step(self, k):
        lr = self._current_lr(k)
        self.current_lr = lr

        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue

            grad = p.grad
            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * p.data

            self.square_avg[i] = self.rho * self.square_avg[i] + (1.0 - self.rho) * (grad ** 2)
            denom = np.sqrt(self.square_avg[i]) + self.eps
            scaled_grad = grad / denom

            if self.momentum:
                self.velocity[i] = self.momentum * self.velocity[i] + scaled_grad
                p.data -= lr * self.velocity[i]
            else:
                p.data -= lr * scaled_grad


class RMSPropMomentum(Optimizer):
    """Pylearn2/MATLAB-style RMSProp with heavy-ball momentum.

    This mirrors the MATLAB NICE files ``rmspropMomentumUpdateArray2018.m`` and
    ``rmspropMomentumUpdateNet2018.m``::

        avg_grad_sq = averaging_coeff * avg_grad_sq + (1 - averaging_coeff) * grad**2
        rms_grad    = maximum(sqrt(avg_grad_sq), stabilizer)
        mom         = momentum_coeff * mom - lr * grad / rms_grad
        parameter   = parameter + mom

    There is no Adam bias correction.  The denominator is a hard floor
    ``max(sqrt(avg_grad_sq), stabilizer)``, not ``sqrt(avg_grad_sq) + eps``.
    """

    def __init__(
        self,
        parameters,
        lr=1e-3,
        total_steps=None,
        averaging_coeff=0.95,
        momentum=0.5,
        stabilizer=1e-2,
        weight_decay=0.0,
        initial_avg_grad_sq=0.0,
        min_lr=None,
        lr_decay_factor=None,
        init_momentum=None,
        final_momentum=None,
        momentum_start_epoch=None,
        momentum_saturate_epoch=None,
        steps_per_epoch=None,
        # Compatibility aliases.
        rho=None,
        eps=None,
        initial_square_avg=None,
    ):
        super().__init__(parameters, lr, total_steps)

        if rho is not None:
            averaging_coeff = rho
        if eps is not None:
            stabilizer = eps
        if initial_square_avg is not None:
            initial_avg_grad_sq = initial_square_avg

        if not 0.0 <= averaging_coeff < 1.0:
            raise ValueError("averaging_coeff must satisfy 0 <= value < 1")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must satisfy 0 <= momentum < 1")
        if stabilizer <= 0.0:
            raise ValueError("stabilizer must be positive")
        if min_lr is not None and min_lr <= 0.0:
            raise ValueError("min_lr must be positive when provided")
        if lr_decay_factor is not None and lr_decay_factor < 1.0:
            raise ValueError("lr_decay_factor must be >= 1 when provided")

        self.averaging_coeff = float(averaging_coeff)
        self.rho = self.averaging_coeff
        self.momentum = float(momentum)
        self.stabilizer = float(stabilizer)
        self.eps = self.stabilizer
        self.weight_decay = float(weight_decay)
        self.min_lr = None if min_lr is None else float(min_lr)
        self.lr_decay_factor = None if lr_decay_factor is None else float(lr_decay_factor)

        self.init_momentum = init_momentum
        self.final_momentum = final_momentum
        self.momentum_start_epoch = momentum_start_epoch
        self.momentum_saturate_epoch = momentum_saturate_epoch
        self.steps_per_epoch = steps_per_epoch

        schedule_args = (init_momentum, final_momentum, momentum_start_epoch, momentum_saturate_epoch, steps_per_epoch)
        if any(v is not None for v in schedule_args):
            if any(v is None for v in schedule_args):
                raise ValueError(
                    "momentum scheduling requires init_momentum, final_momentum, "
                    "momentum_start_epoch, momentum_saturate_epoch, and steps_per_epoch"
                )
            if steps_per_epoch <= 0:
                raise ValueError("steps_per_epoch must be positive")
            if momentum_saturate_epoch <= momentum_start_epoch:
                raise ValueError("momentum_saturate_epoch must be greater than momentum_start_epoch")
            if not 0.0 <= init_momentum < 1.0 or not 0.0 <= final_momentum < 1.0:
                raise ValueError("scheduled momenta must satisfy 0 <= value < 1")

        self.avg_grad_sq = [
            np.full_like(p.data, initial_avg_grad_sq, dtype=p.data.dtype)
            for p in self.parameters
        ]
        self.mom = [np.zeros_like(p.data) for p in self.parameters]

        # Common aliases for inspection/debugging.
        self.square_avg = self.avg_grad_sq
        self.velocity = self.mom
        self.current_lr = float(lr)
        self.current_momentum = float(momentum)

    def _current_lr(self, k):
        if self.lr_decay_factor is not None:
            lr = self.lr / (self.lr_decay_factor ** max(0, int(k)))
            if self.min_lr is not None:
                lr = max(self.min_lr, lr)
            return lr
        lr = super()._current_lr(k)
        if self.min_lr is not None:
            lr = max(self.min_lr, lr)
        return lr

    def _current_momentum(self, k):
        if self.steps_per_epoch is None:
            return self.momentum

        # MATLAB loop epochs are 1-based; step k is zero-based here.
        epoch = int(k) // int(self.steps_per_epoch) + 1
        start = self.momentum_start_epoch
        saturate = self.momentum_saturate_epoch

        if epoch < start:
            return float(self.init_momentum)
        if epoch >= saturate:
            return float(self.final_momentum)

        t = (epoch - start) / (saturate - start)
        return float(self.init_momentum + t * (self.final_momentum - self.init_momentum))

    def step(self, k):
        lr = self._current_lr(k)
        momentum_coeff = self._current_momentum(k)
        self.current_lr = lr
        self.current_momentum = momentum_coeff

        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue

            grad = p.grad
            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * p.data

            self.avg_grad_sq[i] = (
                self.averaging_coeff * self.avg_grad_sq[i]
                + (1.0 - self.averaging_coeff) * (grad ** 2)
            )
            rms_grad = np.maximum(np.sqrt(self.avg_grad_sq[i]), self.stabilizer)
            self.mom[i] = momentum_coeff * self.mom[i] - lr * grad / rms_grad
            p.data += self.mom[i]

        self.square_avg = self.avg_grad_sq
        self.velocity = self.mom

class Adam(Optimizer):
    def __init__(self, parameters, lr=1e-2, total_steps=None, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-2):
        super().__init__(parameters, lr, total_steps)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(p.data) for p in self.parameters]
        self.v = [np.zeros_like(p.data) for p in self.parameters]

    def step(self, k):
        lr = self._current_lr(k)
        t = k + 1
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue
            if self.weight_decay != 0.0:
                p.data -= lr * self.weight_decay * p.data
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (p.grad ** 2)
            m_hat = self.m[i] / (1.0 - self.beta1 ** t)
            v_hat = self.v[i] / (1.0 - self.beta2 ** t)
            p.data -= lr * m_hat / (np.sqrt(v_hat) + self.eps)


class Lion(Optimizer):
    def __init__(self, parameters, lr=1e-4, total_steps=None, beta1=0.9, beta2=0.99, weight_decay=1e-2):
        super().__init__(parameters, lr, total_steps)
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay

        self.m = [np.zeros_like(p.data) for p in self.parameters]

    def step(self, k):
        lr = self._current_lr(k)

        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue
            g = p.grad

            # Decoupled weight decay, like AdamW.
            if self.weight_decay != 0.0:
                p.data -= lr * self.weight_decay * p.data

            # Direction uses a beta1 interpolation.
            update = self.beta1 * self.m[i] + (1.0 - self.beta1) * g

            # Sign update.
            p.data -= lr * np.sign(update)

            # Momentum update uses beta2.
            self.m[i] = self.beta2 * self.m[i] + (1.0 - self.beta2) * g
