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
    """Plain RMSProp with an exponential moving average of squared gradients."""

    def __init__(
        self,
        parameters,
        lr=1e-3,
        total_steps=None,
        rho=0.99,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(parameters, lr, total_steps)

        if not 0.0 <= rho < 1.0:
            raise ValueError("rho must satisfy 0 <= rho < 1")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        self.rho = float(rho)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.square_avg = [np.zeros_like(p.data) for p in self.parameters]
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
            p.data -= lr * grad / denom


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

