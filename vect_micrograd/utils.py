"""Utility functions for micrograd-vect.

Loss functions and model checkpointing.
"""

import numpy as np

from vect_micrograd.vect_engine import Value

def sample_batch(X, y, batch_size, replace=True):
    """Sample a minibatch explicitly.

    Args:
        X: input array.
        y: target array.
        batch_size: number of examples.
        replace: whether to sample with replacement.

    Returns:
        Xb, yb
    """
    n = X.shape[0]

    if batch_size is None or batch_size >= n:
        return X, y

    if replace:
        idx = np.random.randint(0, n, size=batch_size)
    else:
        idx = np.random.choice(n, size=batch_size, replace=False)

    return X[idx], y[idx]

def one_hot(y, classes):
    """Convert an integer label array to a one-hot matrix.

    Args:
        y:       1-D integer array of class indices, shape (batch,).
        classes: total number of classes.

    Returns:
        NumPy float array of shape (batch, classes).
    """
    out = np.zeros((len(y), classes), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out

# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def cross_entropy_loss(model, X, targets, alpha=0.0):
    """Cross entropy loss for one-hot or sparse integer class labels.

    Args:
        model: model returning logits of shape (batch, classes).
        X: input array.
        targets: either one-hot targets of shape (batch, classes) or integer
                 class labels of shape (batch,).
        alpha: L2 regularization strength.

    Returns:
        loss: scalar Value suitable for loss.backward().
        accuracy: float in [0, 1].
    """
    logits = model(Value(X, requires_grad=False))
    targets_arr = np.asarray(targets)

    if targets_arr.ndim == 1:
        loss, probs = logits.softmax_ce_sparse(targets_arr)
        pred = probs.argmax(axis=1)
        accuracy = float((pred == targets_arr).mean())
    else:
        loss, probs = logits.softmax_ce(targets_arr)
        pred = probs.argmax(axis=1)
        accuracy = float((pred == targets_arr.argmax(axis=1)).mean())

    if alpha:
        loss = loss + alpha * sum((p * p).sum() for p in model.parameters())
    return loss, accuracy

def svm_loss(model, X, y, alpha=1e-4):
    """L2-regularized SVM max-margin loss.

    This function is deterministic: it computes the loss on exactly the
    X and y passed to it. It does not sample minibatches internally.

    Args:
        model: MLP or any Module.
        X:     input array of shape (n, features).
        y:     target array of shape (n, 1) with values in {-1, +1}.
        alpha: L2 regularization strength.

    Returns:
        loss:     scalar Value suitable for loss.backward().
        accuracy: float in [0, 1].
    """
    scores = model(Value(X, requires_grad=False))
    margins = (1 + scores * (-y)).relu()
    data_loss = margins.mean()
    reg_loss = alpha * sum((p * p).sum() for p in model.parameters())
    accuracy = float(np.mean((y > 0) == (scores.data > 0)))
    return data_loss + reg_loss, accuracy



# ---------------------------------------------------------------------------
# Flow training helpers
# ---------------------------------------------------------------------------

def save_state(model):
    """Return a copy of all model parameters.

    This lightweight state format is used by the example notebooks to keep
    the best validation checkpoint without introducing serialization code.
    """
    return [param.data.copy() for param in model.parameters()]


def load_state(model, state):
    """Restore parameters previously produced by :func:`save_state`.

    Args:
        model: Model with a ``parameters()`` method.
        state: Sequence of NumPy arrays returned by ``save_state(model)``.

    Raises:
        ValueError: if the number of arrays does not match the model.
    """
    params = list(model.parameters())
    if len(params) != len(state):
        raise ValueError(f"state has {len(state)} arrays but model has {len(params)} parameters")

    for param, saved in zip(params, state):
        param.data[...] = saved


def full_nll(model, X, batch_size=2048):
    """Compute the average negative log-likelihood over ``X`` in batches."""
    total = 0.0
    n_total = len(X)
    if n_total == 0:
        raise ValueError("X must contain at least one example")

    for start in range(0, n_total, batch_size):
        xb = X[start : start + batch_size]
        n = len(xb)
        total += model.nll(xb).item() * n

    return total / n_total


def grad_global_norm(parameters, fail_on_non_finite=True):
    """Return the global L2 norm of all available parameter gradients.

    Args:
        parameters: Iterable of ``Value`` parameters.
        fail_on_non_finite: If True, raise ``FloatingPointError`` when a
            gradient contains NaN or infinity. If False, return ``np.nan``.
    """
    total_sq = 0.0

    for i, param in enumerate(parameters):
        if param.grad is None:
            continue

        grad64 = param.grad.astype(np.float64, copy=False)
        if not np.all(np.isfinite(grad64)):
            msg = f"non-finite gradient in parameter {i} with shape {param.data.shape}"
            if fail_on_non_finite:
                raise FloatingPointError(msg)
            return np.nan

        total_sq += float((grad64 ** 2).sum())

    return total_sq ** 0.5


def clip_grad_norm(parameters, max_norm=5.0):
    """Clip gradients to a maximum global L2 norm.

    Returns the unclipped norm, matching PyTorch's ``clip_grad_norm_`` style.
    """
    params = list(parameters)
    total_norm = grad_global_norm(params, fail_on_non_finite=True)
    scale = min(1.0, max_norm / (total_norm + 1e-12))

    if scale < 1.0:
        for param in params:
            if param.grad is not None:
                param.grad *= scale

    return total_norm


def init_flow_near_identity(model, output_scale=0.01):
    """Initialize coupling transforms near the identity map.

    Coupling layers with a ``net`` attribute have their final dense layer scaled
    down. If the model exposes a final ``scaling`` transform, its log-scales are
    reset to zero. This is useful for the toy flow notebooks.
    """
    for layer in model.flow:
        if hasattr(layer, "net"):
            last = layer.net.layers[-1]
            last.W.data *= output_scale
            if last.b is not None:
                last.b.data.fill(0.0)

    if hasattr(model, "scaling"):
        model.scaling.log_s.data.fill(0.0)

# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(model):
    """Save a snapshot of the model's current parameters.

    Captures a copy of each parameter's data at the current training step.
    Typical use is to record the best model seen so far during training:

        best_checkpoint = None
        best_loss = float('inf')

        for k in range(total_steps):
            loss, acc = compute_loss()
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_checkpoint = save_checkpoint(model)

    Args:
        model: an MLP or any Module whose parameters() returns an iterable
               of Value.

    Returns:
        A list of NumPy arrays, one per parameter, in the order returned
        by model.parameters().
    """
    return [p.data.copy() for p in model.parameters()]


def load_checkpoint(model, checkpoint):
    """Restore model parameters from a previously saved checkpoint.

    Overwrites each parameter's data with the saved values and zeroes
    the gradients, leaving the model ready for inference or continued
    evaluation.

    Args:
        model:      the same model architecture used when saving.
        checkpoint: the list returned by save_checkpoint().

    Raises:
        ValueError: if the checkpoint length does not match the number of
                    model parameters, which usually means the checkpoint was
                    saved from a different architecture.
    """
    params = list(model.parameters())
    if len(params) != len(checkpoint):
        raise ValueError(
            f"checkpoint has {len(checkpoint)} parameter arrays "
            f"but model has {len(params)}"
        )
    for p, data in zip(params, checkpoint):
        p.data = data.copy()
        p.zero_grad()
