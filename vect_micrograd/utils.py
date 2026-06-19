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
