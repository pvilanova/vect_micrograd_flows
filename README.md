# vect_micrograd normalizing flows

A small, didactic implementation of normalizing flows using the vectorized NumPy
micrograd engine (http://github.com/pvilanova/vect_micrograd_plus).

The project implements two closely related flow families:

- **NICE** — additive coupling layers plus a final diagonal scaling layer.
- **Real NVP** — affine coupling layers with input-dependent log-determinants.

The goal is educational: keep the implementation compact enough to inspect, while
still supporting exact log-likelihood, exact latent inference, exact sampling, and
reverse-mode autodiff.

## Why both NICE and Real NVP?

NICE introduced the coupling-layer idea in a form that is very easy to understand:
part of the input is copied unchanged, and the other part is shifted by a neural
network conditioned on the copied part.

```text
y_id     = x_id
y_change = x_change + t(x_id)
logdet   = 0
```

Because additive coupling is volume-preserving, NICE adds a final learned diagonal
scaling layer:

```text
z = exp(s) * h
log |det J| = sum_i s_i
```

Real NVP extends the coupling layer from additive to affine:

```text
y_id     = x_id
y_change = x_change * exp(s(x_id)) + t(x_id)
logdet   = sum_j s_j(x_id)
```

That extra scale network gives the model input-dependent local compression and
expansion. In practice, Real NVP-style affine coupling usually fits toy densities
such as two moons better than additive NICE.

## Design

The code is intentionally organized around generic flow components, not around a
single monolithic `NICE` class.

```text
base distribution + invertible transform stack = normalizing flow
```

Core pieces:

- `StandardNormal`
- `StandardLogistic`
- `AdditiveCoupling` for NICE-style coupling
- `AffineCoupling` for Real NVP-style coupling
- `DiagonalScaling`
- `Reverse`
- `Permute`
- `FlowSequential`
- `NormalizingFlow`
- `make_nice_flow(...)`
- `make_realnvp_flow(...)`
- `make_flow(kind=...)`

Every transform follows the same interface:

```python
z, logdet = transform.forward(x)
x = transform.inverse(z)
params = transform.parameters()
```

`NormalizingFlow` supplies the density-model API:

```python
logp = model.log_prob(x)  # per-example log p(x), shape (batch,)
loss = model.nll(x)       # mean negative log-likelihood, scalar Value
x = model.sample(1024)    # NumPy array sampled through the inverse flow
```

## Quickstart

```bash
pip install -e .
python -m pytest -q
```

## NICE example

```python
from vect_micrograd.flows import make_nice_flow
from vect_micrograd.optim import Adam

model = make_nice_flow(
    dim=2,
    hidden_sizes=(64, 64),
    num_coupling_layers=4,
    prior="normal",
)

optimizer = Adam(model.parameters(), lr=5e-4)
```

This builds a paper-faithful NICE-style flow:

```text
AdditiveCoupling -> AdditiveCoupling -> ... -> DiagonalScaling -> prior
```

## Real NVP example

```python
from vect_micrograd.flows import make_realnvp_flow
from vect_micrograd.optim import Adam

model = make_realnvp_flow(
    dim=2,
    hidden_sizes=(64, 64),
    num_coupling_layers=6,
    prior="normal",
    max_log_scale=2.0,
)

optimizer = Adam(model.parameters(), lr=5e-4)
```

This builds a Real NVP-style flow:

```text
AffineCoupling -> AffineCoupling -> ... -> prior
```

A final `DiagonalScaling` can also be added to Real NVP-style models with:

```python
model = make_realnvp_flow(
    dim=2,
    hidden_sizes=(64, 64),
    num_coupling_layers=6,
    use_final_scaling=True,
)
```

## Generic builder

The notebooks use the generic builder so the same training code can switch
between NICE and Real NVP:

```python
from vect_micrograd.flows import make_flow

flow_kind = "realnvp"  # or "nice"

model = make_flow(
    kind=flow_kind,
    dim=2,
    hidden_sizes=(64, 64),
    num_coupling_layers=6,
    prior="normal",
    max_log_scale=2.0,
)
```

For `kind="nice"`, pass only arguments accepted by `make_nice_flow`. For
`kind="realnvp"`, pass Real NVP-specific options such as `max_log_scale`.

## Minimal training loop

```python
import numpy as np

from vect_micrograd.flows import make_realnvp_flow
from vect_micrograd.optim import Adam

rng = np.random.default_rng(0)
X = rng.standard_normal((4096, 2)).astype(np.float64)

model = make_realnvp_flow(dim=2, hidden_sizes=(64, 64), num_coupling_layers=4)
optimizer = Adam(model.parameters(), lr=5e-4, weight_decay=0.0)

for step in range(2000):
    idx = rng.integers(0, len(X), size=256)
    loss = model.nll(X[idx])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step(step)

samples = model.sample(1024)
print("final minibatch NLL:", float(loss.data))
print("sample shape:", samples.shape)
```

## Demo notebooks

- `flow_two_moons_micrograd.ipynb`
- `flow_8_gaussians_micrograd.ipynb`

The notebooks plot:

1. The toy dataset.
2. Mini-batch and full-data NLL curves.
3. Real samples vs. generated samples.
4. Learned relative log-density on a 2-D grid.

Use:

```python
flow_kind = "nice"
```

to run the additive NICE model, and:

```python
flow_kind = "realnvp"
```

to run the affine Real NVP-style model.

## Project layout

```text
vect_micrograd/
  vect_engine.py   # vectorized reverse-mode autodiff Value
  vect_nn.py       # Module, Layer, MLP helpers
  flows.py         # priors, transforms, FlowSequential, NormalizingFlow, builders
  optim.py         # SGD, Adam, Lion

tests/
  test_value.py
  test_flows.py

flow_two_moons_micrograd.ipynb
flow_8_gaussians_micrograd.ipynb
```
## References

This project is based on the normalizing-flow view of density estimation: learn an
invertible transformation between data space and a simple latent distribution, while
keeping both sampling and exact log-likelihood evaluation tractable.

- Laurent Dinh, David Krueger, and Yoshua Bengio.  
  **NICE: Non-linear Independent Components Estimation.**  
  arXiv:1410.8516, 2014.  
  https://arxiv.org/abs/1410.8516

- Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio.  
  **Density Estimation using Real NVP.**  
  arXiv:1605.08803, 2016.  
  https://arxiv.org/abs/1605.08803

- Danilo Jimenez Rezende and Shakir Mohamed.  
  **Variational Inference with Normalizing Flows.**  
  ICML 2015.  
  https://arxiv.org/abs/1505.05770

