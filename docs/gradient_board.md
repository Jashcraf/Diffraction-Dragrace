# The gradient board: prysm vs dLux

Compares a **hand-written adjoint** against **automatic differentiation** on
identical physics.

## Problem definition

```
parameters   theta in R^P    Noll Zernikes, waves RMS, seeded
forward      circular pupil -> basis-rendered OPD -> phasor -> MFT to focus
             -> intensity I = |E|^2
loss         L(theta) = mean((I - I_target)^2),  I_target = unaberrated PSF
deliverable  g = dL/dtheta in R^P
```

Both θ and L are real, so **the returned gradient is unambiguously real and the
two codes are directly comparable with no Wirtinger reconciliation.**

That is the design's whole trick. The intermediate complex cotangents *will*
differ between prysm's convention and JAX's — by a conjugation, possibly a
factor of 2, depending on whether the code carries ∂L/∂z or ∂L/∂z̄. Defining
the board at the parameter level makes that difference invisible and irrelevant.
**Never compare intermediates across the two codes.**

Using intensity rather than the complex field in the loss is deliberate: `|·|²`
is precisely the node where the conventions diverge, so it belongs inside the
tested path.

## prysm: manual reverse mode

Verified present in prysm v0.22 (`prysm.propagation` is a subpackage; PyPI's
0.21.1 predates this):

| step | forward | reverse |
|---|---|---|
| 1 | `sum_of_2d_modes(basis, theta)` | `sum_of_2d_modes_adjoint(basis, phsbar)` |
| 2 | `W = amp·exp(2πi·phs)` | `phsbar = -4π·Im(Wbar·W)` |
| 3 | `focus_dft(W, executor)` | `focus_dft_adjoint(Ebar, executor)` |
| 4 | `I = |E|²` | `Ebar = Ibar·conj(E)` |
| 5 | `L = mean((I-I_t)²)` | `Ibar = 2(I-I_t)/n` |

Steps 1 and 3 are prysm's own API. Steps 2 and 4 are pure arithmetic and are
written out.

The derivation for steps 2 and 4, since the factors are easy to get wrong:

```
I = E·E*        =>  dI/dθ = 2·Re(E*·dE/dθ)
=>  dL/dθ = 2·Re( Σ Ebar·dE/dθ )  with  Ebar = Ibar·conj(E)

E = scale·Kx·W·Kxᵀ is linear
=>  Wbar = scale·Kxᵀ·Ebar·Kx     (plain transpose -- the conjugation is
                                  already carried in Ebar)

W = amp·exp(2πi·phs)  =>  dW = W·2πi·dphs
=>  dL/dθ = 2·Re(Σ Wbar·W·2πi·dphs) = -4π·Σ Im(Wbar·W)·dphs
```

`adapters/numpy_baseline/adapter.py` implements exactly this chain, and
`tests/test_baseline.py::test_gradient_matches_finite_differences` confirms it
against central differences at `max_rel_err = 2.1e-9`, cosine similarity
`1.000000000000000`, scale ratio `1.000000000`.

Properties that matter for the board: **no tape, no tracing, no compile step.**
The adapter chooses explicitly which forward intermediates to retain, so
prysm's adjoint memory is minimal and predictable — roughly `16·(N_p² + N_f²)`
bytes over the forward pass. Since the adjoint of an MDFT is another MDFT, the
theoretical cost is backward ≈ forward, i.e. **total ≈ 2× forward, independent
of P**. The ledger should show exactly 4 GEMMs. If it shows 6, the chain is
wrong.

## dLux: JAX autodiff

`jax.value_and_grad(loss)`, AOT-compiled in `build()`:

```python
compiled = jax.jit(jax.value_and_grad(loss)).lower(theta).compile()
```

so the timed region is pure execution and compile time is a measured number
rather than something hidden in a warm-up. Two things this unlocks that nothing
else in the suite offers:

- `compiled.cost_analysis()` — XLA's exact FLOP count for the **gradient**,
  giving `flops(grad)/flops(forward)` measured rather than assumed.
- `compiled.memory_analysis()` — exact buffer sizes, showing what XLA chose to
  materialise versus rematerialise. That is a genuinely different memory/compute
  tradeoff from prysm's explicit retention, and these let you say so
  quantitatively rather than speculatively.

## What the board reports

| metric | why |
|---|---|
| `t_grad / t_forward` | empirical reverse-mode overhead; theory says ~2–3 for both |
| `flops_grad / flops_forward` | **the headline**: does autodiff cost more arithmetic than a hand-written adjoint? |
| peak memory, grad vs forward | tape/checkpointing cost: prysm explicit, XLA-decided |
| scaling in P (1 → 15 → 1024) | must be ~flat for both; any P-dependence means forward mode or recomputation |
| scaling in N | where each adjoint becomes arithmetic-bound |
| compile time | large for dLux's gradient, zero for prysm |

The **P-sweep is the primary figure**. Reverse mode's entire value proposition is
P-independence, so a flat line for both codes against a linearly-rising
finite-difference baseline is the plot that makes the board legible to someone
who does not already think in adjoints.

## Correctness gate

Central differences at N=256, P=15, complex128:

```
g_i ≈ (L(θ + h·e_i) − L(θ − h·e_i)) / 2h,    h = 1e-6
```

Gate: per-component relative error `< 1e-6` **and** cosine similarity
`> 1 − 1e-9`. The cosine check is not redundant — a uniform factor of 2 (the
classic Wirtinger slip) can hide under a loose per-component tolerance while
showing up immediately in `scale_ratio`.

Complex-step differentiation is unavailable here because the model is already
complex-valued, so central differences with a carefully chosen `h` is the
available ground truth. **dLux at its default complex64 will not pass this
gate**; run it with `JAX_ENABLE_X64=1` even when the timing board runs single
precision.

## The confound

**prysm's adjoint chain is written by the benchmark author; dLux's is generated
by JAX.** A clumsy chain penalises prysm, an over-tuned one flatters it — and
either way the comparison is partly "hand-written adjoint vs automatic
differentiation", which is interesting but is *not* the same claim as "prysm vs
dLux". Three structural mitigations:

1. The chain uses prysm's own `*_adjoint` API rather than reimplementing it.
2. The expected primitive count is asserted against the ledger, so an
   accidentally inefficient chain fails rather than quietly producing a slow
   number.
3. Each adapter carries `reviewed_by`. Invite Brandon Dube and Louis Desdoigts
   to review their respective adapters before publishing. A benchmark the
   maintainers have signed off on is a contribution; one they have not is an
   argument.

Report `t_grad/t_forward` alongside absolute `t_grad`: the ratio factors out the
forward-propagation differences already covered by the main board, isolating the
differentiation machinery, which is what this board is actually about.
