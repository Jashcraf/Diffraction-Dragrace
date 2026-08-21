# The phase-retrieval board

Nonlinear-optimisation phase retrieval, following Jurling & Fienup (2014), JOSA A
31(7) 1348 — *Applications of algorithmic differentiation to phase retrieval
algorithms*. Their argument is that on a parametrised forward model the gradient
is the whole cost story: a finite-difference gradient costs `P+1` forward models
and an analytic one costs `O(1)`, independent of `P`. This board measures that
claim on six real optical-propagation codes.

```
parameters   theta in R^11   Noll 1..11, waves RMS, seeded
forward      obstructed pupil -> OPD = sum(theta_i Z_i) -> phasor
             -> propagate to focus -> I = |E|^2
loss         L(theta) = mean( ((I(theta) - I_obs)/s)^2 ),   s = max(I_obs)
solve        L-BFGS-B from a perturbed estimate of the truth
deliverable  theta_hat, and the wall time it took to get there
```

**The timed region is one complete retrieval**, not one propagation. Times are
seconds where every other board's are milliseconds, which is why `repeats` is 3
rather than 25.

## Two cases, because there are two questions

| case | gradient | codes |
|---|---|---|
| `pr_zernike11_numeric_scan` | scipy's own two-point finite differences | POPPY, lentil, PROPER, HCIPy |
| `pr_zernike11_analytic_scan` | the code differentiates its own forward model | prysm, dLux |

The two case files are **identical in every value except `retrieval.gradient`**,
and `tests/test_retrieval.py::test_the_two_boards_differ_only_in_gradient`
asserts that field by field. If the physics drifted apart, comparing the two
figures would silently stop meaning anything.

`numpy_baseline` runs on both. It is the floor — "what does this retrieval cost
if you just write it down" — not a competitor, and `plots.py` draws it as a
corridor rather than a seventh line.

- **prysm** uses its own reverse-mode API, `sum_of_2d_modes_adjoint` and
  `focus_dft_adjoint`, driven by the same `scipy.optimize.minimize` call the
  numerical board uses with `jac=True`.
- **dLux** uses `jax.value_and_grad` over the dLux optical system, minimised with
  `optax.lbfgs`, with the whole optimisation compiled into one XLA program.

## The optical system

Circular aperture, 30% linear secondary obscuration, two spider vanes at +30°
and −30°.

None of that is decoration:

- **The obscuration** breaks rotational invariance. An unobstructed circular
  pupil has a PSF that cannot separate the two members of a rotated Zernike
  pair, leaving the retrieval degenerate in the astigmatism, coma and trefoil
  planes.
- **The vanes are half-rays, not diameters** (`spider_span: radius`), and that
  choice decides whether the problem is solvable at all. See below.

## The twin ambiguity, and the two things needed to defeat it

A centro-symmetric pupil makes OPD `φ(x)` and `−φ(−x)` produce the **identical**
PSF, so a single in-focus image cannot separate them. Measured on a
centro-symmetric variant of this case, L-BFGS-B drives the loss from 1.5e−3 down
to 3.2e−9 and returns the exact twin of the truth — every mode of odd radial
order sign-flipped, coefficient error 1.27. The optimiser is not at fault; the
inverse problem has two equally good answers.

**1. Half-ray vanes.** A pupil with arms at +30° and −30° and nothing at +210°
and −210° is not centro-symmetric, which makes the truth the unique global
minimum: loss 0 there against 1.1e−6 at the twin. This is also the more literal
reading of "one vane at +30 and another at −30" — diameter-spanning bars would
draw *four* arms from two vanes.

**2. A perturbed starting estimate.** Step 1 alone is **not enough**, and the
measurement that shows it is worth recording:

| start | N=64 | N=128 | N=256 | N=512 | N=1024 |
|---|---|---|---|---|---|
| `zeros` | truth | truth | **twin** | **twin** | **twin** |
| `truth_perturbed` | truth | truth | truth | truth | truth |

1.1e−6 of symmetry breaking is invisible from a cold start's 1.5e−3, so which
basin the path enters is decided by details that move with the grid. Widening the
vanes until every size happened to work was tried and rejected — 0.05 D was
robust across the scan and 0.10 D broke at N=64, which is fitting the telescope
to the optimiser.

`initial: truth_perturbed` starts from a seeded estimate 25% away from the truth.
That is how fine phasing is actually deployed, and it buys the board more than
correctness: **the iteration count then holds at 21–23 at every size and on both
gradient paths**, so the runtime curve isolates the cost of *one forward model*
instead of confounding it with an iteration count that wanders between sizes and
between codes.

Jurling & Fienup break the ambiguity with phase-diversity images instead. This
board does not, because the question here is a timing one and a second plane
would double every runtime without changing what is being compared.

## What is pinned, and the one place idiom is overridden

Everything an optimiser could differ on is in the case file and applied by
`retrieval.minimise`: truth coefficients, starting guess, `ftol`, `gtol`,
iteration cap, L-BFGS history length. A timing comparison between two codes that
stopped on different criteria is not a comparison, and the failure is silent.

**The Zernike basis and the amplitude mask are supplied by the harness.** This is
a deliberate exception to "let each library do it its way". Every one of these
codes ships Zernikes — `poppy.zernike`, `prysm.polynomials`, `hcipy.mode_basis`,
`lentil.zernike`, `proper.prop_zernikes`, `dLux.ZernikeBasis` — and every one
normalises differently. If each rendered its own, `theta` would mean a different
wavefront in each code, so the six optimisers would be descending six different
landscapes and their iteration counts would not be comparable.

What each library **is** charged for is everything downstream of the OPD array:
rebuilding its optic, its wavefront, re-applying its mask, and the propagation
itself. That is where the real per-call differences live. Per the harness's
standing contract, anything the API permits hoisting is hoisted into `build()`
(POPPY's `OpticalSystem`, HCIPy's `FraunhoferPropagator`, prysm's executor,
lentil's `Pupil`) — and PROPER can hoist nothing, because it executes a
prescription end to end against module-level global state.

**Each code fits its own observed PSF**, generated by its own forward model at
the truth coefficients, untimed, in `build()`. One harness-generated measurement
fitted by all six would be more like real phase retrieval but would make the
board measure model error: PROPER reaches the focal plane by a near-field Fresnel
propagation through a lens and lands ~1e−4 from an exact Fourier transform, so it
would grind against an irreducible residual that says nothing about how fast its
optimisation is.

## Three gates

| gate | question | how it fails |
|---|---|---|
| `forward_accuracy` | is this code modelling the telescope the case describes? | untimed; compares the code's own PSF at the truth against the harness reference |
| `accuracy` | did the optimiser find the truth? | coefficient error over the observable modes |
| the twin check | did it find the *other* valid answer? | reported as `twin_rel_l2` so it is diagnosable, not just "inaccurate" |

The forward gate is not redundant. Because each code fits its own data, a code
with a subtly wrong pupil would converge beautifully onto its own private physics
and the coefficient gate would pass. This is the only check that would notice.

## Two things the board found

**POPPY carries the opposite OPD sign convention.** Its PSF at `+theta`
reproduces the harness reference at `−theta` to 3.4e−16 — exactly.
`exp(+2iπ·OPD/λ)` and `exp(−2iπ·OPD/λ)` are both in circulation and neither is
wrong; the forward board never noticed because `validate.compare` conjugates the
difference away, and an intensity comparison cannot, since squaring the modulus
destroys the sign. So the gate tests both hypotheses and reports which one held.
It does not reach the recovered coefficients — each code fits data its own model
produced, so both conventions recover the same `theta` — but it does set the sign
of any wavefront read out of that code.

**The prysm *adapter* crossed two Wirtinger conventions.** Not a prysm bug and
not a `numpy_baseline` bug — both are correct, and this is worth stating
carefully because the first draft of this file got it wrong.

For a real loss of a complex variable there are two self-consistent bookkeeping
conventions, and they are complex conjugates of one another:

| convention | tracks | a linear map `E = A(W)` backpropagates as |
|---|---|---|
| holomorphic | `dL/dz` | `Aᵀ`, plain transpose |
| conjugate | `dL/dz*` | `A^H`, conjugate transpose |

`numpy_baseline` is in the first throughout: it forms `Ebar = Ibar·conj(E)` and
uses a plain transpose. prysm's API is in the second: `executor.adjoint` applies
the conjugate transpose (`fttools.MDFT.adjoint`), so it must be handed
`dL/dE* = Ibar·E` — with `E`, not `conj(E)`, since `I = E·conj(E)` gives
`dI/dE* = E` — and the conjugation is taken at the phasor step instead.

Measured on this case, the two chains agree with central differences at
**4.43e−8 apiece**, their intermediate cotangents are **exact** conjugates
(difference identically `0.0`), and their parameter gradients are
**bit-identical**.

What was wrong was transcribing `numpy_baseline`'s `Ebar` into prysm's `A^H`,
which conjugates twice: that gradient is off by up to **68× per component**. The
symptom is not an error — L-BFGS-B partly absorbs a bad gradient into its step
length, so it *stalls*: 1 iteration, 44 function evaluations, a line search
failing over and over. On a timing board that would have been published as
"prysm is slow".

[gradient_board.md](gradient_board.md) already states the rule this broke — *the
intermediate complex cotangents differ between codes by a conjugation, so never
compare or transcribe them across codes* — which is exactly why the board is
defined at the parameter level, where both conventions agree. The same crossing
was present in the adapter's pre-existing gradient-board method, which had never
been exercised. Both are now pinned against central differences in
`tests/test_retrieval.py`.

## Piston is in the vector and is unobservable

Noll 1..11 is what "the first 11, up to primary spherical" means, and Noll 1 is
piston, which a PSF cannot see: `dL/dtheta_1` is identically zero — measured at
7.5e−20 through JAX's reverse mode. The retrieval leaves it wherever it started.
It stays in the parameter vector because that is what was asked for, and the
accuracy gate excludes it rather than charging every code for a mode none of them
could recover. `coefficient_rel_l2_all` reports the error with piston in.

## dLux was not single-threaded, and it reversed the result

The first version of this board reported dLux beating prysm at N=1024, 877 ms to
1353 ms. **It was using ~10 cores while prysm used 1.**

`threads: 1` in the config exports `OMP_NUM_THREADS` and friends, which
OpenBLAS, MKL and numexpr all honour — so prysm, lentil, HCIPy, POPPY and the
baseline genuinely ran on one core (measured cpu/wall = 1.00 for each). XLA
honours none of them, and the `XLA_FLAGS` the config emitted turned out to be
inert. Measuring process CPU time over the timed region, dLux at N_p=1024:

| condition | cores used | wall |
|---|---|---|
| the harness env as it was | 9.92 | 927 ms |
| no `XLA_FLAGS` at all | 10.06 | 909 ms |
| `--xla_cpu_multi_thread_eigen=false` | 10.04 | 909 ms |
| `--xla_cpu_use_thunk_runtime=false` + eigen off | 10.05 | 911 ms |
| **CPU affinity pinned to one core** | **1.00** | **2117 ms** |

Re-measured through the harness with the pin in place, dLux is slower than prysm
at **every** size, not just at N=1024 — the apparent crossover was the extra
cores:

| N_p | prysm | dLux | as previously published |
|---|---|---|---|
| 64 | 6.6 ms | 9.9 | 10.1 |
| 128 | 16.9 | 29.6 | 22.1 |
| 256 | 62.7 | 106.3 | 53.1 |
| 512 | 312.3 | 467.2 | 209.6 |
| 1024 | **1481.2** | **1989.1** | 877.4 |
See [methodology.md](methodology.md#threads-was-in-that-list-and-the-reason-given-was-wrong)
for what was wrong with the flags and how the axis is enforced now — CPU
affinity in `worker._pin_cpus`, plus a `cpu_wall_ratio` recorded on every timed
region so the label is checked rather than asserted.

Two things worth keeping from this. First, in CPU-seconds dLux spends **9.4 s
against prysm's 1.8 s** for the same retrieval — about 5× more total work, which
is a real difference between a compiled XLA program and a hand-written adjoint
and is invisible on a wall-clock axis. Second, an unpinned dLux row is not
worthless, it just answers a different question: *what does this cost with the
machine's cores available*, which is what a user actually gets by default. Both
are legitimate; publishing one under the other's label is not.

## One deviation worth naming

**dLux is driven by `optax.lbfgs`, which is L-BFGS, not L-BFGS-B.** optax has no
box-constrained variant. Nothing on this board is bounded and no bounds are
passed to scipy either, so the only thing the `-B` adds — a projection onto
bounds that do not exist — is absent from both. On this problem they are the same
algorithm. They are still different implementations with different line searches,
and a reader comparing prysm's row against dLux's is comparing those too.

The adapter transcribes scipy's three stopping tests rather than approximating
them, so "time to converge" means the same thing on both boards:

```
gtol   max |g_i| <= gtol
ftol   (f_k - f_k+1) / max(|f_k|, |f_k+1|, 1) <= ftol
cap    maxiter
```

Without this the comparison would be silent nonsense: optax's default stopping is
nothing at all, so dLux would run the full iteration cap while the scipy-driven
codes stopped at 22, and would look slow for having done four times the work.

## Reading the figure

Wall time is `(forward-model evaluations) × (cost per evaluation)`, and the two
factors are independent. `dragrace report` prints both, plus `ms/fev`, because a
code can lose on total time while winning per evaluation — a completely different
finding from being slow.

There is **no ideal-FLOP line** on these figures. The per-evaluation floor *is*
derivable and `ideal_work` computes it into the result's `detail` string, but how
many evaluations the optimiser needs is measured, not predicted — it depends on
the line search, the L-BFGS history and the curvature the code happens to meet.
Multiplying a real per-evaluation floor by a guessed evaluation count would
produce a total carrying the authority of the first and the reliability of the
second. `flops = 0` is what suppresses the line and the table row.

## The GPU legs

Both cases also run under `gpu_f64` — POPPY on the numerical board, prysm and
dLux on the analytic one — at complex128 on both devices, so the device is the
only thing that moves. `scripts/plot_retrieval_cpu_gpu.py` draws the six series
as one figure (`docs/figures/pr_zernike11_cpu_vs_gpu.png`); `dragrace plot`
will not, because the figure crosses two cases and two configs, which are
exactly the axes that command refuses to merge. The crossing is legitimate here
only because the two cases are one optical system, one truth wavefront, one
starting theta and one set of stopping tests.

Three pieces of harness had to change for it, each worth knowing about:

- **`Adapter.retrieval_devices`.** The board used to refuse every GPU config
  from `retrieval_support`. That was right when no retrieval chain was
  device-aware and wrong afterwards, so the refusal is now per adapter. An
  adapter whose forward model is host NumPy still declines rather than
  publishing a row labelled `gpu_f64` that never touched the device.

- **`Config.conda_env_by_adapter`.** `gpu_f64` needs two environments —
  `dragrace-gpu-cupy` for POPPY and prysm, `dragrace-gpu-jax` for dLux, because
  pip's `jax[cuda12]` wheels vendor their own CUDA runtime and must not share an
  interpreter with conda's. Splitting the *config* instead would have put
  dLux-on-CUDA and prysm-on-CUDA on separate boards, which is the comparison
  the GPU configs exist to make.

- **`runner.cuda_path_for`.** conda-forge puts the CUDA headers under
  `$PREFIX/targets/<arch>/`, CuPy appends `/include` to `CUDA_PATH`, and CuPy
  needs those headers to NVRTC-compile its elementwise kernels at first use.
  The runner launches workers as `$PREFIX/bin/python`, so `activate.d` hooks
  never fire and the variable has to be set by the runner.

### What the figure says

At `N_p = 1024` the GPU is worth **37×** to prysm and **45×** to dLux, and
**3.5×** to POPPY. At `N_p = 64` it is worth nothing at all — prysm and POPPY
are marginally *slower* on the device — because at that size every code is
launch-bound and the retrieval is a few hundred sub-millisecond kernels behind
a host-side optimiser.

That POPPY gains least is not a POPPY result. Its 300–324 forward models per
retrieval are 12× the analytic codes' 25–28, and on the GPU that count buys
12× as many launches over kernels too small to hide them: the numerical
gradient costs more *and* it is the part the device helps least with. Panel (b)
divides the count out and shows POPPY's per-evaluation curve behaving like
everyone else's.

dLux is the only code here whose optimiser runs on the device — the whole
L-BFGS loop is inside `jax.lax.while_loop`, so nothing crosses the host
boundary per iteration, while prysm and POPPY hand scipy a host scalar and a
host gradient every evaluation. At these problem sizes that architectural
difference is not what decides the figure; at smaller ones it is why dLux is
the only code faster on GPU than CPU at `N_p = 64`.
