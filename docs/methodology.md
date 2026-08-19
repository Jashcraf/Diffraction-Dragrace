# Methodology

What is measured, what is deliberately excluded, and what confounds remain.
This is the document to read before drawing a conclusion from any table this
repo produces.

## What is timed: the measurement contract

Every result carries a `measurement_contract` field, because the same adapter
name can measure two quite different things and nothing else in the file would
say which.

**`idiomatic-v1` (current).** The timed call is the one the library's own
documentation puts in front of a user, with everything the API permits hoisting
already hoisted into `build()`:

| code | untimed (`build()`) | timed (`propagate()`) |
|---|---|---|
| POPPY | `OpticalSystem` + planes | `calc_psf(wavelength)` |
| prysm | `Wavefront`, `prepare_executor` | `wf.focus_dft(executor)` |
| lentil | `lentil.Pupil(...)` | `Wavefront(λ) * pupil`, `propagate_dft(...)` |
| dLux | `AngularOpticalSystem`, jit+compile | `optics.propagate_mono(λ)` |
| HCIPy | `Wavefront`, `FraunhoferPropagator` | `prop(wf)` |

**`primitive-v1` (superseded).** The timed call was the library's transform
entry point — `poppy.matrixDFT.perform`, `lentil.fourier.dft2`, a hand-written
`jnp` kernel for dLux. That measures a transform kernel, not a library, and it
reports a POPPY nobody runs: at N=1024, `matrixDFT.perform()` takes 20 ms while
`calc_psf()` takes 30 (66 ms under the 1-thread benchmark config), so nearly
half of what a user waits for lives outside the transform, in `astropy.units`
handling, normalisation and FITS HDU construction.

`dragrace report` groups by contract and refuses to put the two in one table,
exactly as it refuses to merge machine fingerprints.

### The hoisting rule, and what it charges

"Everything the API permits hoisting" is doing real work in that sentence, and
it is not the same for every code:

- POPPY's `OpticalSystem` is reusable, so it is built once. But `calc_psf` calls
  `input_wavefront()` and **re-applies every optic on every invocation**
  (`poppy_core.propagate_mono`), so per-call model application is inside the
  clock — because POPPY gives a user no way to hoist it.
- lentil's `Pupil` is reusable and hoisted; its `Wavefront` is *consumed* by the
  propagation, so `Wavefront(λ) * pupil` is per-PSF work and is timed. That
  costs it: 19 ms for `propagate_dft` alone at N=1024 against 45 ms for the
  documented sequence.
- prysm's `Wavefront` survives propagation, so it is hoisted. prysm's OO layer
  then costs nothing measurable — 16.5 ms via `Wavefront.focus_dft` against
  16.8 ms for the module-level function on a raw array.

The consequence to state wherever these results appear: **the forward board is
no longer propagators-only.** It measures one PSF through each library's
documented path, and a code that rebuilds its wavefront per call is charged for
it. That is a property of the API, not an inefficiency in the transform, and the
two must not be conflated in a conclusion.

**What is still pinned.** The aperture is the harness's hard-edged mask, injected
into each library's own model object (`ArrayOpticalElement`, `dLux.Optic`,
`lentil.Pupil`), never the library's own `CircularAperture`. Antialiasing is a
choice each code makes differently and an antialiased edge costs more to render
than a hard one; letting adapters differ there would put rasterisation cost into
a propagation comparison. Grid *centring* is the one thing not pinned — see
[conventions.md](conventions.md).

**Gradient board (`kind: gradient`).** Unchanged by the contract: the basis
render and phasor are timed, because they are part of the differentiated chain,
and how each framework handles them (cached, rematerialised, fused) is part of
what the board measures.

**`build()` is untimed but not unmeasured.** Grids, DFT kernels, FFT plans, JIT
compilation and device staging happen there — anything a real user would hoist
out of a loop. It is reported separately as `setup.build_s` and
`setup.first_call_s`, and the amortisation curve (`report.amortisation`)
combines them: `T(k) = import + build + first_call + k·steady`. The lines cross,
and where they cross is the practically useful answer.

## The three rules

1. **`sync()` is inside the clock.** JAX dispatches asynchronously and CuPy
   launches kernels asynchronously; without a blocking sync, `propagate()`
   returns before any arithmetic has happened and reports a ~100× speedup that
   is entirely dispatch latency. `metrics.check_sync_scaling` flags any adapter
   whose time fails to grow with problem size.
2. **Timing runs are never traced.** VizTracer's overhead is per-Python-call, so
   it penalises loop-heavy codes far more than vectorised ones. Traced results
   carry `traced: true` and are excluded from every comparison table.
3. **Device→host transfer is measured separately.** Reported as
   `host_available` alongside `device_compute`. Conflating them is how GPU
   benchmarks mislead in both directions.

A fourth, added with the idiomatic contract: **the accuracy gate never inflates
the timed call.** POPPY's `calc_psf` and dLux's `propagate_mono` return
*intensity*, so gating on what they return would discard the phase-sign and
normalisation diagnostics that make a cross-code comparison meaningful. Both
libraries document a way to get the complex field back — `return_final=True`,
`return_wf=True` — and the adapter's `complex_field()` uses it in a separate,
untimed call. It costs ~15% more than the plain call for POPPY, and that 15%
must not land in the measurement of what a user pays.

## Accuracy gates timing

A run that fails its accuracy gate produces no timing row. Without this, a code
that is fast because it is doing something less accurate wins.

Two references, and the distinction matters:

- **`internal_mft`** — a direct double-precision matrix DFT sharing the case's
  discretisation. Agreement to ~1e-13 is expected. This is the gate.
- **`analytic_airy`** — the closed form for an unaberrated circular pupil.
  Reported but **ungated**, because it is the *continuous*-aperture answer while
  every adapter propagates a pixelated hard-edged mask. The measured
  disagreement is ~8.8e-4 at N=1024. Gating on it would fail everyone.

A known weakness: for a `matrix_dft` case the internal reference performs the
same operations as an MFT adapter, so agreement there is close to trivial. The
genuinely independent constraints are the analytic check and the cross-algorithm
comparison — the FFT and MFT paths reach identical focal samples by completely
different routes and agree to <1e-12 (`tests/test_baseline.py`).

Normalisation and phase-sign conventions differ legitimately between these
codes. The comparison fits a single complex scale factor by least squares and
tries the conjugate, then **reports** both (`scale_abs`, `conjugated`) rather
than discarding them. A code whose normalisation is 4× off is not wrong, but it
is worth knowing.

## When the gate can only check intensity

`validate.compare` gates the complex field, and absorbs exactly two legitimate
convention differences: a global complex scale (normalisation) and conjugation
(phase sign). It cannot absorb a *quadratic* phase, and PROPER produces one.

PROPER reaches the focal plane by propagating through a lens (`prop_lens` then
`prop_propagate`) and tracking a reference sphere, rather than assuming the
Fraunhofer limit. Decomposing its 6.1e-3 field residual on `fft_n1024_q4`:

| component | value |
|---|---|
| amplitude vs the reference | **9.1e-8** |
| residual phase, peak-to-peak | 0.157 rad |
| fit of that phase against r² | −3.06e-4 rad per (λF/D)² — quadratic, i.e. defocus-like |

So its PSF is right to a part in ten million and only its phase convention
differs. Relevant to what follows: `prop_end` returns **intensity** by default;
the complex field requires `NOABS=True`. PROPER's documented output is a PSF.

An adapter may therefore declare `output_quantity = "intensity"`, which gates
|E|² against |ref|² with a fitted real scale. PROPER scores 1.79e-8 that way.

**What is given up is recorded, not hidden.** Squaring the modulus destroys the
phase, so `conjugated` and `scale_phase_rad` come back `null` and
`accuracy.quantity` reads `intensity`. Such a row must not feed a
phase-sensitive claim. This is a per-adapter *declaration* precisely so it can
never become a fallback the harness reaches for when a field comparison fails —
which would turn the gate into a formality.

`fft_array_scan` gates at 1e-6 rather than 1e-10 for the same physical reason: a
code that propagates through a lens accumulates roundoff through large phase
factors and lands near 1e-8, where an FFT-of-the-pupil lands at 1e-15. The gate
is a floor for "did you compute the right thing", not a ranking — every point
records its own `rel_l2`, and the seven-order gap between 1e-15 and 1e-8 is
itself worth reading as the price of the more general propagator.

## Backend axes a code does not have

The config fixes "the machinery to compute with" — device, FFT library, BLAS,
threads. Not every code has those knobs. dLux runs on XLA, which emits its own
kernels: there is no FFT library to select and no BLAS to point at, and the same
is true of prysm once its array module is pointed at `jax.numpy`.

Three options were available and two are wrong. Refusing to run dLux on a NumPy
config keeps the fastest code in the suite off the board for a reason that has
nothing to do with its physics. Running it silently puts an XLA row next to
OpenBLAS rows with nothing saying so. So the adapter **declares** the axes it
cannot honour (`config_axes_not_selectable`), the run proceeds, and every result
records it:

```
"axes_not_selectable": ["fft", "blas", "threads"]
"warnings": ["FFT: requested 'numpy', adapter resolved 'xla' -- this adapter has
              no selectable FFT backend, so the config's FFT axis did not apply",
             "BLAS axis is not selectable ... not comparable along the BLAS axis",
             "thread axis is not verifiable ... a label, not a measured property"]
```

`dragrace report` prints these per (adapter, config) under BACKEND AXES THAT DID
NOT APPLY, and `dragrace plot` puts them in the figure caption — with the
stronger "not a like-for-like backend comparison" line only when some lines run
a different engine from the rest. **A row with a declared axis is a valid
measurement of the code and not a data point along that axis.**

The declaration depends on the config, not just the adapter: dLux cannot honour
`fft=numpy`, but on `cpu_xla_1t` the config and the adapter agree and only
`blas`/`threads` remain inapplicable.

### Why `threads` is in that list

`threads` is declared for a different reason from `fft` and `blas`. XLA *does*
have a thread pool, but threadpoolctl cannot see it — it reports NumPy's idle
OpenBLAS sitting in the same process, which says nothing about the code being
measured. `Config.jax_env()` emits
`XLA_FLAGS=--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=N`
before the first `import jax`, but at N_p=2048 that produced 30.2 ms at 1 thread
and 30.0 ms at 8 — indistinguishable. So the harness cannot currently prove the
thread count either way for an XLA-backed row, and says so rather than letting a
false reading stand in as evidence.

## The XLA board

`cpu_xla_1t` exists so the backend axis can be taken *out* of the comparison:
every adapter that runs there executes through the same XLA kernels, so a
remaining difference is a difference in what the library asks XLA to do.

| adapter | on XLA? | how |
|---|---|---|
| dLux | natively | JAX is what dLux is |
| prysm | yes | `prysm.mathops` is a backend shim; pointing its array module at `jax.numpy` works unmodified (rel_l2 = 1.3e-15) |
| HCIPy, POPPY, lentil, PROPER | no | their backend layers offer numpy/scipy/mkl/fftw/cupy only; they report `unsupported` rather than run on NumPy under an XLA label |

Read `cpu_xla_1t` against `cpu_numpy_1t` to separate "what the library asks for"
from "what the backend does with it" — prysm is ~3.4× faster at N=1024 on XLA
than on OpenBLAS (9.3 ms against 31.8) for the identical prysm code path. Never
plot the two configs on one axis.

**A caution on the absolute numbers.** prysm-on-XLA reaches ~150 GFLOP/s
complex128 at N_p=2048, which is above what a single core should manage with
NEON alone; the likely explanation is that XLA routes through Apple's AMX, which
OpenBLAS here does not. That is a real result on this machine and not
necessarily one on another, which is what `machine.id` grouping exists for.

## Confounds that remain

These are not fixed by the harness. They must be stated wherever results are
presented.

**MKL on AMD.** This suite was developed on a Ryzen 9 7900X. MKL selects its
kernel path from a CPU vendor check and dispatches conservatively on non-Intel
parts; the `MKL_DEBUG_CPU_TYPE=5` workaround was removed in MKL 2020. An
"OpenBLAS beats MKL" result for the zgemm-bound propagators on Zen is likely
measuring MKL's AMD dispatch, not the propagators. Reproduce on an Intel part
before drawing a conclusion. `dragrace report` refuses to plot across machine
fingerprints so this cannot silently contaminate a comparison.

**The hand-written adjoint.** On the gradient board, prysm's adjoint chain is
written by the benchmark author while dLux's is generated by JAX. A clumsy chain
penalises prysm; an over-tuned one flatters it. Mitigations: the chain uses
prysm's own `*_adjoint` API rather than reimplementing it, the expected
primitive count (2 forward GEMMs + 2 adjoint GEMMs) is asserted in the ledger,
and each adapter carries a `reviewed_by` field that should be filled in by the
library's maintainer before publication. The comparison is partly "hand-written
adjoint vs automatic differentiation", which is interesting but is not the same
claim as "prysm vs dLux", and must be labelled as such.

**dLux pays overhead to be differentiable.** A forward-only board penalises it
for a design choice, not an inefficiency. The gradient board exists partly to
correct for this; report `t_grad / t_forward` alongside absolute times, since
the ratio isolates the differentiation machinery from the forward propagation
already covered elsewhere.

**POPPY's `astropy.units` and PROPER's global state** are usability and safety
choices with real runtime cost. The trace-category breakdown will surface them
(`units: N% of self time`). Report these as attributions, not verdicts.

**PROPER has almost nothing to hoist.** Its unit of work is a prescription
executed against module-level global state, so its cold and warm numbers nearly
coincide. That is a property of the API, not a measurement artifact. The adapter
also calls the prescription directly rather than through `prop_run()`, which
skips PROPER's file-based prescription lookup — per-call overhead a real user
does pay, noted here rather than quietly excluded.

**Small N is dispatch-bound.** At N ≤ 512, and on GPU at almost any N these
codes are run at, the arithmetic is a minority of runtime and Python dispatch
dominates. That is a real result, not a measurement failure — but it means
single-N conclusions will be wrong, and the N-sweep is load-bearing. That sweep
is the `mft_array_scan` case: every size measured in one process against one
configured adapter and written to one file, so the curve's slope is comparable
even where its absolute values are not. Read the slope against the ideal-FLOP
scaling `dragrace plot` draws alongside it; a code flatter than ideal at small N
is paying fixed per-call overhead, not running efficiently.

**The ledger under-counts GEMMs.** `A @ B` between plain ndarrays calls
`ndarray.__matmul__` in C and does not route through `np.matmul`, so it is
invisible to the interception layer. FFT calls are caught reliably. For
MFT-heavy codes the analytic model remains the primary FLOP source; validate
against hardware counters (`perf stat -e fp_arith_inst_retired.*`, or LIKWID's
`FLOPS_DP`) once per machine. dLux needs none of this — XLA reports exact costs
through `compiled.cost_analysis()`.

## CI

CI runs correctness gates and schema validation only. Shared runners are far too
noisy for timing, and a timing number produced there would be indistinguishable
in the results directory from a good one.
