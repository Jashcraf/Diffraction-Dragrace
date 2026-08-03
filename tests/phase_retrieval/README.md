# Physical-optics propagator drag race — phase retrieval

A head-to-head comparison of four physical-optics propagators —
**POPPY**, **HCIPy**, **prysm**, and **dLux** — solving the *same* single-image
phase-retrieval problem by nonlinear optimization (scipy `L-BFGS-B`), and timing
each one.

## The problem

* **Aperture:** a Subaru-like, deliberately **asymmetric** pupil — a filled
  circle with a central obscuration and a four-vane spider whose intersection is
  *offset* from the center (as the real Subaru spider is). Because the amplitude
  is not point-symmetric, the Fourier-modulus twin-image ambiguity is broken and
  a **single in-focus PSF is sufficient** — no focus diversity required.
* **Unknown:** the pupil phase, parameterized by a modal **Zernike (Noll 4–14)**
  basis (11 modes). A modal basis keeps the finite-difference gradient tractable
  for the non-differentiable packages.
* **Data:** each package generates its own in-focus PSF from a known truth phase
  (a self-consistent "inverse crime"), then retrieves the coefficients starting
  from zero. This isolates *propagator + gradient* speed from cross-package
  sampling differences. Every package converges to **0.0000 rad RMS** phase
  error, so the comparison is purely about speed.

Everything shared — the pupil, the Zernike basis, the truth coefficients, the
focal sampling, the optimizer and its tolerances — lives in [`common.py`](common.py)
and [`bench_util.py`](bench_util.py) (pure numpy/scipy), so all four packages
solve an identical problem.

## Numerical precision

Every propagator runs in **double precision** (`float64` pupil/PSF,
`complex128` field), and each script **asserts** it at runtime and records the
PSF dtype in its JSON:

* **prysm / HCIPy** — numpy, `complex128` by default.
* **POPPY** — `poppy.conf.double_precision = True` (checked and recorded).
* **dLux** — jax defaults to `float32`; the script sets
  `jax.config.update("jax_enable_x64", True)` **before** importing jax and
  asserts `jax_enable_x64` is on. Without x64, jax would silently drop to
  `float32`, which both stalls the retrieval and makes it incomparable to the
  numpy packages. The `Dtype` column in `results/summary.md` shows `float64` for
  all runs.

## Comparisons

The retrieval sweeps run over **N = 2⁶…2¹⁰ (64…1024)** (the forward-only
Comparison G goes to **2048**), and **dLux is always shown as two lines — jit and
no-jit (eager)** — so you can see how JIT and non-JIT dLux each stack up against
POPPY/HCIPy/prysm (and PROPER, in the forward comparison).

* **Comparison A — no gradient back-propagation.** POPPY, HCIPy, prysm, dLux all
  use `L-BFGS-B` with a **finite-difference** gradient (forward model only).
  dLux is run both jit and `--no-jit`.
* **Comparison B — gradient back-propagation.** Only the differentiable
  propagators: **prysm** via its DFT adjoint (`focus_dft` + `focus_dft_adjoint`,
  the `_backprop` path, mirroring `dygdug/phase_retrieval.py`) and **dLux** via
  jax `value_and_grad` (shown jit and `--no-jit`). Both feed an analytic/AD
  gradient to `L-BFGS-B`.
* **Comparison C — dLux JIT vs eager.** dLux back-prop with `jax.jit`
  (`--grad`) vs eager execution (`--grad --no-jit`). JIT-compilation time is
  excluded via a warm-up call, so this isolates compiled-vs-interpreted
  dispatch cost.
* **Comparison D — memory footprint.** Peak resident memory (RSS above a
  post-import baseline) of the retrieval, for seven cases over N = 2⁶…2¹⁰:
  POPPY, HCIPy, prysm (finite diff), prysm (back-prop), dLux (finite diff, no
  jit), dLux (back-prop, no jit), dLux (back-prop, jit). Run with `run_mem.py`.
  Peak RSS (`resource.getrusage().ru_maxrss`) is used rather than `tracemalloc`
  because it captures native numpy/jax/poppy allocations, not just the Python
  heap.
* **Comparison E — batched throughput.** The trade-study metric: **cases per
  second** when evaluating a batch of B independent aberrations, vs batch size.
  Strategies — POPPY/HCIPy/prysm via a Python `loop`, and dLux via `loop` (jit
  and no-jit) and `jax.vmap` (one fused kernel). Run with `run_throughput.py`.
  This is the metric that most directly means "how many trades/second," and the
  `--device` flag makes it GPU-ready — `vmap` on a GPU is where throughput scales
  far beyond what any per-case loop can reach.
* **Comparison F — code layers per image simulation.** A `cProfile` count of
  how many Python-visible function calls each package makes to turn
  coefficients into one PSF — a proxy for how many layers of framework wrap the
  diffraction math. dLux is measured both eager and jit. Run with
  `run_callcount.py`.
* **Comparison G — forward-model speed (no retrieval).** Like Comparison A but
  with *no* optimization: the time for a single image simulation (pupil → PSF)
  vs N (swept to **2048** here), with dLux shown jit and no-jit. This isolates
  raw propagation cost, without the ~600× finite-difference multiplier that
  Comparison A's wall time folds in. Also includes **PROPER** (John Krist) —
  which has no autodiff and so appears only in this forward-only comparison.
  Run with `run_forward.py`.

## Layout

| File | Role |
|---|---|
| `common.py` | Shared problem: Subaru pupil, Zernike basis, truth phase, sampling (numpy only) |
| `bench_util.py` | Shared CLI, timing, L-BFGS-B driver, result JSON (numpy+scipy only) |
| `bench_poppy.py` | POPPY forward model (no-grad) |
| `bench_hcipy.py` | HCIPy Fraunhofer forward model (no-grad) |
| `bench_prysm.py` | prysm matrix-DFT, no-grad **and** `--grad` adjoint back-prop |
| `bench_dlux.py` | dLux MFT, no-grad **and** `--grad` jax `value_and_grad` |
| `bench_proper.py` | PROPER (John Krist) FFT forward model — forward-only (Comparison G) |
| `run_all.py` | Orchestrator — runs each script in its own conda env, sweeps N (Comparisons A/B/C) |
| `run_mem.py` | Memory-footprint orchestrator (Comparison D) → `results/mem/*.json` |
| `bench_throughput.py` | Batched forward-model throughput, `loop` vs `vmap` (one file, lazy per-env import) |
| `run_throughput.py` | Throughput orchestrator (Comparison E) → `results/throughput/*.json` |
| `bench_callcount.py` | `cProfile` function-call count per propagation (one file, lazy per-env import) |
| `run_callcount.py` | Call-count orchestrator (Comparison F) → `results/callcount/*.json` |
| `run_forward.py` | Forward-model-only speed vs N (Comparison G, reuses `bench_throughput.py --batch 1`) → `results/forward/*.json` |
| `plot_results.py` | Aggregates `results/*.json` (+ `mem/`, `throughput/`, `callcount/`) into plots + `summary.md` |

## Environments

The packages have incompatible dependency stacks, so each runs in its own conda
env. `run_all.py` shells out to the right interpreter per package:

| Package | Env | Version |
|---|---|---|
| POPPY | `grater_jax` | 1.1.2 |
| HCIPy | `joss` | 0.6.0 |
| prysm | `prysm_dev` | (editable `~/prysm`, DFT-adjoint API) |
| dLux | `dlux` | 0.15.0 |
| PROPER | `corgi` | 3.3.3 (forward-only, Comparison G) |

Edit `ENV_PYTHON` in `run_all.py` if your paths differ.

## Running

```bash
cd tests/phase_retrieval
python run_all.py                      # full sweep N = 2^6..2^10, CPU (Comp. A/B/C)
python run_mem.py                      # memory footprint N = 2^6..2^10 (Comp. D)
python run_throughput.py               # batched throughput (Comp. E)
python run_callcount.py                # function calls per propagation (Comp. F)
python run_forward.py                  # forward-model-only speed vs N (Comp. G)
python plot_results.py                 # -> results/*.png, results/summary.md
```

The default sweep is `N = 64, 128, 256, 512, 1024`. POPPY's finite-difference
retrieval dominates the wall time at large N (several minutes for the whole
sweep); pass e.g. `--ns 64 128 256` for a quick pass.

Individual runs:

```bash
/opt/anaconda3/envs/prysm_dev/bin/python bench_prysm.py --n 128 --grad --check-grad
/opt/anaconda3/envs/dlux/bin/python     bench_dlux.py  --n 128 --grad            # JIT
/opt/anaconda3/envs/dlux/bin/python     bench_dlux.py  --n 128 --grad --no-jit   # eager
/opt/anaconda3/envs/grater_jax/bin/python bench_poppy.py --n 128
/opt/anaconda3/envs/joss/bin/python     bench_hcipy.py --n 128
```

## CPU vs GPU

The harness is **GPU-ready**: every script takes `--device {cpu,gpu}`. jax
packages (dLux, and prysm via a cupy backend) honor the device; the numpy
packages (HCIPy, POPPY without cupy) are CPU-only and say so.

This machine is an **Apple M1** — there is no CUDA/CuPy and jax sees only a
`CpuDevice`, so all numbers here are **CPU**. Re-run `run_all.py --device gpu`
on a CUDA box to populate GPU numbers (dLux and prysm-cupy will move to GPU; the
others remain CPU with a warning).

## Results (Apple M1, CPU)

Median wall time to retrieve the phase (full `L-BFGS-B` solve). See
[`results/summary.md`](results/summary.md) for the full table and
`results/*.png` for the plots.

Median wall time in **milliseconds**, sweep N = 2⁶…2¹⁰. dLux is shown twice —
**jit** and **no-jit (eager)** — in both the finite-difference (Comparison A)
and back-prop (Comparison B) blocks:

| Case | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|
| POPPY (fd) | 12259 | 24063 | 32984 | 43292 | 69089 |
| HCIPy (fd) | 145 | 334 | 1114 | 4335 | 15869 |
| prysm (fd) | 89 | 280 | 1127 | 3917 | 16826 |
| **dLux (fd, jit)** | 176 | 419 | 882 | 2497 | 8713 |
| **dLux (fd, no-jit)** | 2490 | 2758 | 3252 | 5657 | 15601 |
| prysm (bp) | 12 | 34 | 108 | 474 | 1830 |
| **dLux (bp, jit)** | 22 | 56 | 120 | 295 | 1233 |
| **dLux (bp, no-jit)** | 294 | 425 | 463 | 796 | 2598 |

(fd = finite-difference gradient; bp = analytic/AD back-prop. POPPY is
overhead-bound and its absolute numbers vary by a factor of a few with machine
load — it is consistently seconds vs milliseconds regardless.)

### Takeaways

* **JIT is decisive for dLux, most of all at low resolution.** In both fd and bp
  modes, eager dLux carries a large fixed dispatch overhead: at N=64 no-jit is
  ~14× slower than jit (fd: 2490 vs 176 ms; bp: 294 vs 22 ms). The gap shrinks
  as N grows and the propagation FFT dominates (~1.8× in fd, ~2.1× in bp at
  N=1024), but **jitted dLux beats eager dLux at every size**.
* **Where non-JIT dLux lands among the others:** in finite-difference mode,
  eager dLux is *slower than prysm/HCIPy* at small–mid N (its dispatch floor
  dominates when the arrays are small) and only catches up by N≈1024. In
  back-prop mode, eager dLux is still far faster than any finite-difference
  package but 2–13× slower than jitted dLux or prysm-adjoint. **JIT is what
  makes dLux competitive.**
* **Back-prop wins big.** ~50 forward evaluations to converge vs ~600 for finite
  differences. jitted dLux and prysm-adjoint are neck-and-neck (prysm leads at
  small N, dLux edges ahead by N≥512 via kernel fusion).
* **POPPY is the finite-difference outlier** — per-`calc_psf` overhead keeps it
  1–2 orders of magnitude slower across the whole sweep.
* **All 40 runs retrieve the phase to 0.0000 rad RMS** across 2⁶–2¹⁰, jit or
  eager — confirming the asymmetric pupil makes single-image retrieval well
  posed for every configuration.

### dLux — JIT vs eager (Comparison C)

`jax.jit` on the dLux back-prop loop (compilation excluded via warm-up):

| N | eager [ms] | JIT [ms] | speedup |
|---|---|---|---|
| 64   | 294  | 22   | **13.7×** |
| 128  | 425  | 56   | **7.6×** |
| 256  | 463  | 120  | **3.9×** |
| 512  | 796  | 295  | **2.7×** |
| 1024 | 2598 | 1233 | **2.1×** |

The eager path pays a roughly fixed per-call Python/dispatch overhead (note how
its time barely moves from N=64 to N=256), so the JIT win is largest at small N
and steadily shrinks as the propagation FFT itself comes to dominate — from
~14× at N=64 down to ~2× at N=1024. Both paths converge identically (same
iterations, same `float64` result) — JIT only changes speed. See
`results/dlux_jit_vs_eager.png`. Note that the *first* jitted call also pays a
one-off compilation cost (excluded here by warming up before timing).

### Memory footprint (Comparison D)

Peak RSS **above the post-import baseline** [MiB], N = 2⁶…2¹⁰
(`results/memory_footprint.png`):

| Case | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|
| POPPY                     | 11  | 18  | 49  | 163 | 396  |
| HCIPy                     | 3   | 10  | 30  | 119 | 296  |
| prysm (finite diff)       | 25  | 29  | 61  | 118 | 281  |
| prysm (back-prop)         | 24  | 30  | 53  | 130 | 292  |
| dLux (finite diff, no jit)| 93  | 132 | 238 | 750 | 976  |
| dLux (back-prop, no jit)  | 135 | 168 | 238 | 365 | 789  |
| dLux (back-prop, jit)     | 110 | 136 | 175 | 360 | 1156 |

* **The numpy propagators are lean.** HCIPy is lightest; POPPY, prysm-fd and
  prysm-bp all sit within a factor of ~2 of each other and scale ~N² (a few
  hundred MiB at N=1024). prysm's back-prop costs almost nothing extra over its
  forward model — the adjoint reuses the same buffers.
* **jax/dLux carries a large, roughly fixed floor** (~100 MiB even at N=64) from
  the XLA runtime, and is 3–10× heavier than the numpy packages throughout.
* **dLux eager finite-diff is the memory hog at mid N** (750 MiB at N=512):
  eager execution materializes many un-fused temporaries, and the finite-diff
  loop drives ~576 forward passes. `jax.jit` fuses these and roughly halves the
  footprint at N≤512 — though by N=1024 the JIT'd back-prop's peak rises again
  (compiled-kernel working set + retained buffers), so the JIT-vs-eager memory
  ordering is *not* uniform across N.

Peak RSS (`resource.getrusage`) is the true high-water resident set, so it
captures native allocations that `tracemalloc` would miss; the shared Zernike
basis (11·N²·8 B, e.g. 88 MiB at N=1024) is a common component of every line.

### Batched throughput (Comparison E)

Forward-model **throughput [cases/second]** vs batch size, N = 256, CPU
(`results/throughput.png`) — this is the closest metric to "trades per second":

| Strategy | B=1 | B=4 | B=16 | B=64 | B=256 |
|---|---|---|---|---|---|
| POPPY (loop) | 23 | 15 | 12 | 20 | 42 |
| HCIPy (loop) | 498 | 648 | 686 | 652 | 594 |
| prysm (loop) | 731 | 704 | 677 | 737 | 715 |
| dLux (loop, jit) | 592 | 685 | 448 | 722 | 720 |
| dLux (loop, no jit) | 292 | 288 | 213 | 291 | 232 |
| **dLux (vmap, jit)** | 575 | 945 | 1241 | 1414 | **1314** |

* **Loops are flat** — total time scales linearly with batch, so cases/second is
  ~constant. HCIPy, prysm, and jitted dLux-loop cluster at ~600–700 cases/s;
  **eager (no-jit) dLux-loop sits ~2–3× lower** (~250 cases/s) — the same
  per-call dispatch overhead seen everywhere else; **POPPY is ~20–40× lower**
  (~20 cases/s), its per-`calc_psf` overhead dominating exactly as a trade study
  would experience it.
* **`vmap` is the only strategy that scales with batch** — fusing the batch into
  one compiled kernel lifts dLux from ~700 to ~1400 cases/s (≈2× the best loop)
  on this CPU. The gain is modest here because a laptop CPU has little
  data-parallelism to exploit; **the same code on a GPU is where `vmap`
  throughput pulls away by 1–2 orders of magnitude** — rerun with
  `run_throughput.py --device gpu` on a CUDA box.
* **Takeaway for trade studies:** among the classical numpy propagators, prysm
  and HCIPy give the best per-case loop throughput and POPPY is disqualified by
  overhead; but the structural win is a `vmap`-able (jax/torch) forward model,
  because only vectorised batching converts more hardware into more
  trades/second.

This benchmarks the *forward model* (the propagation that dominates any solve);
gradients (`value_and_grad`) `vmap` the same way, so a batched gradient-based
trade study inherits the same scaling.

### Code layers per image simulation (Comparison F)

`cProfile` count of Python-visible **function calls per forward propagation**
(N = 256; the count is essentially N-independent — only the array math scales):

| Propagator | calls / propagation | distinct functions |
|---|---|---|
| prysm         | **33**   | 23  |
| dLux (jit)    | 55       | 38  |
| HCIPy         | 246      | 102 |
| POPPY         | 9 539    | 369 |
| dLux (eager)  | 9 764    | 502 |

* **prysm is the leanest path to a PSF** — 33 calls, essentially "build the
  field, one matrix-DFT, take the modulus." HCIPy is a moderate 246.
* **POPPY and dLux-eager both make ~10 000 calls, for opposite reasons.** POPPY's
  come from framework machinery — astropy-`Quantity` unit checks, `Wavefront`
  objects, and per-PSF FITS-header construction (see the profiling in the notes
  below). dLux-eager's come from jax *tracing every primitive op in Python* on
  each call.
* **`jax.jit` collapses dLux from 9 764 calls to 55** — compilation folds all
  those Python layers into a single fused kernel, so the steady-state call count
  is tiny. This is the same mechanism behind dLux's throughput and runtime wins,
  seen from the code-path side: the eager layers exist only at trace time, not
  per evaluation.

The call count is a clean proxy for "how many layers of code wrap the math":
lean numerical kernels (prysm, jitted dLux) sit at tens of calls; a
metadata-rich instrument simulator (POPPY) or an un-compiled tracing framework
(dLux eager) sit at thousands. This is *why* POPPY's per-call overhead dominates
inside an optimizer loop — see `results/callcount.png`.

### Forward-model speed (Comparison G)

Time for **one** image simulation (pupil → PSF), no retrieval [ms], swept to
N = 2048 (`results/comparison_G_forward.png`). Includes **PROPER** (John Krist),
which — being FFT/AD-free — only participates in this forward-only comparison:

| Propagator | 64 | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|
| POPPY        | 27   | 31   | 37   | 66   | 120 | 300 |
| HCIPy        | 0.28 | 0.52 | 1.85 | 7.31 | 24  | 104 |
| prysm        | 0.14 | 0.43 | 1.55 | 6.08 | 26  | 105 |
| PROPER       | 18   | 19   | 24   | 43   | 118 | 412 |
| dLux (jit)   | 0.32 | 0.61 | 1.61 | 4.28 | 15  | 52  |
| dLux (no jit)| 2.21 | 2.68 | 3.53 | 8.46 | 24  | 90  |

* Stripped of the ~600× finite-difference multiplier, the propagation cost is
  clear: **prysm and jitted dLux are the leanest**, HCIPy close behind, and by
  N≥512 **jitted dLux is fastest** (kernel fusion). At N=2048 it is ~2× the
  others in that group.
* **PROPER and POPPY are the two overhead-bound propagators** — both sit on a
  ~20–35 ms floor at small N (framework setup dominating the tiny FFTs) rather
  than the sub-millisecond kernels. **At large N PROPER becomes the slowest of
  all** (412 ms at N=2048): it propagates the *full* N×N grid by FFT, whereas
  POPPY/prysm/dLux use a matrix-DFT to a small (64²) detector region, so PROPER
  pays the full N²·log N transform where the MFT packages pay only N²·M_det.
  This is the classic FFT-to-full-grid vs MFT-to-ROI trade, made concrete.
* **dLux eager (no jit) has a ~2.5 ms fixed floor** — a single un-compiled
  forward pass is dominated by Python-level op dispatch until the arrays grow
  large enough (N≈1024) for the FFT to catch up. This is the per-call cost that,
  multiplied across an optimizer loop, produced the large fd/bp gaps above.
* **POPPY's flat small-N floor** is its astropy-units + FITS-header machinery,
  ~10 000 Python calls per PSF (Comparison F).

### Caveats

* Runtime in Comparison A conflates propagator speed with per-call framework
  overhead — that is the honest cost of driving each package inside an
  optimizer loop, but it is not a pure FFT/MFT micro-benchmark.
* The "inverse crime" (each package fits data it generated) is intentional: it
  removes sampling-convention mismatches so the timing is comparable. It is not
  a test of retrieval robustness to noise or model error.
* Comparison E times the forward model only, and the dLux `loop` blocks on each
  case (materialising every PSF) so it is a fair "naive loop"; `vmap` issues one
  kernel. The numpy loops (POPPY/HCIPy/prysm) are inherently sequential — those
  packages have no first-class batch axis, which is itself the finding. GPU
  numbers require rerunning on CUDA hardware.
