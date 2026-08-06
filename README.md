# Diffraction-Dragrace

Standardised benchmarks for open-source physical optics propagators:
**PROPER, POPPY, HCIPy, prysm, lentil, dLux**.

Compares runtime, memory footprint and — the part that survives being run on
someone else's machine — *algorithmic* efficiency: how much arithmetic each
code actually performs relative to what the physics requires.

A benchmark run is the triple **(case × config × adapter)**:

| | what it fixes | where |
|---|---|---|
| **case** | the physics to reproduce | `cases/*.yaml` |
| **config** | the machinery to use (device, FFT, BLAS, threads, precision) | `configs/*.yaml` |
| **adapter** | the code under test | `adapters/*/adapter.py` |

---

## Install

```bash
git clone git@github.com:Jashcraf/Diffraction-Dragrace
cd Diffraction-Dragrace
conda env create -f environment.yml        # ~5-10 min
conda activate dragrace
bash scripts/setup_env.sh                  # PROPER + activation-time variables
pip install -e .                           # the harness itself
```

`environment.yml` runs all six adapters on CPU. The MKL and GPU variants live in
`envs/` — see [envs/README.md](envs/README.md) for why a single environment
cannot cover the backend axis.

Check what you got:

```bash
dragrace doctor
```

```
machine   AMD Ryzen 9 7900X 12-Core Processor  (amd, 24 logical cores)
gpu       none detected -- gpu_* configs will be skipped
blas      openblas
          openblas   threads=1    /.../libscipy_openblas.so
fft       importable: {'mkl': True, 'pyfftw': False, 'scipy_pocketfft': True, 'numpy': True}
```

---

## Run some benchmarks

### One measurement

```bash
dragrace run --case mft_n1024_q4 --adapter numpy_baseline --config cpu_numpy_1t --mode timing
```

```
status=ok  numpy_baseline mft_n1024_q4 [cpu_numpy_1t] timing
  accuracy: rel_l2=0.000e+00 gate=pass scale=1 conj=False
  timing:   median=17.189 ms  min=17.021 ms
  ideal:    1.2080 GFLOP
```

That is 70 GFLOP/s of complex GEMM on one core — a plausible single-threaded
zgemm rate, which is the first sanity check on any number this suite produces.

### Modes

Each mode is a separate pass over the same case, in its own interpreter.

```bash
dragrace run --case mft_n1024_q4 --adapter prysm --mode timing    # the only mode that goes on a plot
dragrace run --case mft_n1024_q4 --adapter prysm --mode memory    # tracemalloc peak + RSS high-water
dragrace run --case mft_n1024_q4 --adapter prysm --mode ledger    # what it actually computed
dragrace run --case mft_n1024_q4 --adapter prysm --mode trace     # VizTracer, for attribution
dragrace run --case grad_zernike_p15_n256 --adapter prysm --mode gradient
dragrace run --case mft_n1024_q4 --adapter prysm --mode all       # everything except trace
```

**Timing and tracing are deliberately separate.** VizTracer's overhead is
per-Python-function-call, so it penalises loop-heavy codes far more than
vectorised ones. Traced runs are stamped `traced: true` and excluded from every
comparison table.

### The kernel-shape ledger

The single most useful command for the question "why is A slower than B":

```bash
dragrace ledger --case fft_n1024_q4 --adapter numpy_baseline
```

```
op                           shape                     n      GFLOP
--------------------------------------------------------------------
numpy.fft.fft2               (4096, 4096)              1     2.0133
--------------------------------------------------------------------
TOTAL                                                  1     2.0133

ideal (model):  2.0133 GFLOP   [1 fft2(4096x4096) with padding factor q=4]
algorithmic overhead A = ledger/ideal = 1.000
```

Run it for two adapters on the same case and diff the tables. A stray extra
transform, a rebuilt kernel matrix, an array one power of two larger than
necessary — these show up as a structural difference rather than an unexplained
1.8× in a bar chart, which is what makes them filable as issues.

### A sweep

```bash
dragrace sweep --cases mft_n1024_q4 fft_n1024_q4 \
               --adapters numpy_baseline prysm hcipy poppy lentil proper \
               --configs cpu_numpy_1t cpu_mkl_1t \
               --modes timing ledger
dragrace report
```

The runner launches each adapter in the interpreter named by the config's
`conda_env`. If that environment does not exist the run is **skipped with a
reason**, never silently executed in whatever interpreter happens to be active:

```
  -- numpy_baseline   cpu_numpy_1t     mft_n1024_q4    timing    skipped
     conda environment 'dragrace' does not have the harness installed
     (ModuleNotFoundError: No module named 'yaml'). Create it from
     environment.yml / envs/, then `pip install -e .` ...
```

### Roofline peaks

```bash
dragrace machine          # measured zgemm peak + STREAM triad, not spec-sheet numbers
```

---

## What gets recorded

Every run writes `results/raw/<machine>/<run_id>/<adapter>/<config>/<case>/<mode>/result.json`
containing the timing distribution (not just a mean), the accuracy comparison,
the resolved backend, the machine fingerprint, and the setup-cost breakdown.

Two guard rails are enforced rather than left to discipline:

- **Backend verification.** Several of these libraries probe for `mkl_fft` or
  `pyfftw` at import and use them silently. Every adapter reports what it
  actually resolved, and a run whose resolved backend contradicts its config is
  **refused**, because a mislabelled result is worse than no result. During
  development this caught a run labelled `threads=1` that was using 24 — an 8×
  error that would otherwise have looked like a plausible measurement.
- **Cross-machine separation.** `dragrace report` never plots two machine
  fingerprints on one axis. On this suite that is not pedantry: MKL dispatches
  conservatively on AMD parts, so the same MKL-vs-OpenBLAS board can legitimately
  invert between a Ryzen and a Xeon.

---

## The two things this suite measures that a stopwatch cannot

### Algorithmic overhead, `A = flops_actual / flops_ideal`

`flops_ideal` comes from an analytic model of the case's physics — no library
involved ([docs/flop_model.md](docs/flop_model.md), hand-derivations checked in
`tests/test_ideal_flops.py`). `A` is hardware-independent and is the closest
thing here to "is this code doing more work than the physics requires".

Paired with execution efficiency `E = flops_actual / (t × roofline)`, it splits
the question a wall-clock ranking conflates: a code can lose on time while
winning on `A` (right algorithm, poor execution — fixable) or the reverse
(brute force on a fast backend — architectural).

Worked example of why `algorithm_class` is part of the case, not the adapter's
choice: reaching 4 samples per λ/D over a 32 λ/D field costs an MFT
**1.208 GFLOP** and an FFT **2.013 GFLOP** — but at a wide enough field of view
the ranking inverts, because the FFT's cost does not grow with output extent
while the MFT's does. Ranking those on one axis would be meaningless.

### The gradient board: prysm vs dLux

`∂L/∂θ` for a real parameter vector θ and a real loss, so the returned gradient
is unambiguously real and the two codes are **directly comparable with no
Wirtinger reconciliation** — see [docs/gradient_board.md](docs/gradient_board.md).

- **prysm** uses its own manual reverse mode: `sum_of_2d_modes_adjoint`,
  `focus_dft_adjoint` — no tape, no tracing, explicit control of what is kept.
- **dLux** uses `jax.value_and_grad`, AOT-compiled so compile time is measured
  rather than hidden, with `cost_analysis()` giving XLA's exact FLOP count.

Headline question: **does automatic differentiation cost more arithmetic than a
hand-written adjoint for this physics?** Gated against central differences at
`max_rel_err < 1e-6` *and* cosine similarity `> 1 − 1e-9` — the cosine check
catches a uniform factor of 2, the classic Wirtinger slip, that a loose
per-component tolerance would let through.

---

## Status

| adapter | status | notes |
|---|---|---|
| `numpy_baseline` | **verified** | the floor, and the harness's own test subject |
| `prysm` | unverified | API read from the v0.22 source; adjoint chain is prysm-native |
| `poppy` | unverified | written from documented API |
| `hcipy` | unverified | written from documented API |
| `lentil` | unverified | written from documented API |
| `proper` | unverified | written against the PROPER 3.3.4 source |
| `dlux` | unverified | JAX machinery is complete; dLux model construction needs checking |

"Unverified" means the adapter has not yet been run against a real install.
`dragrace doctor` surfaces this so an untested adapter is never mistaken for a
measured result. **No adapter should be published without its `reviewed_by`
field filled in** — adapter authors are not neutral parties, and the
hand-written-adjoint confound on the gradient board is real and documented.

---

## Documentation

- [docs/methodology.md](docs/methodology.md) — what is timed, what is not, and the known confounds
- [docs/conventions.md](docs/conventions.md) — grids, normalisation, phase sign
- [docs/flop_model.md](docs/flop_model.md) — the cost model, with derivations
- [docs/gradient_board.md](docs/gradient_board.md) — prysm vs dLux in detail
- [docs/adding_an_adapter.md](docs/adding_an_adapter.md) — the contract
- [envs/README.md](envs/README.md) — the environment matrix and its caveats

## Tests

```bash
pytest tests -q      # 26 passed
```

Correctness only. **Never** use CI for timing — shared runners are far too
noisy.
