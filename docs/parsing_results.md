# Parsing results

Every measurement this repo produces is a `result.json` written by
`src/dragrace/worker.py`. This document is the field-by-field reference for that
file, plus the two aggregate files (`results/machine.json`, `results/index.json`)
that `dragrace machine` and `dragrace report` write.

The guiding principle behind the schema: **nothing that could change a
conclusion is discarded**. The raw per-repeat timings are kept, not just their
summary; the fitted normalisation and conjugation flag are kept, not just the
residual they were fitted to absorb; the backend that was *resolved* is kept
alongside the one that was *requested*. Parsers are expected to be tolerant of
absent keys and intolerant of merging across machines.

## Where the files live

```
results/
├── machine.json        # measured roofline peaks for this host
├── index.json          # flat aggregate of every result.json, written by `dragrace report`
└── raw/<machine_id>/<run_id>/<adapter>/<config_id>/<case_id>/<mode>/result.json
```

`<machine_id>` is `machine.id` with `:` replaced by `_`
(`sha256:d0ae…` → `sha256_d0ae…`), because a colon is not portable in a path.
`<run_id>` is a 12-hex-digit sweep identifier from `runner.new_run_id()`. The
remaining four segments duplicate `adapter.name`, `config_id`, `case_id` and
`mode` inside the file, so a result is self-describing if it is moved.

## Top level

| key | type | notes |
|---|---|---|
| `schema_version` | int | Currently `1`. Absent on runner-written skips (see [Skipped results](#skipped-results-have-a-different-shape)). |
| `measurement_contract` | str | **What the timed region means.** `idiomatic-v1` times the call the library documents; `primitive-v1` timed its transform entry point. Absent on results written before contracts existed — treat a missing marker as `primitive-v1`. Two contracts are never comparable; see [Measurement contracts](#measurement-contracts). |
| `case_id` | str | The case YAML's `id`, e.g. `mft_n1024_q4`. Physics only — never names a library or backend. |
| `config_id` | str | The config YAML's `id`, e.g. `cpu_numpy_1t`. Backend, threads, device, conda env. |
| `mode` | str | `timing` \| `memory` \| `ledger` \| `trace` \| `gradient` \| `all`. Determines which blocks are present. |
| `status` | str | See below. **Check this before reading anything else.** |
| `reason` | str | Present only when `status != "ok"`. Human-readable, and written to be actionable. |
| `traceback` | str | Present only when `status == "failed"`. Full Python traceback. |
| `adapter` | obj | Identity and provenance of the code under test. |
| `machine` | obj | Host fingerprint. |
| `provenance` | obj | Git SHA, timestamp, argv. |
| `backend` | obj | Requested vs resolved vs detected. |
| `setup` | obj | Untimed-but-measured build and first call. |
| `flops` | obj | Analytic model, efficiency decomposition, and (ledger mode) observed kernel shapes. |
| `accuracy` | obj | Forward-board gate. |
| `gradient_accuracy` | obj | Gradient-board gate. `gradient` mode only. |
| `timing` | obj | The distribution, not a mean. |
| `memory` | obj | `memory`/`all`/`gradient` modes. |
| `trace` | obj | `trace` mode only. |
| `scan` | obj | Scan cases only, and **mutually exclusive with the six measurement blocks above** — they move inside each point. See [The `scan` block](#the-scan-block). |

### `status`

| value | meaning | written by |
|---|---|---|
| `ok` | Ran to completion and passed its gate. The only status whose numbers belong on a plot. | worker |
| `partial` | Scan cases only: some points measured, others did not. Never `ok` — a curve with holes must not read as a complete measurement. Each point carries its own status. | worker |
| `unsupported` | The adapter declined the run: library not installed, case out of scope, `configure()` failed, or no gradient support. Returned rather than raised so the report can render an honest matrix of *why* each hole exists. | worker |
| `backend_mismatch` | The resolved backend contradicts the config. The run is **refused** — a mislabelled result is worse than no result. | worker |
| `accuracy_fail` | Gate failed. No timing is recorded, deliberately: a code that is fast because it is less accurate must not win. | worker |
| `failed` | An unhandled exception. `traceback` is populated. | worker |
| `skipped` | The config's conda env is missing or lacks the harness, so the run never started. | runner |
| `pending` | Skeleton default. Should never be observed on disk; every return path overwrites it. | — |

Treat anything other than `ok` as "no measurement", but do not discard it —
`reason` is the content of the "NOT MEASURED" section of `dragrace report`.

## `adapter`

| key | type | notes |
|---|---|---|
| `name` | str | Adapter directory under `adapters/`. |
| `status` | str | `verified` (exercised against the real library on this machine) or `unverified` (API calls written from documentation). Do not present `unverified` numbers as measurements without saying so. |
| `reviewed_by` | str | Who has reviewed this adapter for fairness. Empty means nobody. Adapter authors are not neutral parties — see [methodology.md](methodology.md) on the hand-written-adjoint confound. |
| `versions` | obj | Adapter-supplied `{package: version}`. Keys vary by adapter; values may be `"unknown"` if the library exposes no `__version__`. |
| `grid_centering` | str \| obj | `pixel` (origin on sample `N//2`) or `interpixel` (origin between the middle samples) — or a `{"pupil": …, "focus": …}` mapping when a library's two planes disagree, as HCIPy's do. The convention this code's output obeys, declared by the adapter; the reference and the injected pupil are built to match. POPPY and dLux are `interpixel` and cannot be talked out of it. **Parse both shapes.** |

`status`, `reviewed_by` and `versions` are absent on runner-written skips, which
carry only `{"name": ...}`.

## `machine`

| key | type | notes |
|---|---|---|
| `cpu` | str | `model name` from `/proc/cpuinfo`, else `platform.processor()`. On macOS this is typically just `arm`. |
| `cpu_vendor` | str | `intel` \| `amd` \| `apple` \| `unknown`, inferred from `cpu`. Load-bearing: MKL dispatches conservatively on AMD parts. |
| `logical_cores` | int | `os.cpu_count()`. |
| `ram_bytes` | int \| null | psutil, falling back to `sysconf`. |
| `platform` | str | `platform.platform()`. |
| `python` | str | Interpreter version — the *worker's*, i.e. the config's conda env. |
| `gpus` | list | One `{name, driver, memory}` per `nvidia-smi` row. Empty list means no NVIDIA GPU **or** no `nvidia-smi`; the two are not distinguished. |
| `id` | str | `sha256:` + 16 hex chars of `cpu\|logical_cores\|platform`. Deliberately excludes the hostname so published result sets do not leak machine names. |

**`id` is the grouping key.** Results from different fingerprints must never
share an axis: the same MKL-vs-OpenBLAS board can legitimately invert between a
Ryzen and a Xeon, and a merged plot presents that as a property of the
propagators. `report.render_text` enforces this and warns when rows span more
than one.

## `provenance`

| key | type | notes |
|---|---|---|
| `git_sha` | str | Short SHA, suffixed `-dirty` if the tree had uncommitted changes, or `unknown` outside a repo. **A `-dirty` result is not reproducible from the SHA alone.** |
| `utc` | str | ISO 8601, second resolution. |
| `run_id` | str | Always `""` in practice — `fingerprint.provenance()` is called without an argument. Recover the sweep id from the path segment instead. |
| `argv` | list[str] | The worker's exact `sys.argv`, including absolute case/config/output paths. This is what makes a single run re-executable by copy-paste. |

## `backend`

The premise of this block: never trust that a config was honoured. HCIPy probes
for `mkl_fft` and `pyfftw` at import; POPPY's `accel_math` toggles fall back
silently; prysm swaps its whole array module. Three independent views are
recorded so a mislabelled run is detectable after the fact.

### `backend.requested` — from the config YAML

`fft`, `blas`, `threads`, `device`, `precision_override` (null unless the config
forces the case's dtype).

### `backend.resolved` — what the adapter says it actually used

Adapter-defined, so **the key set varies**. Common keys are `fft_backend`,
`array_module`, `device`, `blas`; prysm additionally reports `fft_module`
(`scipy.fft`), which is exactly the kind of difference this block exists to
expose. Parse defensively.

### `backend.detected` — independent detection, not adapter self-report

| key | notes |
|---|---|
| `blas` | `mkl` \| `openblas` \| `accelerate` \| `unknown`, from threadpoolctl's view of loaded libraries. |
| `thread_counts` | List of `{api, threads, path}`, one per loaded threading runtime. A **list**, not a dict: a process routinely has two OpenBLAS or OpenMP runtimes loaded with different thread counts, and collapsing them would mask the mismatch this exists to find. |
| `numpy_fft_module` | Where `numpy.fft.fft2` actually lives, e.g. `numpy.fft.fft2`. Anything else means a monkeypatched front end — including the harness's own ledger instrumentation, which must never be active during a timing run. |
| `fft_backends_importable` | `{mkl, pyfftw, scipy_pocketfft, numpy}` → bool. Importable, not necessarily used. |

### `backend.threadpool_info`

`threadpoolctl.threadpool_info()` verbatim: `user_api`, `internal_api`,
`num_threads`, `prefix`, `filepath`, `version`, and for OpenBLAS
`threading_layer` and `architecture`. Library paths on disk, not package
metadata — the authoritative answer to "which BLAS, with how many threads". On
failure the list holds a single `{"error": ...}` entry.

### `backend.axes_not_selectable`

Config axes the adapter declared it cannot honour, e.g. `["fft", "blas", "threads"]`
for dLux on a NumPy config. Present on every result; `[]` for the normal case.

**Read this before reading anything else in the block.** A row with a non-empty
list is a valid measurement of the code but is *not* a data point along those
axes — dLux has no FFT library to select and no BLAS to point at, so the
config's `fft=numpy` simply did not apply to it. The matching human-readable
strings appear in `warnings`. The declaration depends on the config as well as
the adapter: dLux on `cpu_xla_1t` drops `fft` from the list, because there the
config and the adapter agree.

`report.aggregate` exposes this as an `axes_not_selectable` column and
`report.latest_axes` resolves the current declaration per (adapter, config),
newest run winning — an older run's broader claim is stale metadata, not a
second measurement.

### `backend.warnings`

Non-fatal mismatches, as strings — typically an idle runtime whose thread count
differs from the config, which does not affect the measurement. A mismatch on
the *active* BLAS is fatal instead and produces `status: backend_mismatch`.
Under `--no-strict-backend` those fatal problems are appended here rather than
raised; such a result is explicitly not publishable.

## `setup`

| key | notes |
|---|---|
| `build_s` | `build()` — grids, DFT kernel matrices, FFT plans, device staging. Untimed with respect to the board, but not unmeasured. |
| `first_call_s` | First `propagate()` (or `gradient()`), including `sync()`. Where JIT, FFTW planning and NVRTC kernel compilation land. Never folded into steady state. |

`report.amortisation` combines these with the steady-state median as
`T(k) = build + first_call + k·steady`, which is the curve that answers "which
code should I reach for at my workload". `setup_cost.breakdown()` adds `import_s`
measured in a clean subprocess, since import cost can only be paid once per
interpreter.

## `flops`

### `flops.ideal` — the physics floor

From `flops/model.py`, derived from the case alone with no library involved.
See [flop_model.md](flop_model.md) for derivations.

| key | notes |
|---|---|
| `flops` | Real FLOPs, FMA-class. |
| `tops` | Transcendental evaluations (`cexp`, `sin`, `cos`), kept as a **separate currency** — their cost relative to a FLOP ranges over ~10–40× depending on SVML/libmvec vectorisation, so folding them in would encode a machine assumption into a hardware-independent metric. `0.0` when the case says the basis is hoisted into `build()` (`basis_caching: precomputed`). |
| `bytes` | Memory traffic, which is what makes the roofline classification possible. |
| `detail` | The model that was applied, e.g. `2 zgemm: (128x1024x1024) + (128x1024x128)`. |
| `arithmetic_intensity` | `flops / bytes`. **`Infinity` when `bytes == 0`** — see [Numeric edge cases](#numeric-edge-cases). |

### `flops.efficiency`

| key | notes |
|---|---|
| `flops_ideal` | Copy of `flops.ideal.flops`. |
| `flops_actual` | Measured FLOPs. **Currently always `null`** in timing mode — the worker passes `flops_actual=None`; the observed count comes from `ledger` mode instead. |
| `seconds` | The **median** of `timing.device_compute`. |
| `roofline_flops_per_s` | Machine bound. `null` unless supplied; `results/machine.json` is not currently joined in automatically. |
| `algorithmic_overhead` | `A = flops_actual / flops_ideal`, ≥ 1 expected. `null` while `flops_actual` is. |
| `execution_efficiency` | `E = flops_actual / (t · R)`, ≤ 1 expected. |
| `overall` | `flops_ideal / (t · R) = E / A`. |

So in a timing-mode file this block is mostly nulls by construction; that is not
a failed run. `A` for a real run comes from the next block.

### `flops.ledger` — `ledger` and `all` modes only

What the library *actually* computed, from intercepting NumPy/SciPy FFT, GEMM
and transcendental entry points for one propagation.

| key | notes |
|---|---|
| `flops_total`, `tops_total` | Summed over `sequence`. |
| `n_calls` | Length of `sequence`. |
| `histogram` | `{"numpy.fft.fft2(4096, 4096)": 1, …}` — op-plus-input-shape → call count. |
| `flops_by_op` | Op name → summed FLOPs. |
| `sequence` | Ordered `{op, in, out, dtype, flops}`. The most valuable artifact here: diffing two adapters on one case turns an unexplained 1.8× into a structural difference — a stray transform, a rebuilt kernel matrix, an array one power of two too large — which is filable as an issue. |
| `limitations` | Machine-readable caveats, currently the `@`-bypass below. |

`flops.algorithmic_overhead` (sibling of `ledger`, not inside it) is
`ledger.flops_total / ideal.flops`, written only when the ledger caught
something.

**Known under-count:** `A @ B` between plain ndarrays calls
`ndarray.__matmul__` in C and never routes through `np.matmul`, so it is
invisible. FFT calls are caught reliably; MFT-heavy codes under-report GEMMs.
The analytic model remains the primary FLOP source.

## `accuracy`

Absent in `gradient` mode. Fields come from `validate.compare`, which fits the
two differences that are legitimately allowed to differ between codes and
tolerates nothing else.

| key | notes |
|---|---|
| `rel_l2` | Relative L2 residual **after** the complex scale fit. This is the gated number. |
| `scale_abs` | \|α\| — normalisation relative to the reference. `256.0` for prysm here is not an error; codes disagree about whether the PSF sums to pupil energy, peaks at 1, or carries `dx²`. Reported rather than discarded because a 4× normalisation difference is worth knowing. |
| `scale_phase_rad` | `arg(α)` — global piston, physically irrelevant. |
| `conjugated` | Whether the field matched the *conjugate* reference better. `exp(+ikz)` vs `exp(−ikz)` is a convention, not an error. |
| `peak_ratio` | Peak intensity ratio, test/reference. Scales as `scale_abs²`. |
| `peak_offset_px` | `[dy, dx]` — a JSON **array** of two ints (row, column). Non-zero almost always means a grid-centring convention mismatch rather than a propagation error, which is why it is reported separately from `rel_l2`. |
| `gate` | `pass` \| `fail`, against the case's `accuracy.max_rel_l2`. |
| `reference` | `internal_mft` (direct double-precision matrix DFT sharing the case discretisation; ~1e-13 expected — this is the gate) or `analytic_airy`. |

### `accuracy.physics_check_analytic_airy`

Written only for cases with no aberration coefficients. `rel_l2`, `peak_ratio`
and a `note`. **Ungated, by design**: the analytic Airy pattern is the
continuous-aperture answer while every adapter propagates a pixelated hard-edged
mask, so ~8.8e-4 at N=1024 is expected and correct. Gating on it would fail
everyone. Do not treat this `rel_l2` as an error metric to rank on.

## `gradient_accuracy` — `gradient` mode only

| key | notes |
|---|---|
| `max_rel_err` | Max per-component relative error against the reference gradient. |
| `cosine_similarity` | Direction agreement. Not redundant with the above: a uniform factor of 2 — the classic Wirtinger slip — can hide under a loose per-component tolerance. |
| `scale_ratio` | Median `g_test / g_ref`. A value near `2` or `−1` indicates a Wirtinger convention mismatch, and the failure `reason` says so explicitly. |
| `gate` | `pass` if `max_rel_err ≤ 1e-6` **and** `1 − cos ≤ 1e-9`. |
| `reference` | `central_differences`. |
| `loss_adapter`, `loss_reference` | The scalar loss from each, for sanity-checking that both differentiated the same objective. |

## `retrieval` and `forward_accuracy` — `kind: phase_retrieval` only

On this board one timed iteration is a **whole nonlinear optimisation**, not a
propagation, so `timing` alone is unreadable: a row is
`(forward-model evaluations) × (cost per evaluation)` and the two factors are
independent. Both are recorded. See
[phase_retrieval_board.md](phase_retrieval_board.md).

`accuracy` on these results is the recovered coefficient vector, not a field:
`rel_l2` is the relative error over the **observable** modes, `metric` is
`coefficient_relative_l2`, and `quantity` is `zernike_coefficients`.

| key | notes |
|---|---|
| `retrieval.theta` | The recovered coefficients, length `retrieval.count`, waves RMS, Noll ordered from `first_noll`. |
| `retrieval.n_iterations` | Optimiser iterations. |
| `retrieval.n_fev` | **Forward-model evaluations.** On the numerical board this includes the finite-difference probes — `P+1 = 12` per gradient — which is the entire point of the board. dLux derives this from optax's line-search counter rather than measuring it (its loop is one compiled XLA program with no host-side call to count); `n_fev_note` says so on those rows. |
| `retrieval.n_jev` | Gradient evaluations. |
| `retrieval.loss_initial`, `loss_final`, `loss_reduction` | The loss is dimensionless and identically scaled across codes — each PSF is divided by the peak of that code's own observed PSF — so these are comparable between adapters. Exactly zero at the truth. |
| `retrieval.converged` | Whether the optimiser met `ftol`/`gtol` rather than hitting `max_iterations`. |
| `retrieval.seconds_per_forward_model` | `timing` median ÷ `n_fev`. The number that makes two rows comparable. |
| `retrieval.optimizer`, `forward_model` | Free text naming what actually ran. dLux's says `optax.lbfgs`, which is L-BFGS and **not** L-BFGS-B — optax has no box-constrained variant, and nothing here is bounded. |
| `accuracy.coefficient_rel_l2_all` | The same error **with piston included**. A PSF cannot see piston (`dL/dθ₁ ≡ 0`), so the optimiser leaves it at its starting value and the gate excludes it. |
| `accuracy.twin_rel_l2` | Distance to the twin solution `−φ(−x)`. If this is *smaller* than `rel_l2`, the code found the other valid answer rather than computing something wrong — the failure `reason` says so explicitly. |
| `accuracy.max_coefficient_error_waves` | Worst single observable mode, in waves. |
| `forward_accuracy` | **Untimed** gate: this code's own PSF at the truth coefficients, against the harness reference, after fitting one overall scale. Each code fits data its *own* forward model produced, so without this a wrong pupil would converge beautifully onto its own private physics and `accuracy` would pass. Shape is a normal `Comparison` with `quantity: psf_intensity`. |
| `forward_accuracy.opd_sign_convention` | `"+"` or `"-"`. Which way this code carries OPD into phase. **POPPY is `"-"`** — its PSF at `+θ` reproduces the reference at `−θ` to 3.4e−16. Both conventions are in use and neither is wrong; it does not affect the recovered coefficients, but it sets the sign of any wavefront read out of that code. |
| `flops.ideal.flops` | **Always `0.0` here**, which is what suppresses the ideal line and row. The per-evaluation floor *is* derivable and appears in `flops.ideal.detail`; how many evaluations the optimiser needs is measured, not predicted. |

## `timing`

| key | notes |
|---|---|
| `warmup`, `repeats` | From the case's `execution` block (defaults 3 and 25). Warm-up fills FFTW wisdom, the cuFFT plan cache and the NVRTC kernel cache — real cost, but setup cost. |
| `device_compute` | Per-repeat seconds for `propagate()` **including `sync()`**. Without the sync inside the clock, an async backend returns before any arithmetic has happened and reports a ~100× speedup that is entirely dispatch latency. |
| `host_available` | `device_compute` + the `to_host()` copy for that repeat. Equal to `device_compute` in `gradient` mode. Device→host transfer is measured, but never conflated with compute. |
| `cpu_seconds`, `cpu_wall_ratio` | Process CPU time (user+sys) over the timed region, and that divided by elapsed wall time — i.e. **how many cores the run actually used**. `~1.0` is single-threaded, `~k` is k cores. Present because `threads` in the config is a *request* that one backend ignored for the life of this repo: XLA honours no `*_NUM_THREADS` variable and no `XLA_FLAGS` setting reaches its pool, so dLux ran on ~10 cores on boards labelled `threads=1`. Affinity pinning now enforces the request and this verifies it. Absent on results written before the measurement existed. |
| `traced` | `true` only under `--mode trace`. **Traced runs must be excluded from every comparison** — VizTracer's overhead is per-Python-call, so it penalises loop-heavy codes far more than vectorised ones. `report.render_text` filters on this. |
| `unit` | Always `"s"`. |
| `device_compute_stats`, `host_available_stats` | `{min, median, mean, p95, iqr}`, or `{}` if the sample list is empty. |

The stats are index-based on the sorted sample, not interpolated percentiles:
`p95 = s[min(n−1, int(0.95·n))]`, and `iqr = s[int(0.75·n)−1] − s[int(0.25·n)]`
(`0.0` when `n < 4`). For the default `repeats: 25` that makes `p95` the 24th of
25 samples. `min` is the least noise-contaminated estimator here; `iqr` is the
one to check before believing a small difference between two adapters. The raw
lists are kept so any other estimator can be computed after the fact.

## `memory`

| key | notes |
|---|---|
| `tracemalloc_peak_bytes` | Peak traced allocation for one untimed iteration. NumPy array data **is** visible (NumPy registers under `np.lib.tracemalloc_domain`); FFTW/MKL scratch buffers and anything on a GPU are **not**. |
| `rss_peak_bytes` | `ru_maxrss` high-water. Covers the native scratch tracemalloc misses. |
| `device_bytes` | `null` unless the adapter implements `device_memory()`. RSS is meaningless for JAX and CuPy, so device-owning adapters must supply this themselves. |
| `tool` | `tracemalloc+rusage`, with `+device` appended when `device_bytes` is present. |

**Platform caveat:** `rss_peak_bytes` is `ru_maxrss × 1024`, which is correct on
Linux (where `ru_maxrss` is KiB) but **1024× too large on macOS**, where the
kernel already reports bytes. Check `machine.platform` before comparing RSS
across hosts.

## `trace` — `trace` mode only

`path` (absolute path to `trace.json` beside the result), `tool` (`viztracer`),
`mode` (`full`), plus, when summarisation succeeded:

- `self_time_frac` — category → fraction of total self time, ordered descending.
  Categories come from `adapters/<name>/categories.yaml` or the defaults in
  `trace_summary.py` (`fft`, `gemm`, `elementwise`, `alloc`, `units`, `coords`,
  and `python_overhead` for everything unmatched). This is what turns a flame
  graph into a comparable number, and it survives being run on another machine.
- `total_self_time_s`, `top_functions` — 20 entries of `{name, self_s}`.

If summarisation raised, `summary_error` holds `"TypeError: …"` instead and the
trace file itself is still on disk.

## The `scan` block

A case carrying a `scan:` block (`mft_array_scan` is the shipped one) is measured
at several array sizes **in one worker process** and written to **one**
`result.json`. Points measured in separate processes can differ by more than the
effect being measured — thread placement, heap state, a machine that got busier —
and the product of a scan is the *shape* of the curve, so they are measured
together against one configured adapter.

The consequence for a parser: **`timing`, `accuracy`, `setup`, `flops`, `memory`
and `trace` are not at the top level of a scan result.** They are inside each
point. `adapter`, `machine`, `provenance` and `backend` stay at the top and are
shared by every point, which is also the proof that the whole curve ran on one
machine with one resolved backend.

```json
"scan": {
  "parameter": "n_pupil",
  "values": [128, 256, 512, 1024, 2048],
  "points": [
    {"scan_value": 128, "case_id": "mft_array_scan@n_pupil=128",
     "n_pupil": 128, "n_across": 128, "n_focus": 128,
     "setup": {…}, "flops": {…}, "accuracy": {…}, "timing": {…}, "status": "ok"}
  ]
}
```

| key | notes |
|---|---|
| `parameter` | `n_pupil` (pupil array size; the focal grid is held fixed) or `n_focus` (focal grid size via the field extent). |
| `values` | The measured sizes, ascending. Ascending is deliberate: cheap points first, so a scan whose largest size exhausts memory still yields a usable curve. |
| `points[]` | One entry per size, in the same order as `values`. |

Each point carries its identity — `scan_value`, the derived `case_id`
(`<parent>@<parameter>=<value>`), `n_pupil`, `n_across`, `n_focus` — then exactly
the blocks that mode would have produced for a plain case, plus its **own**
`status` and, on failure, `reason` and `traceback`.

Three properties worth relying on:

- **Every point is gated independently.** A code that is accurate at N=128 and
  drifts at N=2048 produces a hole in the curve rather than a timing row that
  quietly means something different from its neighbours.
- **A failing point does not abort the scan.** The usual reason a large point
  fails is memory, and that is itself the finding.
- **`flops.ideal` differs per point**, since it is derived from that point's
  case. It is the denominator to check a measured curve's slope against.

### Reading a scan without `dragrace report`

```python
r = json.loads(path.read_text())
for p in (r.get("scan") or {}).get("points", []):
    if p["status"] != "ok":
        continue                                    # gate/status still applies per point
    n = p["scan_value"]
    median = p["timing"]["device_compute_stats"]["median"]
```

`report.aggregate` does this for you and emits **one row per point**, carrying
`scan_param` and `scan_value` (both `null` for plain results). `report.best_points`
then collapses a point measured under several `run_id`s to its **fastest** median
— noise adds time to a measurement, it does not remove it — so a curve shows one
point per size instead of zigzagging between run days.

## Measurement contracts

`measurement_contract` says what the clock was measuring, and it is the field to
check before comparing two results that otherwise look alike — same adapter, same
case, same machine, same schema.

| value | timed call |
|---|---|
| `idiomatic-v1` | The call the library's own documentation puts in front of a user (`OpticalSystem.calc_psf`, `Wavefront.focus_dft`, `propagate_dft`, `propagate_mono`), with everything the API permits hoisting already in `build()`. |
| `primitive-v1` | The library's transform entry point (`poppy.matrixDFT.perform`, `lentil.fourier.dft2`, a hand-written `jnp` kernel). Superseded. |

The gap is not cosmetic: POPPY at N=1024 measures 36 ms under `primitive-v1` and
66 ms under `idiomatic-v1`, because `calc_psf` also rebuilds the wavefront,
re-applies every optic, normalises, and assembles a FITS HDUList. Both numbers
are correct; they answer different questions.

`report.aggregate` exposes this as a `contract` column, defaulting to
`primitive-v1` when the marker is absent. `render_text` prints one table per
contract and warns when results span more than one, `best_points` never
collapses across contracts, and `dragrace plot` writes one figure per contract.
A consumer doing its own grouping must include it, or a stale row will land on a
current curve.

## Which blocks appear in which mode

| block | timing | memory | ledger | trace | gradient | all |
|---|---|---|---|---|---|---|
| `accuracy` | ✓ | ✓ | ✓ | ✓ | — | ✓ |
| `gradient_accuracy` | — | — | — | — | ✓ | — |
| `timing` | ✓ | — | — | — | ✓ | ✓ |
| `flops.efficiency` | ✓ | — | — | — | — | ✓ |
| `flops.ledger` | — | — | ✓ | — | — | ✓ |
| `memory` | — | ✓ | — | — | ✓ | ✓ |
| `trace` | — | — | — | ✓ | — | — |

`flops.ideal`, `setup`, `adapter`, `machine`, `provenance` and `backend` are
present in every mode — but only on results that got past the support and
backend checks. A `status: unsupported` result stops before `setup`. On a scan
result every row of this table applies **inside each point** instead.

## Parsing rules

### Skipped results have a different shape

`runner.run_one` writes `status: skipped` files itself, before any worker
starts. They carry only `status`, `reason`, `case_id`, `config_id`, `mode` and
`adapter: {name}` — **no `schema_version`, no `machine`, no `provenance`**. A
parser that keys on `machine.id` must tolerate their absence rather than crash
or silently bucket them under `None`.

Separately, `run_one` adds `_interpreter_note` to the dict it *returns*, after
reading the file back. That key is never written to disk.

### Numeric edge cases

`json.dumps` is called with `default=str`, and Python emits `Infinity`, `-Infinity`
and `NaN` as bare literals. Both matter:

- `arithmetic_intensity` is `Infinity` when `bytes == 0`; `rel_l2` and
  `peak_ratio` are `Infinity` when the reference has zero norm or zero peak.
  Strict JSON parsers in other languages reject these literals — enable the
  non-standard-number extension, or pre-substitute.
- A value that is not JSON-serialisable is **stringified rather than dropped**.
  The harness casts through `float()` at every boundary it owns, but an adapter
  returning a NumPy scalar in `versions` or `resolved` can produce a string
  where a number is expected. Coerce, don't assume.

### Read the gate before the clock

A run that fails its accuracy gate returns before timing, so `timing` is absent
rather than present-and-invalid. Any consumer that reaches for
`timing.device_compute_stats.median` without checking `status == "ok"` will be
reading `None`, not a slow measurement.

### Never merge across `machine.id`

Stated once more because it is the rule most likely to be broken by a
quick script: group by `machine.id` first, always. See
[methodology.md](methodology.md#confounds-that-remain) for why MKL on AMD makes
this a correctness issue rather than a stylistic one.

### A minimal reader

`report.aggregate` is the reference implementation, and its defensive idiom is
worth copying — every level is defaulted, because any block can be absent:

```python
t     = r.get("timing", {}) or {}
stats = t.get("device_compute_stats", {}) or {}
acc   = r.get("accuracy", {}) or {}
median = stats.get("median")
```

The `or {}` after `.get(k, {})` is not redundant: a key can be present and
`null`.

## `results/index.json`

Written by `dragrace report`: a flat array of one row per `result.json` found
anywhere under `results/`, for loading into a dataframe.

`adapter`, `case`, `config`, `mode`, `status`, `machine`, `cpu`, `contract`,
`grid_centering`, `scan_param`, `scan_value`, `median_s`, `min_s`, `p95_s`,
`iqr_s`, `traced`, `rel_l2`, `gate`, `ideal_gflop`, `ledger_gflop`,
`A_overhead`, `build_s`, `first_call_s`, `mem_peak_mib`, `grad_gate`, `reason`,
`path`.

Four things to know. **`contract` must be part of any grouping key** — see
above; a `primitive-v1` row and an `idiomatic-v1` row are different measurements
wearing the same adapter name. The GFLOP and memory columns are built with an
`x / 1e9 or None` idiom, so **a genuine zero becomes `null`**. A scan result
contributes **one row per point**, distinguished by `scan_value` — so `case`
alone is not a unique key for a scan, and grouping by it without `scan_value`
silently averages five array sizes together. And `path` points back at the full
`result.json` — the index is a navigation aid, not a replacement for the raw
file, and anything not in the column list above (per-repeat timings, the backend
snapshot, the ledger sequence) has to be read from there.

## `results/machine.json`

Written by `dragrace machine`. Peaks are **measured, not spec-sheet**:
theoretical peak FLOP/s is not a bound any of these codes could reach, so
scoring against it would make every adapter look uniformly terrible and hide the
differences between them.

| key | notes |
|---|---|
| `peak_flops_per_s` | Best of several `complex128` GEMMs at `gemm_size`, priced at `8n³`. |
| `peak_bandwidth_bytes_per_s` | STREAM triad `a = b + s·c`, 3 arrays touched per element. |
| `gemm_size` | `n` used (1024 under `--quick`, else 2048). |
| `threads_seen` | `detect_thread_counts()` at measurement time — the peak is only meaningful alongside the thread count that produced it. |
| `ridge_point_flops_per_byte` | `peak_flops / peak_bandwidth`. A case whose `arithmetic_intensity` exceeds it is compute-bound. |
| `note` | `measured zgemm + STREAM triad; not theoretical peak`. |

This file is not automatically joined into `result.json`; supplying it is what
would populate `flops.efficiency.roofline_flops_per_s` and the `E` and `overall`
columns.

## Changing the schema

`schema_version` is `1` and lives in `worker.SCHEMA_VERSION`. Adding a key is not
a version bump; removing or repurposing one is. Because results are meant to
accumulate across machines and months, prefer adding a new key to redefining an
existing one — an old file with a missing key is parseable, an old file whose
`rel_l2` means something different is not.
