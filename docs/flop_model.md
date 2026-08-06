# The FLOP model

Every claim this repo makes about algorithmic efficiency rests on the
conventions below, so they are stated explicitly and are open to dispute. The
derived values are hand-checked in `tests/test_ideal_flops.py`; if you think a
number here is wrong, that test file is where to argue.

## Counting conventions

| primitive | real FLOPs |
|---|---|
| complex add | 2 |
| complex multiply | 6 (4 mul + 2 add) |
| complex FMA | 8 |
| `zgemm(M,K,N)` | 8·M·K·N |
| 1-D C2C FFT, length N | 5·N·log₂N |
| 2-D C2C FFT, N₁×N₂ | 5·N₁·N₂·log₂(N₁N₂) |

Note that `fft_2d` already covers **both** the row and column passes:
5·N²·log₂(N²) = 10·N²·log₂N. Multiplying by two on top of it is a factor-of-two
error, and was a real bug during development — hence
`test_fft_2d_counts_both_passes_once`.

### Two currencies, kept separate

`flops` are FMA-class real floating-point operations. `tops` are transcendental
evaluations (`cexp`, `sin`, `cos`), counted **separately** because their cost
relative to a FLOP ranges over roughly 10–40× depending on whether the build
gets SVML/libmvec vectorisation. Folding them together with a fudge factor would
silently bake a machine assumption into a supposedly hardware-independent
metric.

`bytes` is tracked as a third currency, for the memory-bound operations
(padding, fftshift, elementwise passes), which is what makes roofline
classification possible.

## The physics floor

Given N_p pupil samples per side, N_D samples across the aperture diameter,
q samples per λF/D, field of view W λ/D, and N_f = q·W focal samples per side:

**Matrix DFT** — two GEMMs, `(N_f×N_p)·(N_p×N_p)` then `(N_f×N_p)·(N_p×N_f)`:

```
flops = 8·N_p·N_f·(N_p + N_f)
tops  = 2·N_p·N_f            # only when kernels are rebuilt per call
```

**FFT Fraunhofer** — padded to N = q·N_D:

```
flops = 5·N²·log₂(N²) = 10·N²·log₂N
```

**Fresnel transfer-function / angular spectrum** — two FFTs plus one
elementwise multiply:

```
flops = 2·(10·N²·log₂N) + 6·N²
tops  = N²                   # transfer function, if rebuilt
```

**Chirp-Z (Bluestein)** — approximate, with L = next_pow2(N_p + N_f − 1):

```
flops ≈ (N_p + N_f)·(3·5·L·log₂L + 2·6·L)
```

Included because CZT is the theoretical floor for arbitrary output sampling.
prysm exposes it (`prepare_executor(..., kind='czt')`), so unlike the other five
codes it is directly measurable here.

**Basis rendering** — P modes over the pupil, plus the phasor:

```
flops = 2·P·N_p²      tops = N_p²
```

At large P this dominates everything else: P=1024 modes over a 1024² grid is
~2 GFLOP, larger than the propagation itself. Whether it is hoisted into
`build()` is pinned by `case.basis_caching`, or the gradient board quietly
becomes a basis-caching benchmark.

**Reverse mode** — the adjoint of a linear propagation costs about the same as
the forward pass, so the gradient floor is ~2× forward **regardless of P**. That
P-independence is the entire value proposition of reverse mode, and it is what
the board's P-sweep tests.

## Worked example: why `algorithm_class` is part of the case

N_D = 1024, q = 4, so an FFT must pad to 4096 and costs 2.013 GFLOP whatever the
field of view. The MFT's cost grows with the output extent:

| field of view | N_f | MFT | FFT |
|---|---|---|---|
| W = 8 λ/D | 32 | 0.28 GFLOP | 2.01 GFLOP |
| W = 32 λ/D | 128 | **1.21 GFLOP** | **2.01 GFLOP** |
| W = 256 λ/D | 1024 | 17.2 GFLOP | 2.01 GFLOP |

The ranking inverts inside the parameter range people actually use. A benchmark
reporting a single "MFT codes are faster" conclusion would be reporting its
choice of W, not a property of the codes. Hence: bin the leaderboard by
algorithm class, and treat cross-class comparison as "cost to produce a PSF
meeting this (q, W, ε) spec".

## Measuring actual FLOPs

Three tiers, in order of preference.

**Tier 1 — XLA (dLux).** `jax.jit(f).lower(x).compile().cost_analysis()` gives
FLOPs directly; `.as_text()` gives the HLO with every op and shape. Authoritative
and free.

**Tier 2 — call interception (the other six).** `dragrace.flops.ledger` wraps
`numpy.fft`, `scipy.fft`, `np.matmul/dot/einsum/tensordot` and the
transcendentals, pricing each call from its observed shape. Validated against
the baseline adapter: the FFT case recovers exactly 2.0133 GFLOP against an
ideal of 2.0133, i.e. A = 1.000.

Two limitations, both real:

- `A @ B` between plain ndarrays calls `ndarray.__matmul__` in C and bypasses
  `np.matmul`. MFT-heavy codes therefore under-count GEMMs.
- An adapter that caches a callable (`self._fft2 = np.fft.fft2`) at configure
  time holds the *unpatched* function. The worker works around this by re-running
  `configure()` and `build()` inside the patched context and then calling
  `led.reset()`, so only the propagation is priced.

**Tier 3 — hardware counters**, for validating tiers 1–2 rather than routine
use: `perf stat -e fp_arith_inst_retired.*` weighted by lane width (needs
`perf_event_paranoid ≤ 2`, so it will not work in most CI containers), LIKWID's
`FLOPS_DP`, or `ncu` on the GPU side. Run once per machine against the baseline
adapter to confirm the tier-2 model is not systematically off.

## The efficiency decomposition

```
A = flops_actual / flops_ideal          algorithmic overhead   (≥ 1 expected)
E = flops_actual / (t · R)              execution efficiency   (≤ 1 expected)
overall = flops_ideal / (t · R) = E / A
```

where `R = min(peak_flops, AI · peak_bandwidth)` and `AI = flops / bytes`.

Peaks are **measured**, not taken from a spec sheet (`dragrace machine`: a large
zgemm for peak FLOP/s, a STREAM triad for bandwidth). Theoretical peak is not a
bound any of these codes could reach, so scoring against it would make every
adapter look uniformly terrible and hide the differences between them.

`A` is hardware-independent and answers "is this code doing more arithmetic than
the physics requires". `E` answers "is it doing that arithmetic well on this
machine". Ranking on wall time alone conflates the two.
