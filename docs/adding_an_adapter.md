# Adding an adapter

An adapter is the only thing a contributor writes. It lives in
`adapters/<name>/adapter.py` and implements a small contract.

## The contract

```python
from dragrace.adapter import Adapter, Unsupported, register

@register("mycode")
class MyAdapter(Adapter):
    status = "unverified"        # -> "verified" once run against a real install
    reviewed_by = ""             # fill in before publishing results

    def versions(self) -> dict[str, str]: ...
    def supports(self, case, config) -> bool | Unsupported: ...
    def configure(self, config) -> bool | Unsupported: ...
    def resolve_backend(self) -> dict: ...

    def build(self, case, config): ...       # UNTIMED
    def propagate(self, state): ...          # TIMED
    def sync(self, result) -> None: ...      # blocks; called INSIDE the clock
    def to_host(self, result) -> np.ndarray: ...
    def teardown(self, state) -> None: ...
```

### `build()` — untimed

Grids, DFT kernels, FFT plans and wisdom, JIT compilation, staging inputs
on-device. Anything a real user would hoist out of a loop. Its cost is still
reported (`setup.build_s`), so hoisting aggressively is not cheating — it is the
thing being measured, separately.

### `propagate()` — timed

Must return a `(N_f, N_f)` field on the canonical focal grid in the case's
dtype. May return an unmaterialised handle.

### `sync()` — the one that matters

Blocks until the result physically exists.

```python
def sync(self, result):          # CuPy
    import cupy as cp
    cp.cuda.Stream.null.synchronize()

def sync(self, result):          # JAX -- must cover every pytree leaf
    import jax
    jax.block_until_ready(result)
```

Getting this wrong does not produce an error. It produces a ~100× speedup that
is entirely dispatch latency, and it looks like a record. `check_sync_scaling`
flags adapters whose time fails to grow with problem size, but do not rely on it.

### `supports()` returns a *reason*

```python
return Unsupported("PROPER is FFT/Fresnel-based; no matrix-DFT path")
```

Not `False`. The report renders an honest matrix of *why* each hole exists;
"not applicable" and "not implemented" are different findings and both are more
useful than a blank cell.

### `resolve_backend()` — never assume

Report what the library **actually** resolved: array module, FFT module, device,
dtype. HCIPy probes for `mkl_fft` at import; POPPY's `accel_math` toggles fall
back silently; prysm swaps its whole array module. The worker compares your
report against the config and **refuses the run** on a mismatch, because a
mislabelled result is worse than no result.

## Gradient support (optional)

```python
def supports_gradient(self) -> bool | Unsupported: ...
def build_gradient(self, case, config): ...
def gradient(self, state):        # -> (loss: float, grad: real ndarray, shape (P,))
def gradient_theta(self, state) -> np.ndarray:   # the point of differentiation
```

`gradient_theta` lets the harness evaluate central differences at the same θ.
Without it the correctness gate is skipped, which means the board will happily
publish a wrong gradient.

## Checklist

1. `adapters/<name>/adapter.py` implementing the contract.
2. `adapters/<name>/capabilities.yaml` — declared devices, backends, profiler.
3. `adapters/<name>/categories.yaml` — regex map for trace attribution
   (optional; a sensible default is used otherwise).
4. Run `dragrace doctor` — your adapter should appear with its support matrix.
5. Run `dragrace run --case mft_n1024_q4 --adapter <name> --mode all`. The
   accuracy gate must pass before any timing is recorded.
6. Run `dragrace ledger --case fft_n1024_q4 --adapter <name>` and check the
   primitive count against `docs/flop_model.md`. An unexpected extra transform
   is the most common adapter bug.
7. Set `status = "verified"`.
8. **Ask the library's maintainer to review it** and record them in
   `reviewed_by`. Adapter authors are not neutral parties — see the confounds
   section of `docs/methodology.md`.

## Common mistakes

- **Building the pupil field yourself.** Use `dragrace.grid.pupil_field(case)`.
  Aperture antialiasing and Zernike normalisation differ between codes, and
  letting them differ puts rasterisation cost into a propagation comparison.
- **Half-pixel grid offsets.** Both grids centre at index `N//2`. The accuracy
  comparison reports the PSF peak offset in pixels; a non-zero offset with a
  large `rel_l2` is a centring bug, not a propagation bug.
- **Caching a callable at configure time.** `self._fft2 = np.fft.fft2` holds the
  unpatched function and escapes the ledger. The worker works around it by
  re-running `configure()`/`build()` inside the instrumented context, but
  resolving through the module at call time is cleaner.
- **Normalising to match the harness.** Don't. The validator fits and reports a
  complex scale factor; your library's own convention is fine and is worth
  recording.
