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
    grid_centering = "pixel"     # or "interpixel" -- see conventions.md

    def versions(self) -> dict[str, str]: ...
    def supports(self, case, config) -> bool | Unsupported: ...
    def configure(self, config) -> bool | Unsupported: ...
    def resolve_backend(self) -> dict: ...

    def build(self, case, config): ...       # UNTIMED
    def propagate(self, state): ...          # TIMED
    def sync(self, result) -> None: ...      # blocks; called INSIDE the clock
    def to_host(self, result) -> np.ndarray: ...
    def complex_field(self, state, result) -> np.ndarray: ...   # gate only, UNTIMED
    def teardown(self, state) -> None: ...
```

### Which API to call — the first decision

**Call what the library's documentation puts in front of a user**, not its
internal transform. If the tutorials build an object and call a method on it,
that object goes in `build()` and that method is what `propagate()` calls. An
adapter that reaches past the documented API to time a kernel measures something
no user experiences: POPPY's `matrixDFT.perform` is 20 ms at N=1024 where its
documented `calc_psf` is 30, and the difference is real cost a user pays.

Results are stamped `measurement_contract: idiomatic-v1` to say this is what was
measured. If you deliberately time something else, change the contract string —
never leave two meanings under one name.

### `build()` — untimed

The reusable model object, plus grids, DFT kernels, FFT plans and wisdom, JIT
compilation, staging inputs on-device. Anything a real user would hoist out of a
loop **and that the library actually lets them hoist**. Its cost is still
reported (`setup.build_s`), so hoisting aggressively is not cheating — it is the
thing being measured, separately.

The corollary matters as much: work the API does *not* let a user hoist stays in
`propagate()`, even when a competitor hoists it. lentil's `Wavefront` is consumed
by its propagation, so constructing it is timed; POPPY re-applies every optic
inside each `calc_psf`. Charging those honestly is the point.

### `propagate()` — timed

One PSF, as documented. May return an unmaterialised handle, and may return
whatever the library returns — including an intensity array or a FITS HDUList.

### `complex_field()` — untimed, for the gate only

Defaults to `to_host()`, which is right whenever the documented call returns a
field. Override it when the documented call returns *intensity*: gating on
intensity alone throws away the phase-sign and normalisation diagnostics. Use
whatever the library documents for recovering the field (`calc_psf(
return_final=True)`, `propagate_mono(..., return_wf=True)`) in a separate call,
so its extra cost never reaches the clock. It must be the same propagation — this
is not a licence to compute the answer a second way.

### `grid_centering` — declare, do not shim

`"pixel"` (default, origin on sample `N//2`) or `"interpixel"` (origin between
the middle samples). Declare what your library actually produces; the harness
builds the reference and the injected pupil to match. Getting it wrong fails the
gate at `rel_l2 ≈ 0.28` with a one-pixel peak offset, which is the intended
behaviour. See [conventions.md](conventions.md) for why declaring beats
correcting with a half-pixel source offset.

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
