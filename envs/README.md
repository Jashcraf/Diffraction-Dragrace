# Environments

## Quick start

```bash
cd Diffraction-Dragrace
conda env create -f environment.yml      # ~5-10 min
conda activate dragrace
bash scripts/setup_env.sh                # PROPER + activation-time variables
```

That one environment runs **every adapter on CPU**: PROPER, POPPY, HCIPy,
prysm, lentil, dLux. If you only ever run the CPU boards, you are done.

## Why there is more than one environment

The benchmark treats the backend as a first-class variable — a run is
`(case × config × adapter)`, and `config` names the device, the FFT library,
the BLAS, the thread count and the precision. Some of those cannot vary inside
a single environment:

| constraint | consequence |
|---|---|
| NumPy links exactly one BLAS | OpenBLAS and MKL builds need separate envs |
| `mkl_fft` is picked up by *detection* in some libraries | it must be absent from the OpenBLAS env, or baseline runs silently become MKL runs |
| pip's `jax[cuda12]` vendors its own CUDA libraries | it must not share an interpreter with conda's CUDA runtime for CuPy |

This costs nothing operationally: the runner already launches each adapter in
its own interpreter, so it can span environments within a single sweep.

| file | env name | serves | config ids |
|---|---|---|---|
| `../environment.yml` | `dragrace` | all six adapters, CPU, OpenBLAS | `cpu_numpy_1t`, `cpu_openblas_*`, `cpu_pyfftw_1t` |
| `cpu-mkl.yml` | `dragrace-mkl` | all six adapters, CPU, MKL BLAS + `mkl_fft` | `cpu_mkl_1t`, `cpu_mkl_8t` |
| `gpu-cupy.yml` | `dragrace-gpu-cupy` | POPPY, prysm on CUDA | `gpu_f64`, `gpu_f32` |
| `gpu-jax.yml` | `dragrace-gpu-jax` | dLux on CUDA | `gpu_f64`, `gpu_f32` |

Create whichever you need:

```bash
conda env create -f envs/cpu-mkl.yml   && conda activate dragrace-mkl       && bash scripts/setup_env.sh
conda env create -f envs/gpu-cupy.yml  && conda activate dragrace-gpu-cupy  && bash scripts/setup_env.sh
conda env create -f envs/gpu-jax.yml   && conda activate dragrace-gpu-jax   && bash scripts/setup_env.sh
```

## The two packages conda cannot install

**PROPER** is distributed as a source zip, not through PyPI or conda-forge, so
it is always a local-path install. `scripts/setup_env.sh` autodetects
`~/proper_v*_python`; override with `--proper-dir /path/to/proper_vX.Y.Z_python`.
It builds three small C extensions, so a C toolchain must be present
(`/usr/bin/gcc` is sufficient; otherwise uncomment `c-compiler` in
`environment.yml`).

**prysm** is on PyPI, but only up to **0.21.1** (checked 2026-08-05), which
predates the adjoint API the gradient board is built on — `focus_dft_adjoint`,
`focus_adjoint`, `angular_spectrum_adjoint`, `to_fpm_and_back_adjoint`, in the
`prysm.propagation` *subpackage* (it became a package rather than a module in
0.22). So the env files install prysm from git instead:

```yaml
- "prysm @ git+https://github.com/brandondube/prysm.git"
```

The forward boards run fine on `prysm==0.21.1` if you would rather have a
pinned release and no gradient board. For published results, pin a commit
rather than tracking master. To use a local checkout:

```bash
bash scripts/setup_env.sh --local-prysm=$HOME/prysm
```

`setup_env.sh` reports at the end whether the adjoint API actually resolved.

## Backend notes that affect results

**PROPER's MKL FFT is independent of NumPy's BLAS — but conda can't express
that.** `prop_use_ffti()` dlopen's `libmkl_rt` directly through `ctypes`
(`proper/prop_ffti.py`, `proper/prop_dftidefs.py`) rather than going through
NumPy, so "MKL FFT + OpenBLAS BLAS" is a physically meaningful configuration.
conda-forge's `mkl` package, however, participates in the BLAS mutex and forces
`libblas=*=*mkl` — verified by solve, it is genuinely unsatisfiable alongside
the OpenBLAS pin. Two ways to get the configuration:

```bash
bash scripts/setup_env.sh --intel-fft     # PyPI MKL runtime, outside conda's mutex
# ...or just run PROPER's Intel-FFT configs in the dragrace-mkl env.
```

There is also a trap worth knowing about: PROPER builds the library path as
`os.path.join(MKL_DIR, 'libmkl_rt.so')` — the **unversioned** soname — while
both conda-forge and PyPI ship only `libmkl_rt.so.2`. `setup_env.sh` creates
the symlink and pins `MKL_DIR=$CONDA_PREFIX/lib`, without which PROPER's Intel
FFT path fails to load even with MKL correctly installed.

**BLAS matters more than FFT for half the suite.** lentil, prysm's `focus_dft`,
POPPY's `matrixDFT` and HCIPy's `MatrixFourierTransform` are `zgemm`-bound —
swapping `mkl_fft` in does nothing for them, while swapping OpenBLAS→MKL does.
FFT-bound paths (PROPER, POPPY Fresnel, the FFT legs of prysm/HCIPy) respond to
the opposite knob. Both axes are in the matrix for that reason.

**Thread counts are not set here.** `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and
friends belong to the config and are applied per-run by the runner. If they were
baked into `activate.d`, every result would silently depend on which shell you
launched from.

**JAX defaults are overridden at activation.** `setup_env.sh` writes
`JAX_ENABLE_X64=1` and `XLA_PYTHON_CLIENT_PREALLOCATE=false` into
`activate.d`. The first stops dLux from being benchmarked at complex64 against
everyone else's complex128; the second stops JAX from grabbing ~75% of device
memory at first use, which would make every memory measurement meaningless.
The complex64 board is opted into explicitly via the `gpu_f32` config.

## Verifying you got what you asked for

Silent backend substitution is the most likely way to produce wrong conclusions
with this suite, so verification is built in at two levels:

```bash
bash scripts/setup_env.sh --verify-only   # capability + BLAS report, no installs
dragrace doctor                           # same, plus per-adapter config support
```

Both print `threadpoolctl.threadpool_info()` verbatim — library paths on disk,
not package metadata. Every benchmark result records the same block, and the
worker **hard-fails a run when the resolved backend does not match the
requested one** rather than quietly reporting a mislabelled number.

## Reproducibility

The `.yml` files use ranges so they keep solving over time. For a result set you
intend to publish, freeze the exact solve alongside it:

```bash
conda env export --no-builds -n dragrace > envs/locks/dragrace-$(date +%Y%m%d).yml
pip freeze > envs/locks/dragrace-$(date +%Y%m%d).pip.txt
```

Every `result.json` already carries the resolved versions of the adapter and its
dependencies, so a lockfile is belt-and-braces rather than the only record.

## Verification status

Verified by conda dry-run solve on `linux-64`, 2026-08-05:

| env | status | notable resolved versions |
|---|---|---|
| `dragrace` | solves | python 3.11.15, numpy 2.4.6, scipy 1.17.1, poppy 1.1.1, hcipy 0.7.0, pyfftw 0.15.1, `libblas-3.11.0-8_h4a7cf45_openblas` |
| `dragrace-mkl` | solves | as above plus mkl 2026.1.0, mkl_fft 2.3.1, mkl-service 2.8.0, `libblas-3.11.0-8_h5875eb1_mkl` |
| `dragrace-gpu-cupy` | **not verified** | no NVIDIA driver on this machine |
| `dragrace-gpu-jax` | **not verified** | no NVIDIA driver on this machine |

Not verified anywhere: the `pip:` sections (no install was performed), and
PROPER's C-extension build. Existence and latest version were confirmed on PyPI
for prysm (0.21.1), lentil (0.8.9), dLux (0.15.1), vizplugins (0.1.3) and the
MKL runtime (2026.1.0).

## Caveat: MKL on AMD

This machine is an **AMD Ryzen 9 7900X (Zen 4, 24 threads)**. MKL selects its
kernel path from a CPU vendor check, and on non-Intel parts it has historically
dispatched to conservative code paths rather than the best available ones. The
`MKL_DEBUG_CPU_TYPE=5` workaround was removed in MKL 2020 and is not available
here.

The practical consequence for this suite: on Zen hardware, an "OpenBLAS beats
MKL" result for the `zgemm`-bound propagators (lentil, `focus_dft`,
`matrixDFT`, `MatrixFourierTransform`) is likely telling you something about
MKL's dispatch on AMD rather than something about the propagators. The same
board on a Xeon can invert. Two consequences for how results get reported:

- Every `result.json` records the CPU model, and `dragrace report` refuses to
  plot across machine fingerprints — so this cannot silently contaminate a
  cross-machine comparison.
- The MKL-vs-OpenBLAS board should carry the CPU vendor in its caption, and
  ideally be reproduced on an Intel part before any claim is drawn from it.

None of this affects the FFT axis (`pyfftw`, `mkl_fft` vs `numpy.fft`), the
algorithmic-overhead metric `A = flops_actual / flops_ideal`, or the FLOP
ledger — those are the hardware-independent parts of the design, which is
precisely why they are in it.

## Known friction

- **`dragrace` may already exist** on this machine from the earlier iteration of
  this repo. `conda env create` refuses to overwrite, so nothing is destroyed —
  either remove it (`conda env remove -n dragrace`) or pass `--name` to create
  under a different name.
- **`cuda-version=12.6`** in `gpu-cupy.yml` must be ≤ the maximum your driver
  reports (`nvidia-smi | head -3`). Lower it if the solve fails.
- **conda-forge only.** The `.yml` files set `nodefaults`; your global config
  currently points at `pkgs/main`, which ships an MKL-linked NumPy by default
  and would defeat the explicit BLAS pin. The channel list in each file
  overrides that, so no global change is needed.
