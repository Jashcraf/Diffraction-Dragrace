#!/usr/bin/env bash
#
# Diffraction-Dragrace -- post-create environment setup.
#
# Handles the pieces `conda env create` cannot: packages that live only on your
# filesystem (PROPER), optional editable installs of local checkouts (prysm),
# and the environment variables that must be set before the relevant library is
# first imported.
#
# Usage:
#   conda activate dragrace
#   bash scripts/setup_env.sh [options]
#
# Options:
#   --proper-dir PATH   PROPER distribution root (contains setup.py).
#                       Default: autodetected from ~/proper_v3.3.4_python etc.
#   --skip-proper       Do not install PROPER.
#   --local-prysm[=PATH] Install prysm editable from a local checkout instead of
#                       the git build. Default PATH: ~/prysm
#   --intel-fft         Install the PyPI MKL runtime so PROPER's prop_use_ffti()
#                       works in an OpenBLAS env. See the note below.
#   --verify-only       Skip installs, just print the capability report.
#   -h | --help
#
set -euo pipefail

PROPER_DIR=""
SKIP_PROPER=0
LOCAL_PRYSM=""
INTEL_FFT=0
VERIFY_ONLY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --proper-dir)     PROPER_DIR="$2"; shift 2 ;;
    --proper-dir=*)   PROPER_DIR="${1#*=}"; shift ;;
    --skip-proper)    SKIP_PROPER=1; shift ;;
    --local-prysm)    LOCAL_PRYSM="$HOME/prysm"; shift ;;
    --local-prysm=*)  LOCAL_PRYSM="${1#*=}"; shift ;;
    --intel-fft)      INTEL_FFT=1; shift ;;
    --verify-only)    VERIFY_ONLY=1; shift ;;
    -h|--help)        sed -n '2,29p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# --------------------------------------------------------------- guard rails --
if [ -z "${CONDA_PREFIX:-}" ]; then
  echo "ERROR: no conda environment is active." >&2
  echo "       conda env create -f environment.yml && conda activate dragrace" >&2
  exit 1
fi
ENV_NAME="$(basename "$CONDA_PREFIX")"
if [ "$ENV_NAME" = "anaconda3" ] || [ "$ENV_NAME" = "miniconda3" ] || [ "$ENV_NAME" = "base" ]; then
  echo "ERROR: refusing to modify the base environment ($CONDA_PREFIX)." >&2
  echo "       Create and activate a dragrace env first." >&2
  exit 1
fi
echo "==> environment: $ENV_NAME  ($CONDA_PREFIX)"

# ------------------------------------------------------------------- PROPER --
# PROPER is distributed as a source zip (SourceForge / the PROPER manual), not
# through PyPI or conda-forge, so it is always a local-path install. It builds
# three small C extensions, which is why a C toolchain has to be present.
if [ "$VERIFY_ONLY" -eq 0 ] && [ "$SKIP_PROPER" -eq 0 ]; then
  if [ -z "$PROPER_DIR" ]; then
    for cand in "$HOME"/proper_v*_python "$HOME"/proper_v*/ "$HOME"/PROPER*; do
      if [ -f "$cand/setup.py" ]; then PROPER_DIR="$cand"; break; fi
    done
  fi

  if [ -z "$PROPER_DIR" ] || [ ! -f "$PROPER_DIR/setup.py" ]; then
    echo "!!  PROPER not found. Download the Python distribution and re-run with"
    echo "!!      bash scripts/setup_env.sh --proper-dir /path/to/proper_vX.Y.Z_python"
    echo "!!  Skipping -- the PROPER adapter will report itself unavailable."
  else
    echo "==> installing PROPER (PyPROPER3) from $PROPER_DIR"
    if ! command -v gcc >/dev/null 2>&1 && ! command -v cc >/dev/null 2>&1; then
      echo "!!  no C compiler on PATH; PROPER's extensions will fail to build."
      echo "!!  Either install system gcc or uncomment 'c-compiler' in environment.yml."
    fi
    # Two attempts: the modern isolated build first, then a non-isolated build
    # against the env's own numpy/setuptools, which older setup.py files need.
    pip install "$PROPER_DIR" \
      || pip install --no-build-isolation "$PROPER_DIR" \
      || { echo "!!  PROPER install failed; see output above." >&2; }
  fi
fi

# ------------------------------------------------------ Intel MKL FFT (opt) --
# conda-forge's `mkl` package participates in the BLAS mutex, so it cannot be
# installed alongside an OpenBLAS-linked NumPy. The PyPI MKL runtime sits
# outside that mutex, which is what makes the "MKL FFT + OpenBLAS BLAS"
# configuration reachable at all -- PROPER loads libmkl_rt by explicit path
# through ctypes, so it neither knows nor cares what NumPy is linked against.
# Opt-in because it puts a second ~700MB MKL copy on disk.
if [ "$VERIFY_ONLY" -eq 0 ] && [ "$INTEL_FFT" -eq 1 ]; then
  echo "==> installing PyPI MKL runtime for PROPER's prop_use_ffti()"
  pip install mkl || echo "!!  MKL runtime install failed." >&2
fi

# -------------------------------------------------------------- local prysm --
# The gradient board needs prysm's adjoint API: *_adjoint in the
# prysm.propagation subpackage (focus_dft_adjoint, focus_adjoint,
# angular_spectrum_adjoint, to_fpm_and_back_adjoint). The DFT variants take a
# precomputed `executor`, which is exactly why the harness separates build()
# from propagate(). PyPI tops out at 0.21.1, so environment.yml installs prysm
# from git; use this flag to point at a working checkout instead.
if [ "$VERIFY_ONLY" -eq 0 ] && [ -n "$LOCAL_PRYSM" ]; then
  if [ -f "$LOCAL_PRYSM/pyproject.toml" ] || [ -f "$LOCAL_PRYSM/setup.py" ]; then
    echo "==> installing prysm (editable) from $LOCAL_PRYSM"
    pip install -e "$LOCAL_PRYSM"
  else
    echo "!!  no prysm checkout at $LOCAL_PRYSM -- keeping the PyPI build." >&2
  fi
fi

# ------------------------------------------------- activation-time variables --
# These must be set before the relevant library is first imported, so they go in
# activate.d rather than being exported per-run.
#
# Deliberately NOT set here: OMP_NUM_THREADS / MKL_NUM_THREADS and friends.
# Thread count is a benchmark variable owned by the config (configs/*.yml) and
# is applied per-run by the runner. Baking it into the env would make every
# result silently depend on which shell you happened to be in.
if [ "$VERIFY_ONLY" -eq 0 ]; then
  ACTIVATE_D="$CONDA_PREFIX/etc/conda/activate.d"
  mkdir -p "$ACTIVATE_D"
  HOOK="$ACTIVATE_D/dragrace.sh"
  : > "$HOOK"
  echo "#!/bin/sh" >> "$HOOK"
  echo "# written by Diffraction-Dragrace scripts/setup_env.sh" >> "$HOOK"

  # PROPER's prop_use_ffti() builds the path as exactly
  #     os.path.join(MKL_DIR, 'libmkl_rt.so')
  # (proper/prop_use_ffti.py:53, proper/prop_ffti.py:55) -- the UNVERSIONED
  # soname. Both conda-forge and PyPI ship only libmkl_rt.so.2, so PROPER's
  # Intel FFT path fails to load even when MKL is correctly installed. Provide
  # the symlink it expects.
  MKL_LIB_DIR=""
  for d in "$CONDA_PREFIX/lib"; do
    if [ -f "$d/libmkl_rt.so" ] || [ -f "$d/libmkl_rt.so.2" ]; then MKL_LIB_DIR="$d"; fi
  done
  if [ -n "$MKL_LIB_DIR" ]; then
    if [ ! -e "$MKL_LIB_DIR/libmkl_rt.so" ] && [ -f "$MKL_LIB_DIR/libmkl_rt.so.2" ]; then
      ln -s libmkl_rt.so.2 "$MKL_LIB_DIR/libmkl_rt.so"
      echo "==> symlinked libmkl_rt.so -> libmkl_rt.so.2 (PROPER wants the unversioned name)"
    fi
    echo "export MKL_DIR=\"\$CONDA_PREFIX/lib\"" >> "$HOOK"
    echo "==> MKL runtime found; MKL_DIR pinned for PROPER's prop_use_ffti()"
  else
    echo "==> no MKL runtime in this env; PROPER's prop_use_ffti() unavailable."
    echo "    (expected in the OpenBLAS env -- use envs/cpu-mkl.yml, or --intel-fft)"
  fi

  if python -c "import jax" >/dev/null 2>&1; then
    echo "export XLA_PYTHON_CLIENT_PREALLOCATE=false" >> "$HOOK"
    echo "export JAX_ENABLE_X64=1" >> "$HOOK"
    echo "==> JAX found; preallocation disabled and x64 enabled by default"
  fi

  if python -c "import cupy" >/dev/null 2>&1; then
    echo "export CUPY_CACHE_DIR=\"\$CONDA_PREFIX/var/cupy_kernel_cache\"" >> "$HOOK"
    mkdir -p "$CONDA_PREFIX/var/cupy_kernel_cache"
    echo "==> CuPy found; NVRTC kernel cache pinned inside the env"
  fi
  echo "    wrote $HOOK  (takes effect on next 'conda activate $ENV_NAME')"
fi

# ------------------------------------------------------------ verification ---
echo
echo "==> capability report"
python - <<'PY'
import importlib, sys

ROWS = [
    ("numpy",   "numpy",       "core"),
    ("scipy",   "scipy",       "core"),
    ("proper",  "proper",      "propagator"),
    ("poppy",   "poppy",       "propagator"),
    ("hcipy",   "hcipy",       "propagator"),
    ("prysm",   "prysm",       "propagator"),
    ("lentil",  "lentil",      "propagator"),
    ("dLux",    "dLux",        "propagator"),
    ("jax",     "jax",         "backend"),
    ("cupy",    "cupy",        "backend"),
    ("mkl_fft", "mkl_fft",     "backend"),
    ("pyfftw",  "pyfftw",      "backend"),
    ("viztracer", "viztracer", "profiler"),
    ("memray",  "memray",      "profiler"),
]

print(f"  {'package':<12} {'kind':<11} {'version':<12} status")
print(f"  {'-'*12} {'-'*11} {'-'*12} {'-'*40}")
missing = []
for label, mod, kind in ROWS:
    try:
        m = importlib.import_module(mod)
        v = getattr(m, "__version__", "?")
        print(f"  {label:<12} {kind:<11} {v:<12} ok")
    except Exception as e:
        missing.append(label)
        print(f"  {label:<12} {kind:<11} {'-':<12} MISSING ({type(e).__name__})")

print()
# The authoritative answer to "which BLAS am I actually running?" -- library
# names on disk, not what the package metadata claims. Every benchmark result
# records this verbatim.
try:
    import threadpoolctl
    for p in threadpoolctl.threadpool_info():
        print(f"  BLAS/OMP: {p.get('internal_api'):<10} threads={p.get('num_threads')}  {p.get('filepath')}")
except Exception as e:
    print(f"  threadpoolctl unavailable: {e}")

# prysm adjoint API -- required by the gradient board.
try:
    from prysm.propagation import focus_dft_adjoint, focus_adjoint  # noqa: F401
    print("  prysm: adjoint API present (gradient board available)")
except Exception:
    print("  prysm: adjoint API NOT found -- gradient board unavailable.")
    print("         PyPI tops out at 0.21.1, which predates it. Install from git:")
    print("         pip install 'prysm @ git+https://github.com/brandondube/prysm.git'")
    print("         or from a local checkout: bash scripts/setup_env.sh --local-prysm")

try:
    import jax
    print(f"  jax devices: {jax.devices()}  x64={jax.config.jax_enable_x64}")
except Exception:
    pass

try:
    import cupy
    print(f"  cupy device: {cupy.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
except Exception:
    pass

if missing:
    print(f"\n  {len(missing)} package(s) unavailable: {', '.join(missing)}")
    print("  Adapters for these will report themselves unsupported rather than fail.")
PY

echo
echo "==> done. Next:  dragrace doctor"
