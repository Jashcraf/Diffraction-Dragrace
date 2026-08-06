"""PROPER (PyPROPER3) adapter.

STATUS: unverified -- written against the local PROPER 3.3.4 source, not yet run.
Check before publishing.

PROPER's unit of work is a prescription function executed against module-level
global state (prop_begin / prop_end bracket a run). There is essentially nothing
to hoist into build() beyond FFTW wisdom, so its cold and warm numbers will
nearly coincide. That is a property of the API, not a measurement artifact, and
the report should say so rather than presenting PROPER as uniquely slow to warm
up.

Sampling is controlled by beam_ratio = (beam diameter)/(grid diameter), which
makes samples per lambda*F/D equal to 1/beam_ratio -- so beam_ratio = 1/q.

FFT backends, both confirmed present in proper/ on this machine:
  prop_use_fftw()  -> pyFFTW, with prop_fftw_wisdom / prop_load_fftw_wisdom
  prop_use_ffti()  -> Intel MKL, loaded by ctypes from $MKL_DIR/libmkl_rt.so

The Intel path needs the UNVERSIONED soname; conda and PyPI ship only
libmkl_rt.so.2, so scripts/setup_env.sh creates the symlink. Without it
prop_use_ffti() fails to load even with MKL correctly installed.
"""
from __future__ import annotations

import os

import numpy as np

from dragrace.adapter import Adapter, Unsupported, register
from dragrace.case import Case
from dragrace.config import Config


def _prescription(wavelength, gridsize, PASSVALUE=None):
    """Pupil -> focus through a perfect lens. Module-level by necessity:
    PROPER executes prescriptions by name against global state."""
    import proper

    p = PASSVALUE or {}
    wfo = proper.prop_begin(p["diam_m"], wavelength, gridsize, p["beam_ratio"])
    proper.prop_circular_aperture(wfo, p["diam_m"] / 2.0)
    proper.prop_define_entrance(wfo)
    proper.prop_lens(wfo, p["efl_m"])
    proper.prop_propagate(wfo, p["efl_m"])
    return proper.prop_end(wfo, NOABS=True)


@register("proper")
class ProperAdapter(Adapter):
    status = "unverified"
    reviewed_by = ""
    requires = ("proper",)

    def versions(self) -> dict[str, str]:
        import proper
        return {"PyPROPER3": getattr(proper, "__version__", "3.3.4"),
                "numpy": np.__version__}

    def supports(self, case: Case, config: Config) -> bool | Unsupported:
        if case.algorithm_class not in ("fft", "fresnel_tf", "angular_spectrum"):
            return Unsupported(
                f"PROPER is FFT/Fresnel-based; no matrix-DFT path for "
                f"{case.algorithm_class}. Sampling is set by beam_ratio, so it "
                f"cannot hit an arbitrary focal grid the way an MFT can."
            )
        if config.is_gpu:
            return Unsupported("PROPER has no GPU backend")
        return True

    def configure(self, config: Config) -> bool | Unsupported:
        import proper

        self._fft_name = config.fft_backend
        if config.fft_backend == "pyfftw":
            try:
                proper.prop_use_fftw()
            except Exception as exc:                 # noqa: BLE001
                return Unsupported(f"prop_use_fftw() failed: {exc}")
        elif config.fft_backend == "mkl":
            mkl_dir = os.environ.get("MKL_DIR")
            if not mkl_dir:
                return Unsupported(
                    "prop_use_ffti() needs MKL_DIR; run scripts/setup_env.sh "
                    "(or use the dragrace-mkl environment)"
                )
            if not os.path.exists(os.path.join(mkl_dir, "libmkl_rt.so")):
                return Unsupported(
                    f"{mkl_dir}/libmkl_rt.so missing -- PROPER builds this path "
                    "with the unversioned soname while conda/PyPI ship only "
                    "libmkl_rt.so.2. scripts/setup_env.sh creates the symlink."
                )
            try:
                proper.prop_use_ffti(MKL_DIR=mkl_dir)
            except Exception as exc:                 # noqa: BLE001
                return Unsupported(f"prop_use_ffti() failed: {exc}")
        return True

    def resolve_backend(self) -> dict:
        from dragrace.backend import detect_blas
        return {"array_module": "numpy", "device": "cpu",
                "fft_backend": self._fft_name, "blas": detect_blas()}

    def build(self, case: Case, config: Config):
        """Almost empty by necessity -- see the module docstring."""
        n = case.n_fft
        return {
            "case": case,
            "gridsize": n,
            "passvalue": {
                "diam_m": case.pupil.diameter_m,
                "efl_m": case.output.focal_length_m,
                "beam_ratio": 1.0 / case.q,          # samples per lambda F/D = 1/beam_ratio
            },
            "crop": (n // 2 - case.n_focus // 2, case.n_focus),
        }

    def propagate(self, state):
        # The prescription is called directly rather than through prop_run().
        # prop_run takes a prescription by NAME and imports it from a file on
        # sys.path, which cannot reach a function defined inside an installed
        # package. Calling it directly exercises the identical propagation code
        # path, minus PROPER's file-based prescription lookup -- and that lookup
        # is per-call overhead a real user does pay, so it is noted in the
        # report rather than quietly excluded. See docs/methodology.md.
        out = _prescription(state["case"].wavelength_m * 1e6, state["gridsize"],
                            state["passvalue"])
        if isinstance(out, tuple):
            out = out[0]
        c, npix = state["crop"]
        return np.asarray(out)[c:c + npix, c:c + npix]
