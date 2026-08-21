"""Draw the phase-retrieval pupil and the PSF the retrieval is fitting.

The figure is built from the harness's own rasteriser and forward model --
`grid.aperture_mask`, `retrieval.retrieval_parameters`, `retrieval.reference_psf`
-- not from a separate drawing routine, so what it shows is literally the mask
every adapter is handed and literally the observed PSF every adapter minimises
against. A figure redrawn independently could disagree with the benchmark and
nobody would notice.

    python scripts/plot_retrieval_pupil.py [--n-pupil 512] [--out PATH]

The two retrieval cases share one optical system and differ only in
`retrieval.gradient`, so either yields this figure; the numeric case is the
default because it is the one with six adapters on its board.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dragrace.case import Case
from dragrace.grid import aperture_mask, focus_coords, pupil_coords
from dragrace.retrieval import reference_psf, retrieval_parameters

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CASE = REPO / "cases" / "phase_retrieval" / "pr_zernike11_numeric_scan.yaml"
DEFAULT_OUT = REPO / "docs" / "figures" / "pr_zernike11_pupil_and_psf.png"
#: Displayed dynamic range of the PSF, in decades below the peak.
DECADES = 6.0


def _scan_case(case: Case, n_pupil: int) -> Case:
    """The concrete Case at one scan point, by the runner's own expansion."""
    for sub in case.scan_cases():
        if sub.n_pupil == n_pupil:
            return sub
    sizes = [c.n_pupil for c in case.scan_cases()]
    raise SystemExit(f"--n-pupil must be one of {sizes}, got {n_pupil}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", type=Path, default=DEFAULT_CASE)
    p.add_argument("--n-pupil", type=int, default=512,
                   help="scan point to draw (the geometry is size-independent; "
                        "only its rasterisation changes)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    case = _scan_case(Case.from_yaml(args.case), args.n_pupil)
    noll, theta_true, _, basis = retrieval_parameters(case)
    amp = aperture_mask(case)
    opd = np.tensordot(theta_true, basis, axes=(0, 0))       # waves
    psf = reference_psf(case, theta_true)
    psf = psf / psf.max()

    # Outside the aperture the OPD is defined but meaningless -- the Zernikes
    # keep growing past rho = 1 and would set the colour scale from a region
    # that transmits nothing.
    opd_shown = np.where(amp > 0, opd, np.nan)
    lim = float(np.nanmax(np.abs(opd_shown)))

    x = pupil_coords(case)                                   # units of D
    u = focus_coords(case)                                   # units of lambda F/D
    pup_ext = [x[0], x[-1], x[0], x[-1]]
    psf_ext = [u[0], u[-1], u[0], u[-1]]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5), constrained_layout=True)

    axes[0].imshow(amp, cmap="gray", origin="lower", extent=pup_ext,
                   interpolation="nearest", vmin=0, vmax=1)
    ob = case.pupil.obstruction
    axes[0].set_title(f"aperture  ({case.n_pupil}$^2$)", fontsize=10)
    axes[0].set_xlabel("$x / D$")
    axes[0].set_ylabel("$y / D$")

    im1 = axes[1].imshow(opd_shown, cmap="RdBu_r", origin="lower", extent=pup_ext,
                         interpolation="nearest", vmin=-lim, vmax=lim)
    # RMS over the transmitting area, which is the number an optician quotes --
    # not the per-mode sigma the coefficients were drawn from.
    rms = float(np.sqrt(np.nanmean(opd_shown[np.isfinite(opd_shown)] ** 2)))
    axes[1].set_title(f"truth OPD, Noll {noll[0]}–{noll[-1]}  ({rms:.3f} waves RMS)",
                      fontsize=10)
    axes[1].set_xlabel("$x / D$")
    fig.colorbar(im1, ax=axes[1], label="waves")

    im2 = axes[2].imshow(np.log10(np.maximum(psf, 10.0 ** -DECADES)), cmap="inferno",
                         origin="lower", extent=psf_ext, interpolation="nearest",
                         vmin=-DECADES, vmax=0.0)
    axes[2].set_title(f"observed PSF  ({case.n_focus}$^2$ at "
                      f"{case.output.samples_per_lambda_f_d:g}/$\\lambda F/D$)", fontsize=10)
    axes[2].set_xlabel(r"$x\ /\ (\lambda F/D)$")
    axes[2].set_ylabel(r"$y\ /\ (\lambda F/D)$")
    fig.colorbar(im2, ax=axes[2], label="$\\log_{10}$ (I / I$_{peak}$)")

    fig.suptitle(
        f"{case.id.split('@')[0]} — {ob.secondary_ratio:g} secondary, "
        f"{ob.spider_width_ratio:g} D vanes at "
        f"{', '.join(f'{a:g}°' for a in ob.spider_angles_deg)} spanning the "
        f"{ob.spider_span}",
        fontsize=11)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
