"""Case definitions: the physics a benchmark run must reproduce.

A Case is pure physics plus execution knobs. It never names a library and never
names a backend -- those are the adapter and the Config respectively. The point
is that every adapter is handed the identical problem, so that a timing
difference is attributable to the implementation rather than to the two codes
having quietly solved different problems.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

ALGORITHM_CLASSES = {"fft", "matrix_dft", "fresnel_tf", "fresnel_ir", "angular_spectrum", "czt",
                     "segmented_aperture"}
KINDS = {"pupil_to_focus", "plane_to_plane", "gradient", "aperture", "phase_retrieval"}
#: Algorithm classes that propagate between two parallel planes a
#: distance apart, rather than pupil -> focus through a lens.
FREE_SPACE_CLASSES = {"fresnel_tf", "fresnel_ir", "angular_spectrum"}
DTYPES = {"complex64", "complex128"}
SCAN_PARAMETERS = {"n_pupil", "n_focus", "n_zernike"}
#: Scan axes that move a GRID. They share the constraints that come with one --
#: even, at least 8 samples, and (for n_pupil) divisible by the padding ratio --
#: which `n_zernike` does not: a Zernike count is a parameter count, and 3 and
#: 15 are both perfectly good ones.
GRID_SCAN_PARAMETERS = {"n_pupil", "n_focus"}


@dataclass(frozen=True)
class Aberration:
    convention: str = "noll_rms_waves"
    coefficients: dict[int, float] = field(default_factory=dict)

    @property
    def noll_indices(self) -> list[int]:
        return sorted(self.coefficients)

    @property
    def values(self) -> list[float]:
        return [self.coefficients[j] for j in self.noll_indices]


@dataclass(frozen=True)
class Obstruction:
    """A secondary obscuration and its spider vanes, for `aperture: obstructed`.

    Everything is a fraction of the pupil diameter, so the geometry is scale-free
    and a case can move D without redrawing the telescope.

    SPIDER SPAN is the one genuinely ambiguous parameter, so it is named rather
    than assumed. A vane "at +30 degrees" can mean a bar lying along that
    diameter -- which reaches the pupil edge on BOTH sides and so puts arms at
    +30 and +210 -- or a single half-ray running outward from the obscuration.
    `diameter` is the default because a real strut braces its secondary on both
    sides; `radius` is the reading under which "one vane at +30 and another at
    -30" draws two arms rather than four. The difference is visible in the PSF
    -- four diffraction spikes against two -- so it belongs in the case file
    where a reader can see it, not buried in an adapter.

    THE SHIPPED PHASE-RETRIEVAL CASES OVERRIDE THIS TO `radius`, and not for
    aesthetics: a diameter-spanning bar is centro-symmetric, as are the circle
    and the annulus, and a centro-symmetric pupil makes OPD phi(x) and -phi(-x)
    produce the identical PSF. Single-image phase retrieval then has two equally
    good answers and finds the wrong one. Half-ray vanes break that symmetry.
    See dragrace.retrieval and docs/phase_retrieval_board.md.
    """

    secondary_ratio: float = 0.30          # obscuration diameter / D
    spider_width_ratio: float = 0.01       # vane width / D
    spider_angles_deg: tuple[float, ...] = (30.0, -30.0)
    spider_span: str = "diameter"          # diameter | radius

    @property
    def spider_count(self) -> int:
        return len(self.spider_angles_deg)


@dataclass(frozen=True)
class Pupil:
    diameter_m: float
    samples_across_diameter: int
    array_samples: int
    aperture: str = "circular"
    aberration: Aberration = field(default_factory=Aberration)
    #: Only for `aperture: obstructed`. None everywhere else.
    obstruction: Obstruction | None = None


@dataclass(frozen=True)
class Output:
    focal_length_m: float
    samples_per_lambda_f_d: float
    extent_lambda_f_d: float

    @property
    def samples(self) -> int:
        """Focal-plane samples per side, N_f = round(q * W).

        Forced even so that the array centre falls on index N_f // 2 under the
        fftshift convention shared by every adapter (see docs/conventions.md).
        """
        n = int(round(self.samples_per_lambda_f_d * self.extent_lambda_f_d))
        return n + (n % 2)


@dataclass(frozen=True)
class Propagation:
    """Plane-to-plane geometry: how far, and nothing else.

    A free-space case has no lens and no focal plane, so it carries a distance
    instead of an Output block. The observation grid is the illumination grid --
    same N, same dx -- which is what the transfer-function methods produce
    natively and is the whole reason this case is a fairer cross-code comparison
    than pupil-to-focus: nobody has to hit somebody else's output sampling.
    """

    distance_m: float


@dataclass(frozen=True)
class Segmented:
    """A segmented pupil to be *drawn*, for kind=aperture.

    Every other case hands the adapters a mask the harness rasterised; here the
    rasterisation is the measurement, so the case carries a specification
    instead. Defaults are the E-ELT Construction Proposal values -- see
    dragrace.apertures, which owns the layout and the trap in it (the segment
    size is vertex-to-vertex, not flat-to-flat).
    """

    layout: str = "elt"
    rings: int = 17
    segment_vertex_to_vertex_m: float = 1.45
    segment_gap_m: float = 0.004
    central_obscuration_flat_to_flat_m: float = 9.4136
    spider_count: int = 6
    spider_width_m: float = 0.4
    spider_angle_offset_deg: float = 30.0
    n_segments: int = 798

    @property
    def segment_flat_to_flat_m(self) -> float:
        """Centre-to-centre spacing is this plus the gap, NOT vertex-to-vertex
        plus the gap. Reading it the other way builds a pupil 15% too large."""
        return self.segment_vertex_to_vertex_m * math.sqrt(3.0) / 2.0

    @property
    def segment_spacing_m(self) -> float:
        return self.segment_flat_to_flat_m + self.segment_gap_m

    def to_spec(self) -> dict:
        """The dict dragrace.apertures speaks."""
        return {
            "outer_diameter_m": None,          # filled by Case.segmented_spec
            "segment_vertex_to_vertex_m": self.segment_vertex_to_vertex_m,
            "segment_gap_m": self.segment_gap_m,
            "central_obscuration_flat_to_flat_m": self.central_obscuration_flat_to_flat_m,
            "spider_count": self.spider_count,
            "spider_width_m": self.spider_width_m,
            "spider_angle_offset_deg": self.spider_angle_offset_deg,
            "rings": self.rings,
            "n_segments": self.n_segments,
        }


@dataclass(frozen=True)
class Parameters:
    """Gradient board only: the real parameter vector theta the loss is
    differentiated with respect to."""

    basis: str = "zernike_noll"
    count: int = 15
    first_noll: int = 4            # skip piston/tip/tilt by default
    amplitude_waves_rms: float = 0.05
    seed: int = 1234


@dataclass(frozen=True)
class Retrieval:
    """kind=phase_retrieval: the inverse problem, and how it is to be solved.

    Following Jurling & Fienup (2014), JOSA A 31(7) 1348: estimate a Zernike
    coefficient vector by minimising the squared difference between a modelled
    PSF and an observed one, with a quasi-Newton optimiser rather than a
    Gerchberg-Saxton iteration.

    Everything an optimiser could differ on is pinned here rather than left to
    each adapter, because the timed quantity is "how long does the retrieval
    take" and that is meaningless unless every code is solving the same problem
    from the same starting point under the same stopping rule. Pinned: the truth
    coefficients, the starting guess, the convergence tests, the iteration cap
    and the L-BFGS history length.

    `gradient` splits the board in two, and the split is the point:

      numerical   the optimiser is handed a scalar loss only, and forms its own
                  gradient by two-point finite differences at scipy's default
                  absolute step of 1e-8 -- P+1 = 12 forward models per gradient
                  at P=11.
      analytic    the forward model returns (loss, dloss/dtheta), whether from a
                  hand-written adjoint or from reverse-mode AD -- ~2-3 forward
                  models per gradient, independent of P.

    That ratio is the entire argument of the Jurling & Fienup paper, and a code
    appears on whichever board matches the gradient it can actually supply.
    """

    basis: str = "zernike_noll"
    count: int = 11
    first_noll: int = 1                    # Noll 1..11 == piston .. primary spherical
    truth_amplitude_waves_rms: float = 0.05
    #: What `truth_amplitude_waves_rms` is the RMS *of*, and it only becomes a
    #: question once P is a swept axis:
    #:
    #:   per_mode   the standard deviation each coefficient is drawn from. The
    #:              wavefront's total RMS is then that times sqrt(P) -- fine at
    #:              a fixed P, and the convention the n_pupil boards use.
    #:   total_rms  the RMS of the whole WAVEFRONT. The per-coefficient sigma
    #:              is this over sqrt(P), so adding modes subdivides a fixed
    #:              amount of aberration instead of piling on more.
    #:
    #: A P-scan needs the second, and the reason is measured rather than
    #: aesthetic. Under `per_mode` at 0.05 waves the truth grows from 0.11 waves
    #: RMS at P=3 to 1.14 at P=496, and the retrieval stops being the same
    #: problem: the reference implementation recovers the truth to 1.9e-5 at
    #: P=55 and then falls into a local minimum at P=120 (loss stalls at 3.8e-5,
    #: coefficient error 8.4e-2). A runtime scan whose largest points are not
    #: solving the problem at all measures nothing. See
    #: docs/phase_retrieval_board.md.
    truth_amplitude_convention: str = "per_mode"       # per_mode | total_rms
    seed: int = 20260819
    initial: str = "zeros"                 # zeros | truth_perturbed
    #: Standard deviation of the starting guess's offset from the truth, as a
    #: fraction of truth_amplitude_waves_rms. Only read when initial is
    #: 'truth_perturbed'. Declared rather than hardcoded because it sets how
    #: much optimising there is to do, and therefore how much of each point's
    #: runtime is forward models rather than L-BFGS bookkeeping.
    initial_perturbation: float = 0.25
    optimizer: str = "lbfgsb"
    gradient: str = "numerical"            # numerical | analytic
    max_iterations: int = 100
    #: scipy's own test: (f_k - f_k+1) / max(|f_k|, |f_k+1|, 1) <= ftol.
    ftol: float = 1e-14
    #: scipy's own test: max |proj g_i| <= gtol.
    gtol: float = 1e-10
    history_size: int = 10                 # scipy maxcor / optax memory_size
    #: Tolerance for the untimed forward-model check: each code's own PSF at the
    #: truth coefficients, against the harness reference, after fitting one
    #: overall scale. Separate from accuracy.max_rel_l2, which gates the
    #: recovered coefficients, because they are different questions -- "did this
    #: code model the right telescope" and "did the optimiser find the right
    #: answer" -- and a single number could not serve both. Generous enough to
    #: admit a code that reaches the focal plane by a near-field Fresnel
    #: propagation rather than an exact Fourier transform, which is a modelling
    #: choice and not an error: PROPER lands at ~7.5e-5 on this suite's plain
    #: MFT case for exactly that reason.
    max_forward_rel_l2: float = 1.0e-3
    #: How the PSF is scaled before the residual is squared. Each of these codes
    #: normalises its PSF differently -- POPPY divides by the entrance flux,
    #: prysm carries dx^2, HCIPy does neither -- so an unnormalised MSE would
    #: differ between them by orders of magnitude and a shared ftol/gtol would
    #: silently mean a different convergence for each. Dividing by the peak of
    #: that code's OWN observed PSF makes the loss dimensionless, identical
    #: across codes to model accuracy, and exactly zero at the truth.
    loss_normalisation: str = "observed_peak"

    @property
    def noll_indices(self) -> list[int]:
        return list(range(self.first_noll, self.first_noll + self.count))

    @property
    def per_mode_sigma(self) -> float:
        """Standard deviation ONE truth coefficient is drawn from, in waves RMS.

        The single place `truth_amplitude_convention` is interpreted, so an
        adapter can never read the raw field and get a different wavefront from
        the reference.
        """
        if self.truth_amplitude_convention == "total_rms":
            return self.truth_amplitude_waves_rms / math.sqrt(self.count)
        return self.truth_amplitude_waves_rms


@dataclass(frozen=True)
class Execution:
    warmup: int = 3
    repeats: int = 25
    timeout_s: float = 300.0


@dataclass(frozen=True)
class Accuracy:
    reference: str = "analytic_airy"     # analytic_airy | internal_mft
    max_rel_l2: float = 1e-6


@dataclass(frozen=True)
class Scan:
    """An axis a single case is swept along inside one worker process.

    A scan case expands into one concrete Case per value, all measured back to
    back in the same interpreter and written to a single result.json. That is
    not merely convenient: the whole point of a size scan is the *shape* of the
    curve, and points measured in separate processes can differ by more than the
    effect being measured -- a different BLAS thread placement, a different heap
    state, a machine that got busier between runs. Measuring them together makes
    the comparison within a scan internally consistent, and the shared backend
    snapshot proves every point ran on the same machinery.

    Scaling is not free of physics. `n_pupil` grows the pupil array while the
    focal grid stays fixed, so the MFT floor grows as N_p^2 and the FFT floor as
    (q*N_p)^2 log(q*N_p) -- the two algorithm classes have genuinely different
    curves, which is why a scan case names one algorithm_class like any other.
    """

    parameter: str = "n_pupil"
    values: tuple[int, ...] = ()


@dataclass(frozen=True)
class Case:
    id: str
    kind: str
    algorithm_class: str
    wavelength_m: float
    pupil: Pupil
    output: Output | None = None            # pupil_to_focus / gradient
    propagation: Propagation | None = None  # plane_to_plane
    dtype: str = "complex128"
    execution: Execution = field(default_factory=Execution)
    accuracy: Accuracy = field(default_factory=Accuracy)
    parameters: Parameters | None = None
    segmented: Segmented | None = None
    retrieval: Retrieval | None = None
    scan: Scan | None = None
    loss: str = "psf_mse"
    basis_caching: str = "precomputed"   # precomputed | per_call
    notes: str = ""

    # ------------------------------------------------------------ derived --
    @property
    def n_pupil(self) -> int:
        return self.pupil.array_samples

    @property
    def n_across(self) -> int:
        return self.pupil.samples_across_diameter

    @property
    def is_free_space(self) -> bool:
        return self.kind == "plane_to_plane"

    @property
    def is_aperture(self) -> bool:
        """kind=aperture: the timed work is drawing the pupil, not propagating it."""
        return self.kind == "aperture"

    @property
    def is_retrieval(self) -> bool:
        """kind=phase_retrieval: the timed work is a whole nonlinear optimisation.

        One "propagation" here is one complete L-BFGS-B run -- hundreds of
        forward models -- so this board's times are seconds where every other
        board's are milliseconds, and its `repeats` are correspondingly few.
        """
        return self.kind == "phase_retrieval"

    def segmented_spec(self) -> dict:
        """Layout spec for dragrace.apertures, with the case's own diameter."""
        spec = self.segmented.to_spec()
        spec["outer_diameter_m"] = self.pupil.diameter_m
        return spec

    @property
    def n_focus(self) -> int:
        """Samples across the observation plane.

        For a free-space case the observation grid *is* the illumination grid,
        so this is N_p rather than a focal sampling derived from q. An aperture
        case has no second plane at all -- the drawn array is the whole output.
        """
        if self.is_free_space or self.is_aperture:
            return self.n_pupil
        return self.output.samples

    @property
    def n_zernike(self) -> int:
        """Free parameters the retrieval solves for. The `n_zernike` scan axis
        reads this back, so the worker labels a point with the count the case
        really carries rather than with the value it was asked for."""
        if self.retrieval is None:
            raise AttributeError(
                f"case {self.id!r} is not a phase_retrieval case and has no "
                f"Zernike parameter count")
        return self.retrieval.count

    @property
    def q(self) -> float:
        if self.is_free_space:
            raise AttributeError(
                f"case {self.id!r} is plane_to_plane: there is no focal sampling q. "
                f"Use dx_pupil_m and propagation.distance_m."
            )
        return self.output.samples_per_lambda_f_d

    @property
    def dx_pupil(self) -> float:
        """Pupil sample spacing in units of the pupil diameter D."""
        return 1.0 / self.n_across

    @property
    def dx_pupil_m(self) -> float:
        return self.pupil.diameter_m / self.n_across

    @property
    def du_focus(self) -> float:
        """Observation-plane sample spacing, in the same units as the pupil grid.

        Free space preserves sampling, so the two grids coincide and
        focus_coords() returns the illumination coordinates unchanged.
        """
        return self.dx_pupil if self.is_free_space else 1.0 / self.q

    @property
    def dx_focus_m(self) -> float:
        if self.is_free_space:
            return self.dx_pupil_m
        lam_f_over_d = self.wavelength_m * self.output.focal_length_m / self.pupil.diameter_m
        return lam_f_over_d / self.q

    @property
    def padding_factor(self) -> float:
        """Zero-padding factor an FFT implementation needs to reach q."""
        return 1.0 if self.is_free_space else self.q

    @property
    def n_fft(self) -> int:
        """Transform size an FFT implementation must use.

        A free-space transfer-function propagation transforms the array it was
        given: the guard band is already in `array_samples`, declared by the
        case rather than derived from a focal sampling.
        """
        return self.n_pupil if self.is_free_space else int(round(self.q * self.n_across))

    @property
    def fresnel_number(self) -> float:
        """a^2 / (lambda z) -- how deep into the near field this case sits."""
        if not self.is_free_space:
            raise AttributeError(f"case {self.id!r} is not plane_to_plane")
        a = self.pupil.diameter_m / 2.0
        return a * a / (self.wavelength_m * self.propagation.distance_m)

    @property
    def real_dtype(self) -> str:
        return "float32" if self.dtype == "complex64" else "float64"

    # ----------------------------------------------------------------- scan --
    @property
    def is_scan(self) -> bool:
        return self.scan is not None and len(self.scan.values) > 0

    @property
    def scan_points(self) -> int:
        return len(self.scan.values) if self.is_scan else 1

    @property
    def total_timeout_s(self) -> float:
        """Budget for the whole case, which for a scan is every point.

        `timeout_s` stays per-point so it means the same thing in a scan case as
        in a plain one; the runner multiplies rather than each scan case having
        to restate the arithmetic.
        """
        return self.execution.timeout_s * self.scan_points

    def scan_cases(self) -> list["Case"]:
        """One concrete, validated Case per scan value, in ascending order.

        Ascending is load-bearing: the cheap points are measured first, so a
        scan whose largest size exhausts memory still yields a usable curve
        rather than nothing.
        """
        if not self.is_scan:
            return [self]

        param = self.scan.parameter
        out: list["Case"] = []
        for v in sorted(self.scan.values):
            if param == "n_pupil":
                # v IS n_pupil -- the array size, matching the parameter's name
                # and what a reader sees on the x axis. The aperture follows,
                # preserving whatever padding ratio the base case declared: a
                # free-space case with a 4x guard band keeps 1.5 D of guard at
                # every size, rather than silently losing it as N grows.
                pad = self.pupil.array_samples // self.pupil.samples_across_diameter
                pupil = replace(self.pupil, samples_across_diameter=v // pad,
                                array_samples=v)
                sub = replace(self, pupil=pupil)
            elif param == "n_focus":
                # N_f = round(q * W), so the extent is what has to move.
                sub = replace(self, output=replace(self.output, extent_lambda_f_d=v / self.q))
            elif param == "n_zernike":
                # The only scan axis that leaves every grid alone. v is P, the
                # number of Zernike coefficients the retrieval solves for, so
                # the optical system, the sampling and the observed PSF are
                # identical at every point and the curve is the cost of the
                # PARAMETER COUNT and nothing else. The truth amplitude follows
                # from it through Retrieval.per_mode_sigma rather than being
                # restated here.
                sub = replace(self, retrieval=replace(self.retrieval, count=v))
            else:                                      # unreachable after validate()
                raise ValueError(f"unknown scan parameter {param!r}")

            sub = replace(sub, id=f"{self.id}@{param}={v}", scan=None)
            sub.validate()
            out.append(sub)
        return out

    # ---------------------------------------------------------- validation --
    def validate(self) -> None:
        p = []
        if self.kind not in KINDS:
            p.append(f"kind {self.kind!r} not in {sorted(KINDS)}")
        if self.algorithm_class not in ALGORITHM_CLASSES:
            p.append(f"algorithm_class {self.algorithm_class!r} not in {sorted(ALGORITHM_CLASSES)}")
        if self.dtype not in DTYPES:
            p.append(f"dtype {self.dtype!r} not in {sorted(DTYPES)}")
        if self.n_across > self.n_pupil:
            p.append(f"samples_across_diameter ({self.n_across}) > array_samples ({self.n_pupil})")
        if self.pupil.aperture not in ("circular", "segmented", "obstructed"):
            p.append(f"aperture {self.pupil.aperture!r} unsupported (circular; "
                     f"obstructed; or segmented for kind=aperture)")
        if (self.pupil.aperture == "segmented") != self.is_aperture:
            p.append("aperture: 'segmented' and kind: aperture go together; the "
                     "propagation boards inject the harness's own mask and must "
                     "not be handed a shape an adapter has to draw")
        if (self.pupil.aperture == "obstructed") != (self.pupil.obstruction is not None):
            p.append("aperture: 'obstructed' and an `obstruction:` block go "
                     "together -- one without the other either names a secondary "
                     "and spiders with no geometry, or carries geometry nothing "
                     "will draw")
        if self.kind == "gradient" and self.parameters is None:
            p.append("kind=gradient requires a `parameters:` block")
        p += self._retrieval_problems()
        if self.is_aperture:
            if self.segmented is None:
                p.append("kind=aperture requires a `segmented:` block")
            if self.algorithm_class != "segmented_aperture":
                p.append(
                    f"kind=aperture needs algorithm_class 'segmented_aperture', "
                    f"got {self.algorithm_class!r}"
                )
            if self.accuracy.reference != "internal_segmented_aperture":
                p.append(
                    f"kind=aperture must gate against 'internal_segmented_aperture'; "
                    f"{self.accuracy.reference!r} is a propagation reference and has "
                    f"no meaning for a drawn pupil"
                )
            if self.n_across != self.n_pupil:
                p.append(
                    f"kind=aperture draws the pupil across the whole array, so "
                    f"array_samples ({self.n_pupil}) must equal "
                    f"samples_across_diameter ({self.n_across}); a guard band would "
                    f"only change the sampling, not the physics being drawn"
                )
        elif self.is_free_space:
            if self.propagation is None:
                p.append("kind=plane_to_plane requires a `propagation:` block")
            if self.algorithm_class not in FREE_SPACE_CLASSES:
                p.append(
                    f"kind=plane_to_plane needs a free-space algorithm_class "
                    f"{sorted(FREE_SPACE_CLASSES)}, got {self.algorithm_class!r}"
                )
            if self.accuracy.reference == "analytic_airy":
                p.append(
                    "accuracy.reference=analytic_airy is a focal-plane closed form and "
                    "has no meaning for a free-space case; use internal_angular_spectrum"
                )
            if self.n_across >= self.n_pupil:
                p.append(
                    f"a free-space case needs a guard band: array_samples "
                    f"({self.n_pupil}) must exceed samples_across_diameter "
                    f"({self.n_across}), or the diffracted field wraps around the "
                    f"periodic FFT grid and contaminates the answer"
                )
        elif self.output is None:
            p.append(f"kind={self.kind} requires an `output:` block")
        p += self._scan_problems()
        if self.accuracy.reference == "analytic_airy" and self.pupil.aberration.coefficients:
            p.append(
                "accuracy.reference=analytic_airy is only valid for an unaberrated pupil; "
                "use reference=internal_mft for aberrated cases"
            )
        # complex64 cannot meet a float64-grade accuracy gate; catching this in
        # the case rather than as a mysterious run failure.
        if self.dtype == "complex64" and self.accuracy.max_rel_l2 < 1e-5:
            p.append(
                f"dtype=complex64 with max_rel_l2={self.accuracy.max_rel_l2:g} is unreachable; "
                "single precision bottoms out near 1e-6"
            )
        if p:
            raise ValueError(f"case {self.id!r} invalid:\n  - " + "\n  - ".join(p))

    def _retrieval_problems(self) -> list[str]:
        """Validation for kind=phase_retrieval, kept together so a malformed
        inverse problem fails at load rather than after an hour of optimising."""
        if not self.is_retrieval:
            return ["a `retrieval:` block is only meaningful for kind=phase_retrieval"] \
                if self.retrieval is not None else []
        if self.retrieval is None:
            return ["kind=phase_retrieval requires a `retrieval:` block"]

        r, p = self.retrieval, []
        if self.pupil.aperture != "obstructed":
            p.append(
                f"kind=phase_retrieval pins an obstructed pupil (circular aperture, "
                f"secondary obscuration, spider vanes), got "
                f"aperture={self.pupil.aperture!r}. The obscuration and the vanes are "
                f"not decoration: they break the pupil's continuous symmetries and "
                f"are what makes the retrieval well posed at all."
            )
        if self.accuracy.reference != "internal_phase_retrieval":
            p.append(
                f"kind=phase_retrieval must gate against 'internal_phase_retrieval'; "
                f"{self.accuracy.reference!r} compares focal fields and has no meaning "
                f"for a recovered coefficient vector"
            )
        if r.basis != "zernike_noll":
            p.append(f"unsupported retrieval basis {r.basis!r}")
        if r.gradient not in ("numerical", "analytic"):
            p.append(f"retrieval.gradient {r.gradient!r} not in ('numerical', 'analytic')")
        if r.optimizer != "lbfgsb":
            p.append(f"unsupported retrieval optimizer {r.optimizer!r}")
        if r.truth_amplitude_convention not in ("per_mode", "total_rms"):
            p.append(f"retrieval.truth_amplitude_convention "
                     f"{r.truth_amplitude_convention!r} not in ('per_mode', 'total_rms')")
        if r.initial not in ("zeros", "truth_perturbed"):
            p.append(f"retrieval.initial {r.initial!r} not in ('zeros', 'truth_perturbed')")
        if r.initial == "truth_perturbed" and not r.initial_perturbation > 0.0:
            p.append(f"retrieval.initial_perturbation must be > 0 for a perturbed "
                     f"start, got {r.initial_perturbation}")
        if r.loss_normalisation != "observed_peak":
            p.append(f"unsupported retrieval loss_normalisation {r.loss_normalisation!r}")
        if r.count < 1:
            p.append(f"retrieval.count must be >= 1, got {r.count}")
        if r.first_noll < 1:
            p.append(f"retrieval.first_noll must be >= 1 (Noll indices start at piston)")
        if r.max_iterations < 1:
            p.append(f"retrieval.max_iterations must be >= 1, got {r.max_iterations}")
        if r.history_size < 1:
            p.append(f"retrieval.history_size must be >= 1, got {r.history_size}")

        ob = self.pupil.obstruction
        if ob is not None:
            if not 0.0 <= ob.secondary_ratio < 1.0:
                p.append(f"secondary_ratio must be in [0, 1), got {ob.secondary_ratio}")
            if not 0.0 <= ob.spider_width_ratio < 1.0:
                p.append(f"spider_width_ratio must be in [0, 1), got "
                         f"{ob.spider_width_ratio}")
            if ob.spider_span not in ("diameter", "radius"):
                p.append(f"spider_span {ob.spider_span!r} not in ('diameter', 'radius')")
        # The truth aberration is applied through the retrieval block, so a
        # static one on top of it would be an aberration nothing is solving for
        # and would quietly bias every code's answer by the same unknown amount.
        if self.pupil.aberration.coefficients:
            p.append(
                "kind=phase_retrieval draws its wavefront error from the `retrieval:` "
                "block, so `pupil.aberration` must be empty -- a static OPD on top of "
                "the truth coefficients is an aberration no adapter is solving for"
            )
        if self.dtype != "complex128":
            p.append(
                f"kind=phase_retrieval needs complex128, got {self.dtype!r}: a "
                f"finite-difference gradient at single precision is dominated by "
                f"roundoff, and the numerical board would be measuring noise"
            )
        return p

    def _scan_problems(self) -> list[str]:
        """Scan validation, kept here so a malformed axis fails at load.

        A scan is expensive -- one worker measuring every point -- so an error
        that only surfaces at the last value has already cost the whole run.
        """
        if self.scan is None:
            return []
        s, p = self.scan, []
        if s.parameter not in SCAN_PARAMETERS:
            p.append(f"scan.parameter {s.parameter!r} not in {sorted(SCAN_PARAMETERS)}")
        if not s.values:
            p.append("scan.values is empty; omit the scan block instead")
        if len(set(s.values)) != len(s.values):
            p.append(f"scan.values has duplicates: {list(s.values)}")
        if s.parameter == "n_pupil" and self.pupil.samples_across_diameter:
            pad = self.pupil.array_samples // self.pupil.samples_across_diameter
            bad = [v for v in s.values if pad and v % pad]
            if bad:
                p.append(
                    f"scan values {bad} are not divisible by the padding ratio {pad} "
                    f"(array_samples / samples_across_diameter); a scan value is the "
                    f"array size and the aperture is derived from it"
                )
        if s.parameter in GRID_SCAN_PARAMETERS:
            for v in s.values:
                if not isinstance(v, int) or v < 8:
                    p.append(f"scan value {v!r} must be an integer >= 8")
                elif v % 2:
                    # Both grids are centred at index N//2; an odd N would put the
                    # scan's points on a different centring convention from every
                    # other case in the suite.
                    p.append(f"scan value {v} must be even (grids are centred at N//2)")
        elif s.parameter == "n_zernike":
            # A parameter count, not a grid: the rules above are about sample
            # centring and would reject P=3 and P=15 for no reason at all.
            if not self.is_retrieval:
                p.append(
                    f"scan.parameter 'n_zernike' sweeps retrieval.count and is only "
                    f"meaningful for kind=phase_retrieval, not {self.kind!r}"
                )
            for v in s.values:
                if not isinstance(v, int) or v < 1:
                    p.append(f"n_zernike scan value {v!r} must be an integer >= 1")
        if s.parameter == "n_pupil" and self.pupil.array_samples % self.pupil.samples_across_diameter:
            p.append(
                f"an n_pupil scan needs array_samples ({self.pupil.array_samples}) to be an "
                f"integer multiple of samples_across_diameter "
                f"({self.pupil.samples_across_diameter}); the padding ratio is what is held "
                f"fixed across the scan"
            )
        return p

    # ------------------------------------------------------------- loading --
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Case":
        d = dict(d)
        pup = dict(d.pop("pupil"))
        ab = pup.pop("aberration", None) or {}
        coeffs = {int(k): float(v) for k, v in (ab.get("coefficients") or {}).items()}
        obs = pup.pop("obstruction", None)
        if obs is not None:
            obs = dict(obs)
            obs["spider_angles_deg"] = tuple(
                float(a) for a in (obs.get("spider_angles_deg") or ()))
            obs = Obstruction(**obs)
        pupil = Pupil(
            diameter_m=float(pup["diameter_m"]),
            samples_across_diameter=int(pup["samples_across_diameter"]),
            array_samples=int(pup["array_samples"]),
            aperture=pup.get("aperture", "circular"),
            aberration=Aberration(ab.get("convention", "noll_rms_waves"), coeffs),
            obstruction=obs,
        )
        out = d.pop("output", None)
        out = Output(**{k: float(v) for k, v in out.items()}) if out else None
        prop = d.pop("propagation", None)
        prop = Propagation(**{k: float(v) for k, v in prop.items()}) if prop else None
        params = d.pop("parameters", None)
        seg = d.pop("segmented", None)
        retr = d.pop("retrieval", None)
        scan = d.pop("scan", None)
        if scan is not None:
            scan = dict(scan)
            scan["values"] = tuple(int(v) for v in (scan.get("values") or ()))
        case = cls(
            pupil=pupil,
            output=out,
            propagation=prop,
            execution=Execution(**(d.pop("execution", None) or {})),
            accuracy=Accuracy(**(d.pop("accuracy", None) or {})),
            parameters=Parameters(**params) if params else None,
            segmented=Segmented(**seg) if seg else None,
            retrieval=Retrieval(**retr) if retr else None,
            scan=Scan(**scan) if scan is not None else None,
            **d,
        )
        case.validate()
        return case

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Case":
        path = Path(path)
        case = cls.from_dict(yaml.safe_load(path.read_text()))
        if case.id != path.stem:
            raise ValueError(f"case id {case.id!r} does not match filename {path.stem!r}")
        return case

    def summary(self) -> str:
        if self.is_retrieval:
            scan = ""
            if self.is_scan:
                scan = " scan " + self.scan.parameter + "=[" + ",".join(
                    str(v) for v in sorted(self.scan.values)) + "]"
            ob = self.pupil.obstruction
            return (
                f"{self.id}: {self.kind}/{self.retrieval.gradient}-gradient{scan} "
                f"N_p={self.n_pupil} N_f={self.n_focus} "
                f"P={self.retrieval.count} (Noll {self.retrieval.first_noll}"
                f"-{self.retrieval.first_noll + self.retrieval.count - 1}) "
                f"eps={ob.secondary_ratio:g} "
                f"vanes={len(ob.spider_angles_deg)}@"
                f"{','.join(f'{a:+g}' for a in ob.spider_angles_deg)}deg"
            )
        if self.is_aperture:
            scan = ""
            if self.is_scan:
                scan = " scan " + self.scan.parameter + "=[" + ",".join(
                    str(v) for v in sorted(self.scan.values)) + "]"
            return (
                f"{self.id}: {self.kind}/{self.segmented.layout}{scan} "
                f"N={self.n_pupil} segments={self.segmented.n_segments} "
                f"D={self.pupil.diameter_m:g}m spiders={self.segmented.spider_count}"
            )
        if self.is_free_space:
            scan = ""
            if self.is_scan:
                scan = " scan " + self.scan.parameter + "=[" + ",".join(
                    str(v) for v in sorted(self.scan.values)) + "]"
            return (
                f"{self.id}: {self.kind}/{self.algorithm_class}{scan} "
                f"N={self.n_pupil} N_D={self.n_across} "
                f"D={self.pupil.diameter_m * 1e3:g}mm z={self.propagation.distance_m:g}m "
                f"N_F={self.fresnel_number:.0f} {self.dtype}"
            )
        if self.is_scan:
            vals = ",".join(str(v) for v in sorted(self.scan.values))
            return (
                f"{self.id}: {self.kind}/{self.algorithm_class} "
                f"scan {self.scan.parameter}=[{vals}] "
                f"N_f={self.n_focus} q={self.q:g} "
                f"W={self.output.extent_lambda_f_d:g} {self.dtype}"
            )
        return (
            f"{self.id}: {self.kind}/{self.algorithm_class} "
            f"N_p={self.n_pupil} N_D={self.n_across} N_f={self.n_focus} "
            f"q={self.q:g} W={self.output.extent_lambda_f_d:g} {self.dtype}"
        )


def load_cases(root: str | Path = "cases") -> dict[str, Case]:
    out: dict[str, Case] = {}
    for p in sorted(Path(root).rglob("*.yaml")):
        if p.stem.startswith("sweep_"):
            continue          # sweep specs are expanded by the runner, not cases
        c = Case.from_yaml(p)
        out[c.id] = c
    return out
