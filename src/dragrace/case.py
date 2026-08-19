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

ALGORITHM_CLASSES = {"fft", "matrix_dft", "fresnel_tf", "fresnel_ir", "angular_spectrum", "czt"}
KINDS = {"pupil_to_focus", "plane_to_plane", "gradient"}
#: Algorithm classes that propagate between two parallel planes a
#: distance apart, rather than pupil -> focus through a lens.
FREE_SPACE_CLASSES = {"fresnel_tf", "fresnel_ir", "angular_spectrum"}
DTYPES = {"complex64", "complex128"}
SCAN_PARAMETERS = {"n_pupil", "n_focus"}


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
class Pupil:
    diameter_m: float
    samples_across_diameter: int
    array_samples: int
    aperture: str = "circular"
    aberration: Aberration = field(default_factory=Aberration)


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
class Parameters:
    """Gradient board only: the real parameter vector theta the loss is
    differentiated with respect to."""

    basis: str = "zernike_noll"
    count: int = 15
    first_noll: int = 4            # skip piston/tip/tilt by default
    amplitude_waves_rms: float = 0.05
    seed: int = 1234


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
    def n_focus(self) -> int:
        """Samples across the observation plane.

        For a free-space case the observation grid *is* the illumination grid,
        so this is N_p rather than a focal sampling derived from q.
        """
        return self.n_pupil if self.is_free_space else self.output.samples

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
        if self.pupil.aperture != "circular":
            p.append(f"aperture {self.pupil.aperture!r} unsupported (only 'circular')")
        if self.kind == "gradient" and self.parameters is None:
            p.append("kind=gradient requires a `parameters:` block")
        if self.is_free_space:
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
        for v in s.values:
            if not isinstance(v, int) or v < 8:
                p.append(f"scan value {v!r} must be an integer >= 8")
            elif v % 2:
                # Both grids are centred at index N//2; an odd N would put the
                # scan's points on a different centring convention from every
                # other case in the suite.
                p.append(f"scan value {v} must be even (grids are centred at N//2)")
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
        pupil = Pupil(
            diameter_m=float(pup["diameter_m"]),
            samples_across_diameter=int(pup["samples_across_diameter"]),
            array_samples=int(pup["array_samples"]),
            aperture=pup.get("aperture", "circular"),
            aberration=Aberration(ab.get("convention", "noll_rms_waves"), coeffs),
        )
        out = d.pop("output", None)
        out = Output(**{k: float(v) for k, v in out.items()}) if out else None
        prop = d.pop("propagation", None)
        prop = Propagation(**{k: float(v) for k, v in prop.items()}) if prop else None
        params = d.pop("parameters", None)
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
