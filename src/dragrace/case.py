"""Case definitions: the physics a benchmark run must reproduce.

A Case is pure physics plus execution knobs. It never names a library and never
names a backend -- those are the adapter and the Config respectively. The point
is that every adapter is handed the identical problem, so that a timing
difference is attributable to the implementation rather than to the two codes
having quietly solved different problems.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ALGORITHM_CLASSES = {"fft", "matrix_dft", "fresnel_tf", "fresnel_ir", "angular_spectrum", "czt"}
KINDS = {"pupil_to_focus", "gradient"}
DTYPES = {"complex64", "complex128"}


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
class Case:
    id: str
    kind: str
    algorithm_class: str
    wavelength_m: float
    pupil: Pupil
    output: Output
    dtype: str = "complex128"
    execution: Execution = field(default_factory=Execution)
    accuracy: Accuracy = field(default_factory=Accuracy)
    parameters: Parameters | None = None
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
    def n_focus(self) -> int:
        return self.output.samples

    @property
    def q(self) -> float:
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
        """Focal sample spacing in units of lambda*F/D."""
        return 1.0 / self.q

    @property
    def dx_focus_m(self) -> float:
        lam_f_over_d = self.wavelength_m * self.output.focal_length_m / self.pupil.diameter_m
        return lam_f_over_d / self.q

    @property
    def padding_factor(self) -> float:
        """Zero-padding factor an FFT implementation needs to reach q."""
        return self.q

    @property
    def n_fft(self) -> int:
        """Padded array size an FFT implementation must use to reach q."""
        return int(round(self.q * self.n_across))

    @property
    def real_dtype(self) -> str:
        return "float32" if self.dtype == "complex64" else "float64"

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
        out = Output(**{k: float(v) for k, v in d.pop("output").items()})
        params = d.pop("parameters", None)
        case = cls(
            pupil=pupil,
            output=out,
            execution=Execution(**(d.pop("execution", None) or {})),
            accuracy=Accuracy(**(d.pop("accuracy", None) or {})),
            parameters=Parameters(**params) if params else None,
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
