# Conventions

Every adapter receives its pupil array from `dragrace.grid` — the same aperture
rule, the same normalisation, evaluated on the sample positions that adapter
declares. These are the conventions that array and the expected output obey.

## Coordinates

**Pupil.** Lengths in units of the pupil diameter D, so the aperture has radius
0.5 and sample spacing `dx = 1/N_D`. The 1-D coordinate is

```
x[i] = (i - N_p//2) · dx,    i = 0 … N_p-1
```

**Focal plane.** Coordinates in units of λF/D, spacing `du = 1/q`:

```
u[j] = (j - N_f//2) · du,    j = 0 … N_f-1
```

The formulas above are the **`pixel`** convention — the fftshift convention,
origin on sample `N//2`. For even N this leaves the grid asymmetric by one
sample, which is standard, and is consistent between the MFT and FFT paths so
the two agree to roundoff rather than to half a pixel. `N_f` is forced even for
this reason.

### Two conventions, declared not imposed

A code measured through its documented API does not always get a choice about
where its samples sit. So each adapter **declares** which convention its output
obeys, via `grid_centering`, and the harness builds the reference, the injected
pupil and the coordinate grids to match:

| convention | 1-D grid | on-axis PSF | codes |
|---|---|---|---|
| `pixel` (default) | `(i - N//2)·dx` | peaks in one sample | numpy_baseline, prysm, lentil |
| `interpixel` | `(i - N/2 + 0.5)·dx` | centred on the four-pixel cross | POPPY, dLux |
| *mixed* | per plane | — | HCIPy (pupil interpixel, focal pixel) |

`grid_centering` is a string when both planes agree, or a mapping
`{"pupil": ..., "focus": ...}` when they do not. HCIPy needs the second form:
`make_pupil_grid` places no sample at the origin while `make_focal_grid` does,
so the two planes genuinely disagree inside one library. Declared as a single
convention it scores `rel_l2 = 5.9e-3`; declared per plane, `2.3e-15`.

This is not a relaxation. Both are correct discretisations of the same
continuous problem; the aperture rule, the physics and the rasterisation cost
are identical, and only the sample positions move by half a step. What it buys
is that a library which fixes its convention internally can still be gated at
full strength: POPPY's `OpticalSystem` hard-codes
`MatrixFourierTransform(centering='ADJUSTABLE')` inside `_propagate_mft` with no
documented knob reaching it, and dLux's `AngularOpticalSystem` does the same.

The numbers, measured:

| POPPY `calc_psf` compared against | `rel_l2` | peak offset |
|---|---|---|
| an `interpixel` reference (its own) | **1.5e-15** | (0, 0) |
| a `pixel` reference | 0.28 | (-1, -1) |
| a `pixel` reference, shimmed with `source_offset_r/theta` | 2.3e-2 | (0, 0) |

That last row is why declaring beats shimming. POPPY's source offset is the
documented way to move a PSF, and it does land the peak correctly — but the
half-pixel offset lives in the *pupil* grid too, and what survives is a residual
phase ramp across the focal field that no constant scale factor absorbs. Gating
that would mean relaxing `max_rel_l2` from 1e-10 to 1e-1 for every adapter, to
accommodate a mismatch that was never physics.

**Declaring the wrong convention fails loudly**, which is what makes the scheme
safe: the penalty is 0.28 and a one-pixel peak offset, not a slight degradation
that could pass unnoticed. The exception to watch is a *one-plane* error, as in
the HCIPy case above: the peak does not move and only a residual phase ramp
survives, so it lands at 5.9e-3 — still four orders above the gate, but small
enough to be mistaken for a tolerance problem and "fixed" by loosening
`max_rel_l2`. It is a grid error; loosening the gate would hide it.

`validate.compare` reports `peak_offset_px` precisely to catch this class of
bug. A failure with a non-zero offset is almost always a centring mismatch
rather than a propagation error; one *without* an offset, at ~1e-3, is usually a
single plane out of step.

prysm's `fttools.fftrange(n) = arange(-(n//2), -(n//2)+n)` is identical to the
`pixel` form, so that adapter declares nothing and gets the default.

## The transform

```
E_f[v,u] = Σ_{y,x} E_p[y,x] · exp(-2πi(xu + yv)) · dx²
```

The `dx²` factor is what makes the discrete sum approximate the continuous
integral. With the pupil field being the unit-amplitude aperture mask, this
gives `E_f(0) = π/4` (the area of a unit-diameter disk) and matches the analytic
Airy field

```
E(r) = (π/4) · 2J₁(πr)/(πr),    r in λF/D
```

## Normalisation and phase sign

Adapters are **not** required to match this normalisation. Codes disagree
legitimately about whether the PSF sums to the pupil energy, peaks at 1, or
carries `dx²`; and `exp(+ikz)` versus `exp(-ikz)` is a convention, not an error.

`validate.compare` fits a single complex scale factor by least squares,
separately tries the conjugated field, and gates only on the residual after
that fit. Both the fitted scale (`scale_abs`, `scale_phase_rad`) and the
`conjugated` flag are recorded in every result rather than discarded.

What adapters **are** required to match: the output grid. A `(N_f, N_f)` complex
array on the focal coordinates above, in the case's dtype. Shape or dtype
mismatches fail immediately with a message naming the expected values.

## Aberrations

OPD is carried in **waves**, so the phasor is `exp(2πi · opd)` with no
wavelength factor. This keeps the harness free of the metres-versus-microns unit
slips that are otherwise a recurring source of cross-code disagreement — the
conversion happens once, inside each adapter, where the library's own unit
expectations are documented.

Coefficients are **Noll-indexed Zernikes in waves RMS**. The basis is normalised
*numerically* to unit RMS over the discrete aperture rather than relying on the
analytic Noll normalisation, which is only unit-RMS over the continuous unit
disk. This makes a coefficient mean the same thing at every grid size, which
matters because N is a swept axis.

## Apertures

`circular_aperture` is a hard-edged unit-transmission mask, deliberately not
antialiased. A grey edge is a better optical model but it is also a *choice*,
and each of these codes makes a different one; pinning the hard mask keeps the
comparison about propagation rather than rasterisation. Cases needing an
antialiased edge should add an explicit `aperture: circular_antialiased` rather
than letting adapters differ.

## Units at the adapter boundary

The harness works in SI (`wavelength_m`, `diameter_m`, `focal_length_m`) plus
the dimensionless grids above. Libraries differ:

| library | pupil spacing | focal spacing | wavelength | focal length |
|---|---|---|---|---|
| prysm | mm | µm | µm | mm |
| POPPY | — (λ/D units via `nlamD`) | — | m | m |
| HCIPy | m | m | m | m |
| PROPER | m | set by `beam_ratio` | µm | m |
| lentil | dimensionless `alpha` | — | — | — |

Getting a conversion wrong produces a plausible-looking PSF at the wrong scale.
The accuracy gate catches it as a large `rel_l2` with `scale_abs` far from 1 —
which is a much more legible failure than a silently mis-scaled result.
