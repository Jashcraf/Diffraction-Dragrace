# Conventions

Every adapter receives the identical pupil array from `dragrace.grid`. These are
the conventions that array and the expected output obey.

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

Both grids are centred at index `N//2` — the fftshift convention. For even N
this leaves the grid asymmetric by one sample, which is standard, and is
consistent between the MFT and FFT paths so the two agree to roundoff rather
than to half a pixel. `N_f` is forced even for this reason.

prysm's `fttools.fftrange(n) = arange(-(n//2), -(n//2)+n)` is identical to this,
so no re-centring shim is needed for that adapter. Adapters whose library centres
differently must shim in `build()`, not silently return a half-pixel-shifted
field — `validate.compare` reports the PSF peak offset in pixels precisely to
catch this, and a failure with a non-zero offset is almost always a centring
mismatch rather than a propagation error.

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
