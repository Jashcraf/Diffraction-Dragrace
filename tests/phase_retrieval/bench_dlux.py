"""dLux phase-retrieval benchmark.

Two modes:

* ``--grad`` (backprop): jax ``value_and_grad`` supplies the reverse-mode
  gradient; the loss + gradient are ``jax.jit`` compiled once, then handed to
  scipy L-BFGS-B.
* no ``--grad`` (nograd): only the (jitted) forward loss is exposed and scipy
  estimates the gradient by finite differences.

Run inside the dLux env, e.g.::

    /opt/anaconda3/envs/dlux/bin/python bench_dlux.py --n 128 --grad
"""

import argparse
import os
import warnings

# jax setup must precede other jax imports.
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import dLux as dl

import common
import bench_util as bu


def resolve_device(device):
    """Place jax work on GPU if requested and available, else CPU."""
    if device == "gpu":
        try:
            gpus = jax.devices("gpu")
            if gpus:
                jax.config.update("jax_default_device", gpus[0])
                return "gpu"
        except Exception:
            pass
        warnings.warn("jax GPU device unavailable; using CPU.")
    jax.config.update("jax_default_device", jax.devices("cpu")[0])
    return "cpu"


def build_forward(problem, use_jit=True):
    """Return ``(intensity, loss, value_and_grad(loss))`` for dLux' MFT.

    ``loss`` and its gradient are jax.jit compiled when ``use_jit`` is True,
    otherwise they are returned eager (interpreted) so the JIT speedup can be
    measured directly.
    """
    amp = jnp.asarray(problem["amp"])
    basis = jnp.asarray(problem["basis"])
    wavelength = float(problem["wavelength"])
    n = int(problem["n"])
    diameter = float(problem["diameter"])
    spec = dl.CoordSpec(int(problem["focal_pixels"]),
                        float(problem["focal_pixel_scale_rad"]))

    def intensity(coeffs):
        phase = jnp.tensordot(coeffs, basis, axes=(0, 0))
        wf = dl.Wavefront(wavelength, n, diameter)
        wf = wf * amp
        wf = wf.add_phase(phase)
        wf = wf.propagate_MFT(spec, None)
        return wf.psf

    def loss(coeffs, target):
        I = intensity(coeffs)
        m = I / I.sum()
        return 0.5 * jnp.sum((m - target) ** 2)

    fg = jax.value_and_grad(loss)
    if use_jit:
        loss, fg = jax.jit(loss), jax.jit(fg)
    return intensity, loss, fg


def main():
    parser = argparse.ArgumentParser(description="dLux phase-retrieval benchmark")
    bu.add_common_args(parser)
    parser.add_argument("--grad", action="store_true",
                        help="use jax value_and_grad (backprop)")
    parser.add_argument("--no-jit", dest="jit", action="store_false",
                        help="run eager (no jax.jit) to measure the JIT speedup")
    args = parser.parse_args()

    # Double precision is essential for jax: without x64 it silently uses
    # float32 and the retrieval both stalls and is not comparable to the
    # numpy-based packages.
    assert jax.config.jax_enable_x64, "jax x64 (double precision) must be enabled"

    mem0 = bu.rss_peak_mb()   # post-import baseline, before building the problem
    device_used = resolve_device(args.device)
    problem = common.make_problem(args.n)
    intensity, loss_fn, fg_fn = build_forward(problem, use_jit=args.jit)

    # Self-consistent target (inverse crime), same convention as every package.
    psf_truth = np.asarray(intensity(jnp.asarray(problem["truth_coeffs"])))
    assert psf_truth.dtype == np.float64, f"expected float64 PSF, got {psf_truth.dtype}"
    target = jnp.asarray(bu.normalize(psf_truth))

    x0 = np.zeros(problem["n_modes"])

    if args.grad:
        # Warm up the JIT so compilation time is excluded from the timed runs.
        v, g = fg_fn(jnp.asarray(x0), target)
        v.block_until_ready(); g.block_until_ready()

        def objective(x):
            v, g = fg_fn(jnp.asarray(x), target)
            return float(v), np.asarray(g, dtype=np.float64)
        mode = "backprop"
        with_grad = True
    else:
        loss_fn(jnp.asarray(x0), target).block_until_ready()   # warm up JIT

        def objective(x):
            return float(loss_fn(jnp.asarray(x), target))
        mode = "nograd"
        with_grad = False

    times, res, _ = bu.run_lbfgsb(objective, x0, with_grad=with_grad,
                                  trials=args.trials)

    # Keep JIT and eager runs in separate files so they don't overwrite.
    out_path = args.out
    if out_path is None and not args.jit:
        out_path = os.path.join(bu.RESULTS_DIR,
                                f"dLux_{mode}_n{problem['n']}_nojit.json")
    bu.save_result("dLux", mode, problem, device_used, times, res, res.x,
                   out_path=out_path, mem_baseline_mb=mem0,
                   extra={"device_requested": args.device,
                          "jax_backend": jax.default_backend(),
                          "jit": bool(args.jit),
                          "x64": bool(jax.config.jax_enable_x64),
                          "intensity_dtype": str(psf_truth.dtype)})


if __name__ == "__main__":
    main()
