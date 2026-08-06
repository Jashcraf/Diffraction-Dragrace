"""Setup cost as a first-class metric.

For this particular set of codes, steady-state per-propagation time is arguably
the *less* interesting number. `import poppy` pulls astropy; `import dLux` pulls
JAX and then pays a compile on first call. A user computing one PSF cares about
import + build + first call; a user running an optimisation loop cares about
steady state. Both are measured, and the amortisation curve in report.py shows
where the crossover falls.

Import time is measured in a clean subprocess because it can only be paid once
per interpreter -- measuring it in-process after the module is already loaded
would return zero.
"""
from __future__ import annotations

import json
import subprocess
import sys

MODULES = {
    "numpy": "numpy",
    "scipy": "scipy",
    "proper": "proper",
    "poppy": "poppy",
    "hcipy": "hcipy",
    "prysm": "prysm",
    "lentil": "lentil",
    "dLux": "dLux",
    "jax": "jax",
    "cupy": "cupy",
}

_SCRIPT = """
import json, sys, time
mod = sys.argv[1]
t0 = time.perf_counter()
try:
    __import__(mod)
    ok, err = True, None
except Exception as exc:
    ok, err = False, f"{type(exc).__name__}: {exc}"
print(json.dumps({"module": mod, "seconds": time.perf_counter() - t0,
                  "ok": ok, "error": err}))
"""


def import_time(module: str, python: str | None = None, timeout: float = 120.0) -> dict:
    proc = subprocess.run([python or sys.executable, "-c", _SCRIPT, module],
                          capture_output=True, text=True, timeout=timeout)
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:                                  # noqa: BLE001
        return {"module": module, "seconds": None, "ok": False,
                "error": proc.stderr.strip()[-500:] or "no output"}


def import_times(modules: list[str] | None = None, python: str | None = None) -> dict[str, dict]:
    return {m: import_time(MODULES.get(m, m), python) for m in (modules or MODULES)}


def breakdown(import_s: float | None, build_s: float | None,
              first_call_s: float | None, steady_s: float | None) -> dict:
    """The four components, plus the amortised total at several loop counts."""
    setup = sum(x for x in (import_s, build_s, first_call_s) if x)
    return {
        "import_s": import_s,
        "build_s": build_s,
        "first_call_s": first_call_s,
        "steady_s": steady_s,
        "setup_total_s": setup,
        "total_at_k": {k: setup + k * (steady_s or 0.0)
                       for k in (1, 10, 100, 1000, 10000)},
    }
