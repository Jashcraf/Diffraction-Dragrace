"""dragrace -- the Diffraction-Dragrace benchmark harness.

A run is the triple (case x config x adapter):

  case     the physics to reproduce      cases/*.yaml
  config   the machinery to use          configs/*.yaml
  adapter  the code under test           adapters/*/adapter.py

The harness itself depends on nothing a propagator depends on beyond NumPy, so
it can be installed into every per-adapter environment without perturbing the
thing being measured.

IMPORTANT: this module must not import NumPy at package-import time. Thread
counts are a config variable applied via environment variables, and OpenBLAS
and MKL read those at load time -- so if `import dragrace` pulled in NumPy, the
worker's `os.environ.update(config.full_env())` would run too late and every
run would silently use the machine default thread count. Only `case` and
`config` (yaml-only) are eager; everything else is lazy.
"""
from .case import Case, load_cases  # noqa: F401
from .config import Config, load_configs  # noqa: F401

__version__ = "0.1.0"

_LAZY = {
    "Adapter": "adapter", "Unsupported": "adapter", "available": "adapter",
    "discover": "adapter", "get": "adapter", "register": "adapter",
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib
        mod = importlib.import_module(f".{_LAZY[name]}", __package__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_LAZY))
