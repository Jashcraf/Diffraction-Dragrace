import numpy as np
from scipy import fft
import tensorflow as tf
import jax.numpy as jnp
import torch

class BackendShim:
    """A shim that allows a backend to be swapped at runtime.
    Taken from prysm.mathops with permission from Brandon Dube
    """

    def __init__(self, src):
        self._srcmodule = src

    def __getattr__(self, key):
        if key == "_srcmodule":
            return self._srcmodule

        return getattr(self._srcmodule, key)


_np = np
_fft = fft

# Set up the shims
np = BackendShim(np)
fft = BackendShim(fft)

def set_backend_to_numpy():
    import numpy as cp
    from scipy import fft as cpfft
    np._srcmodule = cp
    fft._srcmodule = cpfft

def set_backend_to_jax():
    try:
        import jax.numpy as jnp
        from jax.numpy import fft as jaxfft
        np._srcmodule = jnp
        fft._srcmodule = jaxfft
    except ImportError:
        print("JAX is not installed. Please install JAX to use the JAX backend.")

def set_backend_to_torch():
    try:
        import torch
        np._srcmodule = torch
        fft._srcmodule = torch.fft
    except ImportError:
        print("PyTorch is not installed. Please install PyTorch to use the PyTorch backend.")

def set_backend_to_tensorflow():
    try:
        import tensorflow as tf
        np._srcmodule = tf
        fft._srcmodule = tf.signal
    except ImportError:
        print("TensorFlow is not installed. Please install TensorFlow to use the TensorFlow backend.")

def fft2(x):
    if np._srcmodule is tf:
        return fft.fft2d(x)
    else:
        return fft.fft2(x)
