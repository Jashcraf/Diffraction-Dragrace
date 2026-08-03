import numpy as np
from scipy import fft
import tensorflow as tf
import jax.numpy as jnp
import torch
import scipy

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
    if fft._srcmodule is tf.signal:
        return fft.fft2d(x)
    elif fft._srcmodule is scipy.fft:
        return fft.fft2(x, workers=-1)
    else:
        return fft.fft2(x)
    
def expi(x):
    return np.exp(1j * x)

def sync(x):
    """Force a lazily-computed array to be fully materialised.

    JAX dispatches asynchronously: an op enqueues work and immediately returns a
    handle, so timing it without a barrier measures the enqueue rather than the
    computation. numpy, torch-on-CPU and tensorflow-on-CPU are all synchronous
    already, so this is a no-op for them.

    Returns x, so it can wrap an expression inline: sync(a @ b)
    """
    block_until_ready = getattr(x, "block_until_ready", None)
    if block_until_ready is not None:
        return block_until_ready()

    # On an accelerator, torch also needs an explicit barrier
    if isinstance(x, torch.Tensor):
        if x.device.type == "cuda":
            torch.cuda.synchronize()
        elif x.device.type == "mps":
            torch.mps.synchronize()

    return x
