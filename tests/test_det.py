# Needs to be done at the top otherwise it doesn't work
import jax
jax.config.update("jax_enable_x64", True)
import numpy as tnp
import matplotlib.pyplot as plt
from time import perf_counter

# Just for array conversion
import jax.numpy as jnp
import torch
import tensorflow as tf

# All of our convenience functions
from mathops import (
    np,
    fft2,
    set_backend_to_jax,
    set_backend_to_tensorflow,
    set_backend_to_torch,
    set_backend_to_numpy,
)

# Define a circle
Ns = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
N_TRIALS = 100
methods = [
    set_backend_to_jax,
    set_backend_to_tensorflow,
    set_backend_to_torch,
    set_backend_to_numpy,
]

timing_np = []
timing_tf = []
timing_torch = []
timing_jax = []

err_np = []
err_tf = []
err_torch = []
err_jax = []

for N in Ns:

    # just create arrays of ones
    
    aperture = np.eye(N)
    
    # make sure they are all the same data type
    aperture_numpy = aperture.copy().astype(np.float64)
    aperture_jax = jnp.array(aperture, dtype=jnp.float64)
    aperture_torch = torch.tensor(aperture, dtype=torch.float64)
    aperture_tf = tf.constant(aperture, dtype=tf.float64)

    apertures = [
        aperture_jax,
        aperture_tf,
        aperture_torch,
        aperture_numpy,
    ]

    for i, (method, aperture) in enumerate(zip(methods, apertures)):
        method()
        timing = []

        for _ in range(N_TRIALS):
            t1 = perf_counter()
            dump = np.linalg.det(aperture)
            t2 = perf_counter()

            time = t2 - t1
            timing.append(time)

        if i == 0:
            timing_jax.append(tnp.mean(tnp.array(timing)))
            err_jax.append(tnp.std(tnp.array(timing)))
        elif i == 1:
            timing_tf.append(tnp.mean(tnp.array(timing)))
            err_tf.append(tnp.std(tnp.array(timing)))
        elif i == 2:
            timing_torch.append(tnp.mean(tnp.array(timing)))
            err_torch.append(tnp.std(tnp.array(timing)))
        elif i == 3:
            timing_np.append(tnp.mean(tnp.array(timing)))
            err_np.append(tnp.std(tnp.array(timing)))

timing = [timing_jax, timing_tf, timing_torch, timing_np]
err = [err_jax, err_tf, err_torch, err_np]

set_backend_to_numpy()
plt.figure()

for mean, std, method in zip(timing, err, methods):
    plt.errorbar(Ns, mean, yerr=std, fmt='o', label=method.__name__, capsize=5)

plt.legend()
plt.xlabel('N (for a square N x N matrix)')
plt.ylabel('Time (seconds)')
plt.title('linalg.det Performance Comparison')
plt.yscale("log")
plt.show()
