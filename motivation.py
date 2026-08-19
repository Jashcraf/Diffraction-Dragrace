import numpy as np
import matplotlib.pyplot as plt

n_model_evals = np.logspace(0, 5, 100)
runtimes_ms = [1, 10, 100, 1000, 10_000]

plt.figure()
plt.title("Model Evaluation Runtime vs. Number of Evaluations")
for runtime in runtimes_ms:
    if runtime < 1000:
        plt.plot(n_model_evals, runtime * n_model_evals / 1e3, label=f"Runtime = {runtime} ms")
    else:
        plt.plot(n_model_evals, runtime * n_model_evals / 1e3, label=f"Runtime = {runtime / 1e3} s")

# References
plt.axhline(y=5 * 60, color="gray", linestyle="--", label="Making Coffee")
plt.axhline(y=45 * 60, color="gray", linestyle="dotted", label="Avg. Meeting")
plt.axhline(y=24 * 60 * 60, color="gray", linestyle="dashdot", label="1 Day")

plt.xlabel("Model Evaluations")
plt.ylabel("Total Runtime (s)")
plt.yscale("log")
plt.xscale("log")
plt.xlim(1, 10_0000)
plt.ylim(1e-3, 1e6)
plt.legend(loc="lower right")
plt.show()