# Phase-retrieval propagator benchmark — results

Machine: macOS-14.5-arm64-arm-64bit (arm64)

| Package | Mode | JIT | Dtype | N | Device | Median [ms] | Iters | Fwd evals | Final cost | Phase RMS err [rad] |
|---|---|---|---|---|---|---|---|---|---|---|
| HCIPy | nograd | - | float64 | 64 | cpu | 138.0 | 38 | 588 | 1.30e-13 | 0.0000 |
| HCIPy | nograd | - | float64 | 128 | cpu | 304.4 | 41 | 648 | 2.49e-13 | 0.0000 |
| HCIPy | nograd | - | float64 | 256 | cpu | 941.6 | 36 | 624 | 1.23e-13 | 0.0000 |
| HCIPy | nograd | - | float64 | 512 | cpu | 3960.8 | 36 | 624 | 2.27e-13 | 0.0000 |
| HCIPy | nograd | - | float64 | 1024 | cpu | 13555.9 | 35 | 600 | 3.01e-13 | 0.0000 |
| HCIPy | nograd | - | float64 | 2048 | cpu | 55963.0 | 37 | 600 | 6.44e-14 | 0.0000 |
| HCIPy | nograd | - | float64 | 4096 | cpu | 257736.4 | 38 | 612 | 1.16e-13 | 0.0000 |
| POPPY | nograd | - | float64 | 64 | cpu | 3815.1 | 36 | 552 | 5.44e-14 | 0.0000 |
| POPPY | nograd | - | float64 | 128 | cpu | 5643.4 | 40 | 636 | 6.89e-15 | 0.0000 |
| POPPY | nograd | - | float64 | 256 | cpu | 7194.4 | 35 | 624 | 1.42e-13 | 0.0000 |
| POPPY | nograd | - | float64 | 512 | cpu | 13814.9 | 34 | 576 | 8.21e-14 | 0.0000 |
| POPPY | nograd | - | float64 | 1024 | cpu | 29706.0 | 35 | 588 | 1.45e-13 | 0.0000 |
| POPPY | nograd | - | float64 | 2048 | cpu | 78506.3 | 35 | 576 | 6.31e-14 | 0.0000 |
| POPPY | nograd | - | float64 | 4096 | cpu | 287959.5 | 36 | 588 | 7.25e-14 | 0.0000 |
| dLux | backprop | yes | float64 | 64 | cpu | 22.3 | 36 | 45 | 5.19e-15 | 0.0000 |
| dLux | backprop | no | float64 | 64 | cpu | 307.1 | 36 | 45 | 5.19e-15 | 0.0000 |
| dLux | backprop | yes | float64 | 128 | cpu | 49.2 | 41 | 55 | 3.13e-14 | 0.0000 |
| dLux | backprop | no | float64 | 128 | cpu | 479.8 | 41 | 55 | 3.13e-14 | 0.0000 |
| dLux | backprop | yes | float64 | 256 | cpu | 111.1 | 36 | 52 | 4.26e-14 | 0.0000 |
| dLux | backprop | no | float64 | 256 | cpu | 532.5 | 36 | 52 | 4.26e-14 | 0.0000 |
| dLux | backprop | yes | float64 | 512 | cpu | 286.6 | 34 | 48 | 1.68e-13 | 0.0000 |
| dLux | backprop | no | float64 | 512 | cpu | 820.9 | 34 | 48 | 1.68e-13 | 0.0000 |
| dLux | backprop | yes | float64 | 1024 | cpu | 1152.6 | 35 | 49 | 6.16e-14 | 0.0000 |
| dLux | backprop | no | float64 | 1024 | cpu | 2517.1 | 35 | 49 | 6.16e-14 | 0.0000 |
| dLux | backprop | yes | float64 | 2048 | cpu | 7223.0 | 36 | 48 | 2.89e-14 | 0.0000 |
| dLux | backprop | no | float64 | 2048 | cpu | 11790.9 | 36 | 48 | 2.89e-14 | 0.0000 |
| dLux | backprop | yes | float64 | 4096 | cpu | 54269.5 | 36 | 49 | 7.16e-14 | 0.0000 |
| dLux | backprop | no | float64 | 4096 | cpu | 40091.2 | 36 | 49 | 7.16e-14 | 0.0000 |
| dLux | nograd | yes | float64 | 64 | cpu | 175.1 | 36 | 564 | 3.92e-14 | 0.0000 |
| dLux | nograd | no | float64 | 64 | cpu | 1635.4 | 36 | 552 | 3.23e-14 | 0.0000 |
| dLux | nograd | yes | float64 | 128 | cpu | 366.5 | 40 | 648 | 8.15e-13 | 0.0000 |
| dLux | nograd | no | float64 | 128 | cpu | 2349.9 | 40 | 660 | 6.54e-14 | 0.0000 |
| dLux | nograd | yes | float64 | 256 | cpu | 800.4 | 34 | 588 | 2.52e-14 | 0.0000 |
| dLux | nograd | no | float64 | 256 | cpu | 2864.8 | 34 | 588 | 3.44e-13 | 0.0000 |
| dLux | nograd | yes | float64 | 512 | cpu | 2319.5 | 34 | 576 | 2.43e-13 | 0.0000 |
| dLux | nograd | no | float64 | 512 | cpu | 4781.4 | 34 | 576 | 2.07e-13 | 0.0000 |
| dLux | nograd | yes | float64 | 1024 | cpu | 7747.7 | 35 | 588 | 3.19e-13 | 0.0000 |
| dLux | nograd | no | float64 | 1024 | cpu | 13316.1 | 34 | 600 | 2.75e-14 | 0.0000 |
| dLux | nograd | yes | float64 | 2048 | cpu | 28284.9 | 33 | 576 | 2.09e-13 | 0.0000 |
| dLux | nograd | no | float64 | 2048 | cpu | 47458.7 | 34 | 576 | 1.13e-13 | 0.0000 |
| dLux | nograd | yes | float64 | 4096 | cpu | 114170.3 | 36 | 588 | 6.76e-14 | 0.0000 |
| dLux | nograd | no | float64 | 4096 | cpu | 233054.2 | 36 | 588 | 5.35e-14 | 0.0000 |
| prysm | backprop | - | float64 | 64 | cpu | 11.5 | 36 | 47 | 1.71e-13 | 0.0000 |
| prysm | backprop | - | float64 | 128 | cpu | 32.0 | 40 | 55 | 3.36e-14 | 0.0000 |
| prysm | backprop | - | float64 | 256 | cpu | 114.9 | 37 | 52 | 2.22e-14 | 0.0000 |
| prysm | backprop | - | float64 | 512 | cpu | 492.1 | 37 | 54 | 2.49e-14 | 0.0000 |
| prysm | backprop | - | float64 | 1024 | cpu | 1993.0 | 35 | 52 | 4.21e-13 | 0.0000 |
| prysm | backprop | - | float64 | 2048 | cpu | 7698.1 | 36 | 50 | 6.97e-14 | 0.0000 |
| prysm | backprop | - | float64 | 4096 | cpu | 37013.3 | 39 | 53 | 3.46e-14 | 0.0000 |
| prysm | nograd | - | float64 | 64 | cpu | 85.7 | 36 | 564 | 2.05e-13 | 0.0000 |
| prysm | nograd | - | float64 | 128 | cpu | 260.7 | 40 | 636 | 9.38e-14 | 0.0000 |
| prysm | nograd | - | float64 | 256 | cpu | 848.4 | 38 | 624 | 6.69e-14 | 0.0000 |
| prysm | nograd | - | float64 | 512 | cpu | 3952.4 | 36 | 624 | 1.77e-13 | 0.0000 |
| prysm | nograd | - | float64 | 1024 | cpu | 16066.1 | 35 | 612 | 2.14e-13 | 0.0000 |
| prysm | nograd | - | float64 | 2048 | cpu | 57389.8 | 36 | 588 | 1.01e-13 | 0.0000 |
| prysm | nograd | - | float64 | 4096 | cpu | 248488.4 | 37 | 612 | 1.87e-13 | 0.0000 |
