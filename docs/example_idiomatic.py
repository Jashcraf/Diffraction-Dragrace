import poppy
from time import perf_counter

# One-time optical system setup
osys = poppy.OpticalSystem()
osys.add_pupil(poppy.CircularAperture(radius=0.5))
osys.add_detector(pixelscale=0.01, fov_arcsec=5.0)

# The profiled part
start_time = perf_counter()
osys.calc_psf(2e-6) # pass wavelength in m
end_time = perf_counter() - start_time



