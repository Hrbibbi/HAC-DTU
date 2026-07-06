#%%
import pstats

import numpy as np
from lightcurveinv import LightcurveInverter
import matplotlib.pyplot as plt
import cProfile
import pstats
#%%
coords = np.loadtxt('asteroiddata/coords.txt', delimiter=' ')
obs_dir = coords[:, 3:]
sun_dir = np.array([1, 0, 0])

cylinder_R = np.array([1.12, 1.42, 0.88, 1.475, 1.22, 0.925, 1.205, 1.24, 0.67, 3.95])


asteroid_number = 3
HAC_data = np.loadtxt(f'asteroiddata/Asteroid0{asteroid_number}_lightcurve_intensity_blender.txt', delimiter=',')

LC = LightcurveInverter(HACdata=HAC_data, obs_dirs=obs_dir, sun_dir=sun_dir, subdivisions=3)

#%%
profiler = cProfile.Profile()
profiler.enable()
LC.fit_radii(sh_degree=4, sh_order=4, max_nfev=500, verbose=2, ftol=1e-4, shape_penalty=None, cylinder_radius=cylinder_R[int(asteroid_number)-1],include_l00=False)
profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats("cumtime")  # cumulative time
stats.print_stats(40)        # top 40 functions
LC.write_to_file(f'radii_shapes/A{asteroid_number}_b_int_shape_radii')
