#magnitude -21.6 data
import numpy as np

bin_edges = np.array((0.1, 0.158489319, 0.251188643, 0.398107171,0.630957344, 1.000, 1.58489319, 2.51188643, 3.98107171, 6.30957344, 10.0, 15.8489319, 25.1188643))
wp_vals = [ 0.00029, 10029.85, 4744.70,  2860.37,  2732.31,  1560.89, 1025.62, 629.37, 363.88,  195.87, 128.80,  92.67, 67.65]
cov_matrix = np.genfromtxt("wpcovcmass_dr10v8_z0_480_55_mag21_6")[0:12,0:12]
def get_bins():
    return bin_edges

def get_wp():
    return wp_vals

def get_cov():
    return cov_matrix
