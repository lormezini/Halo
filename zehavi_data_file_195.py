#magnitude 20 data
import numpy as np

bin_edges = np.array((0.13159712, 0.20840302, 0.33003624, 0.52265998, 0.82770744, 1.31079410, 2.07583147,3.28737848, 5.20603787, 8.24451170, 13.05637315, 20.67664966))
wp_vals = [ 0.01004, 307.0, 228.5, 159.3,  110.4,72.9, 49.8, 34.6, 24.6, 16.7,10.7, 5.73]
cov_matrix = np.genfromtxt('sdss_wp_covar_19.5.dat')
print(cov_matrix)
def get_bins():
    return bin_edges

def get_wp():
    return wp_vals

def get_cov():
    return cov_matrix
