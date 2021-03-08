#magnitude 20 data
import numpy as np

bin_edges = np.array((0.13159712, 0.20840302, 0.33003624, 0.52265998, 0.82770744, 1.31079410, 2.07583147,3.28737848, 5.20603787, 8.24451170, 13.05637315, 20.67664966))
wp_vals = [0.00318,  455.7, 296.9, 197.0, 134.1,  89.4, 61.1, 44.0, 31.2, 21.3, 13.7, 7.65]
cov_matrix = np.genfromtxt('sdss_wp_covar_20.5.dat')
def get_bins():
    return bin_edges

def get_wp():
    return wp_vals

def get_cov():
    return cov_matrix
print(cov_matrix)
