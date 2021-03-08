#magnitude 20 data
import numpy as np

bin_edges = np.array((0.13159712, 0.20840302, 0.33003624, 0.52265998, 0.82770744, 1.31079410, 2.07583147,3.28737848, 5.20603787, 8.24451170, 13.05637315, 20.67664966))
wp_vals = [ 0.00656, 366.1, 264.3, 184.0, 128.6, 84.7,  59.4, 42.9, 30.9,  21.9, 14.6,  8.24]
cov_matrix = np.genfromtxt('sdss_wp_covar_20.0.dat')

def get_bins():
    return bin_edges

def get_wp():
    return wp_vals

def get_cov():
    return cov_matrix
