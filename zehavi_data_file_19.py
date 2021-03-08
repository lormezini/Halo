#magnitude 20 data
import numpy as np

bin_edges = np.array((0.13159712, 0.20840302, 0.33003624, 0.52265998, 0.82770744, 1.31079410, 2.07583147,3.28737848, 5.20603787, 8.24451170, 13.05637315, 20.67664966))
wp_vals = [ 0.01676,  322.5, 231.1,  162.4, 114.6, 75.5,  50.6, 35.0, 24.2,  15.3, 9.20, 4.11]
cov_matrix = np.genfromtxt('sdss_wp_covar_19.0.dat')

def get_bins():
    return bin_edges

def get_wp():
    return wp_vals

def get_cov():
    return cov_matrix
