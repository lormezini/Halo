#magnitude 21 data
import numpy as np

bin_edges = np.array((0.13159712, 0.20840302, 0.33003624, 0.52265998, 0.82770744, 1.31079410, 2.07583147,3.28737848, 5.20603787, 8.24451170, 13.05637315, 20.67664966, 34.75361614, 51.50000000))
wp_vals = [0.00116,  586.2, 402.9, 258.7, 163.2, 105.5, 68.9, 50.2, 35.5, 24.5, 15.3,  8.54, 4.11, 2.73]
cov_matrix = np.genfromtxt('sdss_wp_covar_21.0_full.txt')
def get_bins():
    return bin_edges

def get_wp():
    return wp_vals

def get_cov():
    return cov_matrix
get_cov()
