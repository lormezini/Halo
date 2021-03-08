#magnitude 20 data w/o great wall
import numpy as np

bin_edges = np.array((0.13159712, 0.20840302, 0.33003624, 0.52265998, 0.82770744, 1.31079410, 2.07583147,3.28737848, 5.20603787, 8.24451170, 13.05637315, 20.67664966))
wp_vals = [ 0.00656, 364.07296, 256.99536, 173.99469, 123.60770, 82.48638, 55.37721, 39.38650, 26.77838, 16.75592, 10.31587, 4.48836]
cov_matrix = np.genfromtxt('sdss_wp_covar_20_noGW.dat').reshape((13,13))

def get_bins():
    return bin_edges

def get_wp():
    return wp_vals

def get_cov():
    return cov_matrix
