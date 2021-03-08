import numpy as np

f = np.load('mock_data.npz')
            
wp_vals = f['stat1'][1:21].tolist()
bin_edges = np.logspace(-1, 1.5, 21)[0:21]
cov_matrix = f['cov1'][1:21,1:21]
ng_cov = f['cov1'][0,0]
ng = f['stat1'][0]

def get_wp():
    return wp_vals
def get_ng():
    return ng
def get_cov():
    return cov_matrix
def get_ng_cov():
    return ng_cov
def get_bin_edges():
    return bin_edges
