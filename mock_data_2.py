import numpy as np

f = np.load('mock_data_2.npz')            
wp_vals = f['stat1'][1:21].tolist()
bin_edges = np.logspace(-1, 1.2, 12)[0:21]
cov_matrix = f['cov1'][1:12,1:12]
ng_cov = f['cov1'][0,0]
ng = f['stat1'][0]

def get_wp():
    return wp_vals
def get_ng():
    return ng
def get_cov():
    return cov_matrix
def get_ng_cov():
    ng_err = ng_cov**2
    return ng_err
def get_bin_edges():
    return bin_edges
