#analyze results of nearest neighbor averaging of chi2s

from halotools.sim_manager import CachedHaloCatalog, FakeSim
from halotools.empirical_models import PrebuiltHodModelFactory, Zheng07Cens, Zheng07Sats, TrivialPhaseSpace, NFWPhaseSpace, HodModelFactory
from halotools.mock_observables import return_xyz_formatted_array,wp
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool, cpu_count
import emcee
import corner
from numpy.linalg import inv
import scipy.optimize as op
from scipy.stats import chi2
import scipy.stats as stats
import random
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')
from scipy.special import gamma
from scipy.stats import chisquare, gaussian_kde,skewnorm,expon
from sklearn.neighbors import NearestNeighbors, KDTree
from tabcorr import TabCorr

fname = "zehavi_smdpl_mvir_m20_test.h5"
wp_out_file = None#'zehavi_smdpl_mvir_m19_wp.npy'
dname ='zehavi_data_file_20'
param = 'mvir'
threshold = 20
ext = 'mvir_m-20.npz'

files = [fname]
#files = [fname4,fname5]
s = []
log_prob_s = []
wps = []
for f in files: 
    reader = emcee.backends.HDFBackend(f, read_only=True)
    s.append(reader.get_chain(discard=1000, flat=False, thin=1))
    log_prob_s.append(reader.get_log_prob(discard=1000, flat=False, thin=1))
    wps.append(reader.get_blobs(discard=1000))
print(reader.iteration)


m,b = (0.3260532250619037, -1.6354239489541218)
if param == 'vmax_ml':
    s[0][:,:,0]=s[0][:,:,0]*(m)+(b)
    s[0][:,:,3]=s[0][:,:,3]*(m)+(b)
    s[0][:,:,4]=s[0][:,:,4]*(m)+(b)
    s[0][:,:,1]=s[0][:,:,1]*m #sigma
    s[0][:,:,2]=s[0][:,:,2]/m #alpha


if len(files)>1:
    print('greater')
    samples = s[0]
    log_prob_samples = log_prob_s[0]
    wp_samples = wps[0]
    for i in range(len(files)):
        if i+1 < len(s):
            samples = np.concatenate((samples,s[i+1]))
            log_prob_samples = np.concatenate((log_prob_samples,log_prob_s[i+1]))
            wp_samples = np.concatenate((wp_samples,wps[i+1]))
else:
    samples = s[0]
    log_prob_samples = log_prob_s[0]
    wp_samples = wps[0]
min_chi2_loc = np.where(-2*log_prob_samples==(-2*log_prob_samples.max()))
#best_wp = wp_samples[min_chi2_loc][0]
best_chi2 = (-2*log_prob_samples[min_chi2_loc])[0]

dsamples = samples[:,:20,:][::50,:]
print(dsamples.shape)
dsamples=samples[:,:20,:][::50,:].reshape(dsamples.shape[0]*dsamples.shape[1],5)

####Prepare data for nearest neighbor search
print('Median values:')
print(' logMvirmin, sigma_logMvir, alpha, logMvir_0, logMvir1')
params_median = np.median(dsamples, axis=0)
params_std = np.std(dsamples, axis=0)
print('  ', params_median[[0, 1, 2, 3, 4]])
print('+-', params_std[[0, 1, 2, 3, 4]])
print()

# Find the best-fit parameters in segments of the chain
params_1_sigma_range = (np.percentile(dsamples, 84, axis=0) - np.percentile(dsamples, 16, axis=0))/2

print('1-sigma range:')
print(' logMvirmin, sigma_logMvir, alpha, logMvir_0, logMvir1')
print('  ', params_1_sigma_range[[0, 1, 2, 3, 4]])


ndim=5
"""
fig = corner.corner(dsamples.reshape((-1,ndim)),
        labels=["$logVmaxMin$", "${\sigma}logVmax$", "$alpha$", "$logVmax_0$", "$logVmax$"],
        show_titles=True,title_kwargs={"fontsize": 10},quantiles=(0.16, 0.84),bins=20)#, levels=(1-np.exp(-0.5),))
"""

data = dsamples[:, :6]

# Reflect against hard prior boundaries
# sigma_logM

#data_new = np.copy(data)
#data_new[:, 1] = -data_new[:, 1]
#data = np.concatenate([data, data_new], axis=0)

#data_new = np.copy(data)
#data_new[:, 3] = -data_new[:, 3]
#data = np.concatenate([data, data_new], axis=0)

# rescaling the dimensions
data_rescale = data/params_1_sigma_range[None, :6]

# Find the distance to the n-th nearest neighbor
neigh = NearestNeighbors(n_neighbors=501, radius=np.inf, algorithm='kd_tree')
neigh.fit(data_rescale)
dist, _ = neigh.kneighbors(data_rescale, return_distance=True)

# Remove the reflected points
maxdist = np.max(dist, axis=1)
idx_best = np.argsort(maxdist)
mask = idx_best < len(dsamples)
print(np.sum(mask)/len(idx_best))
idx_best = idx_best[mask]

# select the top n_best points
n_best = 500
idx_best = idx_best[:n_best]
print(dsamples[idx_best][0])