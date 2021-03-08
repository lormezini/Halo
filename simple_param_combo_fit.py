from halotools.sim_manager import CachedHaloCatalog, FakeSim
from halotools.empirical_models import PrebuiltHodModelFactory, Zheng07Cens, Zheng07Sats, TrivialPhaseSpace, NFWPhaseSpace, HodModelFactory
from halotools.mock_observables import return_xyz_formatted_array,wp
from halotools import utils
import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool, cpu_count
#from pathos.multiprocessing import ProcessingPool as Pool
import emcee
#import corner
from Corrfunc.theory.wp import wp
import MCMC_data_file
from numpy.linalg import inv
import scipy.optimize as op
from scipy.stats import chi2
import scipy.stats as stats
import random
import warnings
warnings.filterwarnings("ignore")
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')
from scipy.special import gamma
from scipy.stats import chisquare
import gc

from astropy.table import Table 

threshold = 21
dname = "zehavi_data_file_21"
param = "mvir"


if '21' in dname:
    import zehavi_data_file_21
    wp_ng_vals = zehavi_data_file_21.get_wp()[0:12]
    bin_edges = zehavi_data_file_21.get_bins()[0:12]
    cov_matrix = zehavi_data_file_21.get_cov()[0:11,0:11]
    err = np.array([cov_matrix[i,i] for i in range(len(cov_matrix))])
    bin_cen = (bin_edges[1:]+bin_edges[:-1])/2.
    ng = wp_ng_vals[0]
    ng_err = 0.000005
    invcov = inv(cov_matrix)
    wp_vals = wp_ng_vals[1:12]
if '19' in dname:
    print('19')
    import zehavi_data_file_19
    wp_ng_vals = zehavi_data_file_19.get_wp()[0:12]
    bin_edges = zehavi_data_file_19.get_bins()[0:12]
    cov_matrix = zehavi_data_file_19.get_cov()[0:11,0:11]
    err = np.array([cov_matrix[i,i] for i in range(len(cov_matrix))])    
    bin_cen = (bin_edges[1:]+bin_edges[:-1])/2.
if '20' in dname:
    import zehavi_data_file_20
    wp_ng_vals = zehavi_data_file_20.get_wp()[0:12]
    bin_edges = zehavi_data_file_20.get_bins()[0:12]
    cov_matrix = zehavi_data_file_20.get_cov()[0:11,0:11]
    invcov = inv(cov_matrix)
    err = np.array([cov_matrix[i,i] for i in range(len(cov_matrix))])    
    bin_cen = (bin_edges[1:]+bin_edges[:-1])/2.
    ng = wp_ng_vals[0]
    ng_err = 0.00007
    wp_vals = wp_ng_vals[1:12]

halocat = CachedHaloCatalog(fname = '/home/lom31/.astropy/cache/halotools/halo_catalogs/SMDPL/rockstar/2019-07-03-18-38-02-9731.dat.my_cosmosim_halos.hdf5',update_cached_fname = True)
halocat.redshift = 0.
pi_max = 60.
Lbox = 400.

hmvir = halocat.halo_table['halo_mvir']
hvmax = halocat.halo_table['halo_vmax']

def _get_model_inst(a):

    halocat.halo_table.add_column((hmvir**a)*(hvmax**(1-a)),name='combo')
    
    cens_occ_model = Zheng07Cens(prim_haloprop_key = 'combo',
                                 threshold=threshold)
    cens_prof_model = TrivialPhaseSpace()

    sats_occ_model =  Zheng07Sats(prim_haloprop_key = 'combo',
                                  modulate_with_cenocc=True,
                                  threshold=threshold)
    sats_prof_model = NFWPhaseSpace()

    model_instance = HodModelFactory(centrals_occupation = cens_occ_model, 
                                     centrals_profile = cens_prof_model,
                                     satellites_occupation = sats_occ_model,
                                     satellites_profile = sats_prof_model)
    
    return model_instance

def _get_lnlike(theta):
    a,logMmin = theta

    model_instance = _get_model_inst(a)
    model_instance.param_dict['logMmin'] = logMmin
    model_instance.param_dict['sigma_logM'] = 0.74
    model_instance.param_dict['alpha'] = 1.27
    model_instance.param_dict['logM0'] = 13.96-1.7
    model_instance.param_dict['logM1'] = 13.96

    try:
        model_instance.mock.populate()
    except:
        model_instance.populate_mock(halocat)
        
    halo_table = model_instance.mock.halo_table
    pos = return_xyz_formatted_array(model_instance.mock.galaxy_table['x'], 
                                     model_instance.mock.galaxy_table['y'], 
                                     model_instance.mock.galaxy_table['z'],
                                     period = Lbox)
    x = pos[:,0]
    y = pos[:,1]
    z = pos[:,2]
    velz = model_instance.mock.galaxy_table['vz']
    pos_zdist = return_xyz_formatted_array(x,y,z, period = Lbox, 
                                           velocity=velz, 
                                           velocity_distortion_dimension='z')
    mod = wp(Lbox,pi_max,1,bin_edges,pos_zdist[:,0],pos_zdist[:,1],
             pos_zdist[:,2],verbose=False)

    ng_diff = ng-model_instance.mock.number_density
    wp_diff = wp_vals-mod['wp']

    del(halocat.halo_table['combo'])

    return -0.5*np.dot(wp_diff, np.dot(invcov, wp_diff)) + -0.5*(ng_diff**2)/(ng_err**2)

def _get_lnprior(theta):
    a,logMmin = theta
    if 0.0< a <1.0 and 7.0 < logMmin < 14.0:
        return 0.0
    return -np.inf

def _get_lnprob(theta):
    lp = _get_lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + _get_lnlike(theta)

ndim, nwalkers = 2, 35
nsteps = 10000
guess = [0.85,11.]                                               
pos = [guess + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
backend = emcee.backends.HDFBackend('combo_param_mv_fit_m21.h5')
backend.reset(nwalkers, ndim)

#with Pool() as pool:
sampler = emcee.EnsembleSampler(nwalkers, ndim, _get_lnprob,
                                    backend=backend)
start = time.time()
sampler.run_mcmc(pos, nsteps,progress=True)
end = time.time()
multi_time = end-start
print("Multiprocessing took {0:.1f} seconds".format(multi_time))