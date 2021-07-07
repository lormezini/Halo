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
from numpy.linalg import inv
import scipy.optimize as op
from scipy.stats import chi2
import scipy.stats as stats
import random
import warnings
warnings.filterwarnings("ignore")
from scipy.special import gamma
from scipy.stats import chisquare
import gc
from tabcorr import TabCorr
from os import walk
from astropy.table import Table 
import logging

threshold = -20
dname = "zehavi_data_file_20"
param = "combo"
output = 'combo_param_m20_a50.h5'

#guess = [ 0.5, 13., 0.5, 1.6, 12.06, 13.]
guess = [ 0.50, 11.9, 0.2, 1., 12.4, 13.]
backend = emcee.backends.HDFBackend(output)

log_fname = str(output[0:-2])+'log'
logger = logging.getLogger(output[0:-2])
hdlr = logging.FileHandler(log_fname,mode='w')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)

logger.info(output)
logger.info("guess: " + str(guess))

gc.collect()

if '21' in dname:
    print(threshold)
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

halocat = CachedHaloCatalog(fname='/home/lom31/Halo/smdpl.dat.smdpl2.hdf5',update_cached_fname = True)
halocat.redshift=0.
ht = halocat.halo_table
vmax_ml = 10**(np.log10(ht['halo_vmax'])*3.1069839419403174+5.015822745222037)
ht.add_column(vmax_ml, name='vmax_ml')

pi_max = 60.
Lbox = 400.

hmvir = halocat.halo_table['halo_mvir']
hvmax = halocat.halo_table['vmax_ml']


class TabNames(object):
    def __init__(self, f):
        self.f = f
        self.a = float(f[:-5].split('combo_a')[1])
        self.model_instance = _get_model_inst(self.a)

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
    del(halocat.halo_table['combo'])
    return model_instance

def _get_wp_ng(model_instance,f):
    ngals,wp = TabCorr.read('/home/lom31/Halo/tabcorr_tables/{}'.format(f)).predict(model_instance)
    return ngals,wp

_, _, filenames = next(walk('/home/lom31/Halo/tabcorr_tables/'))
tab_names = [TabNames(f) for f in filenames]
mod_inst = [TabNames(f).model_instance for f in filenames]
tab_names_dict = dict(zip(filenames, tab_names))
mod_names_dict = dict(zip(filenames, mod_inst))
gc.collect()



def _get_lnlike(theta):
    a,logMmin,sigma_logM,alpha,logM0, logM1 = theta

    model_instance = mod_names_dict['smdpl_combo_a{}.hdf5'.format(round(a,2))]

    #model_instance = _get_model_inst(a)
    model_instance.param_dict['logMmin'] = logMmin
    model_instance.param_dict['sigma_logM'] = sigma_logM
    model_instance.param_dict['alpha'] = alpha
    model_instance.param_dict['logM0'] = logM0
    model_instance.param_dict['logM1'] = logM1

    """
    try:
        model_instance.mock.populate()
    except:
        model_instance.populate_mock(halocat)

    pos = return_xyz_formatted_array(model_instance.mock.galaxy_table['x'],
                                         model_instance.mock.galaxy_table['y'],
                                         model_instance.mock.galaxy_table['z'],
                                         period = Lbox)
    x = pos[:,0]
    y = pos[:,1]
    z = pos[:,2]

    velz = model_instance.mock.galaxy_table['vz']
    pos_zdist = return_xyz_formatted_array(x,y,z,period=Lbox,velocity=velz,
                                velocity_distortion_dimension='z')

    pi_max = 60.
    nthreads = 1
    wp_calc = wp(Lbox,pi_max,nthreads,bin_edges,pos_zdist[:,0],
                     pos_zdist[:,1],pos_zdist[:,2],verbose=False)

    wp_diff = wp_vals-wp_calc['wp']
    ng_diff = ng-model_instance.mock.number_density
    """

    ngal,wp = _get_wp_ng(model_instance,'smdpl_combo_a{}.hdf5'.format(round(a,2)))
    
    #ngal, wp = halotab.predict(model_instance)
    wp_diff = wp_vals-wp
    ng_diff = ng-ngal

    gc.collect()
    
    return -0.5*np.dot(wp_diff, np.dot(invcov, wp_diff)) + -0.5*(ng_diff**2)/(ng_err**2)

def _get_lnprior(theta):
    a,logMmin,sigma_logM,alpha,logM0,logM1 = theta
    if 0.0< a <1.0 and 10.0 < logMmin < 15.0 and 0.0 < sigma_logM < 3.0 and 0.0 < alpha < 4.0 and 10.0 < logM0 < 15.0 and 10.0 < logM1 < 15.0:
        return 0.0
    return -np.inf

def _get_lnprob(theta):
    lp = _get_lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + _get_lnlike(theta)

ndim, nwalkers = 6, 35
nsteps = 250000
logger.info('ndim, nwalkers, nsteps: {},{},{}'.format(ndim,nwalkers,nsteps))

#guess = [0.558, 8.172, 0.200, 1.411, 8.373, 8.937]
pos = [guess + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
#backend.reset(nwalkers, ndim)
with Pool(15) as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, _get_lnprob,
                                    backend=backend, pool=pool)
    start = time.time()
    sampler.run_mcmc(pos, nsteps,progress=True)
    end = time.time()
multi_time = end-start
print("Multiprocessing took {0:.1f} seconds".format(multi_time))
logger.info("Final size: {0}".format(backend.iteration))
