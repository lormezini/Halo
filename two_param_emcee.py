#!/usr/bin/env python
# coding: utf-8


from halotools.sim_manager import CachedHaloCatalog, FakeSim
from halotools.empirical_models import PrebuiltHodModelFactory, Zheng07Cens, Zheng07Sats, TrivialPhaseSpace, NFWPhaseSpace, HodModelFactory
from halotools.mock_observables import return_xyz_formatted_array
import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool, cpu_count
import emcee
from Corrfunc.theory.wp import wp
from numpy.linalg import inv
from warnings import simplefilter
#ignore all future warnings                                                     
simplefilter(action='ignore', category=FutureWarning)
import zehavi_data_file_20


wp_ng_vals = zehavi_data_file_20.get_wp()
bin_edges = zehavi_data_file_20.get_bins()
cov_matrix = zehavi_data_file_20.get_cov()
wp_vals = wp_ng_vals[1:len(wp_ng_vals)]
invcov = inv(cov_matrix)
ng = wp_ng_vals[0]
ng_cov = 0.00007
halocat = CachedHaloCatalog(fname='/Users/lmezini/.astropy/cache/halotools/halo_catalogs/SMDPL/rockstar/2019-07-03-18-38-02-9731.dat.my_cosmosim_halos.hdf5')
pi_max = 60.
Lbox = 400.
fixed_params = [0.25,13.28-1.7,13.28]
alpha, logM0, logM1 = fixed_params

#cens_occ_model = Zheng07Cens(prim_haloprop_key = 'halo_vmax')
cens_occ_model = Zheng07Cens()
cens_prof_model = TrivialPhaseSpace()

#sats_occ_model =  Zheng07Sats(prim_haloprop_key = 'halo_vmax', modulate_with_cenocc=True)
sats_occ_model =  Zheng07Sats(modulate_with_cenocc=True)
sats_prof_model = NFWPhaseSpace()
model_instance = HodModelFactory(centrals_occupation = cens_occ_model, centrals_profile = cens_prof_model, 
                                 satellites_occupation = sats_occ_model, satellites_profile = sats_prof_model)
try:
    model_instance.mock.populate()
except:
    model_instance.populate_mock(halocat)


def _get_lnlike(theta):
    logMmin, alpha = theta # sigma_logM, logM0, logM1 = theta
    model_instance.param_dict['logMmin'] = logMmin
    model_instance.param_dict['sigma_logM'] = sigma_logM
    model_instance.param_dict['alpha'] = alpha
    model_instance.param_dict['logM0'] = logM0
    model_instance.param_dict['logM1'] = logM1
    
    model_instance.mock.populate()
    pos = return_xyz_formatted_array(model_instance.mock.galaxy_table['x'],
            model_instance.mock.galaxy_table['y'],
            model_instance.mock.galaxy_table['z'],
            period = Lbox)
    x = pos[:,0]
    y = pos[:,1]
    z = pos[:,2]
    velz = model_instance.mock.galaxy_table['vz']
    pos_zdist = return_xyz_formatted_array(x,y,z,period=Lbox,
                velocity=velz,velocity_distortion_dimension='z')
    pi_max = 60.
    nthreads = 1
    wp_calc = wp(Lbox,pi_max,nthreads,bin_edges,pos_zdist[:,0],
                     pos_zdist[:,1],pos_zdist[:,2],verbose=False,
                     xbin_refine_factor=3, ybin_refine_factor=3,
                     zbin_refine_factor=2)
    wp_diff = wp_vals-wp_calc['wp']
    ng_diff = ng-model_instance.mock.number_density
    
    return -0.5*np.dot(wp_diff, np.dot(invcov, wp_diff)) + -0.5*(ng_diff**2)/(ng_cov**2)


def _get_lnprior(theta):
    logMmin, sigma_logM = theta#, alpha, logM0, logM1 = theta
    if 10.< logMmin <13. and 0.25 < sigma_logM < 0.5: #and 1.0<alpha<1.3 and 11.<logM0<11.5 and 13.<logM1<13.5:
        return 0.0
    return -np.inf


def _get_lnprob(theta):
    lp = _get_lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + _get_lnlike(theta)


ndim, nwalkers = 2, 10
guess = [11.96,1.16]#1.16,11.3,13.28]
pos = [guess + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]


sampler = emcee.EnsembleSampler(nwalkers, ndim, _get_lnprob)
sampler.run_mcmc(pos, 1)



