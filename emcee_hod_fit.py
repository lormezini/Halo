from halotools.sim_manager import CachedHaloCatalog, FakeSim
from halotools.empirical_models import PrebuiltHodModelFactory, Zheng07Cens, Zheng07Sats, TrivialPhaseSpace, NFWPhaseSpace, HodModelFactory
from halotools.mock_observables import return_xyz_formatted_array
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool, cpu_count
import emcee
from Corrfunc.theory.wp import wp
import yaml
import logging
from numpy.linalg import inv
from warnings import simplefilter
#ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

import argparse
parser = argparse.ArgumentParser(description='data type')
parser.add_argument('--config',type=str, help='yaml file')

class fit_dict(dict):
    """perform monte carlo to fit wp HOD parameters"""
    
    def __init__(self,config):

        self.update(config)
        self.sim = self['sim']
        self.param = self['param']
        self.output = self['output']
        self.nwalkers = self['nwalkers']
        self.ndim = self['ndim']
        self.nsteps = self['nsteps']
        self.guess = self['guess']
        self.prior_ranges = self['prior_ranges']
        self.file_ext = self['file_ext']

    def _get_log(self):
        log_fname = str(self['output'][0:-2])+'log'
        logger = logging.getLogger(str(self['output'][0:-2]))
        hdlr = logging.FileHandler(log_fname,mode='w')
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)
        logger.setLevel(logging.INFO)

        logger.info(self['output'])
        logger.info("guess: " + str(self['guess']))
        logger.info("ndim, nwalkers, nsteps: "+str(self['ndim'])+","+
                    str(self['nwalkers'])+","+str(self['nsteps']))
        logger.info("logMmin_r: " + str(self['prior_ranges']['logMmin_r']))
        logger.info("sigma_logM_r: " + str(self['prior_ranges']['sigma_logM_r']))
        logger.info("alpha_r: " + str(self['prior_ranges']['alpha_r']))
        logger.info("logM0_r: " + str(self['prior_ranges']['logM0_r']))
        logger.info("logM1_r: " + str(self['prior_ranges']['logM1_r']))

        return logger

    def _get_data(self):
        if self['data'] == "zehavi":
            data_file = self['data_file']
            if '19' in data_file:
                import zehavi_data_file_19
                wp_ng_vals = zehavi_data_file_19.get_wp()
                bin_edges = zehavi_data_file_19.get_bins()
                cov_matrix = zehavi_data_file_19.get_cov()
                ng_err = 0.000068
            elif '20' in data_file:
                import zehavi_data_file_20
                wp_ng_vals = zehavi_data_file_20.get_wp()
                bin_edges = zehavi_data_file_20.get_bins()
                cov_matrix = zehavi_data_file_20.get_cov()
                ng_err = 0.00007
            elif '21' in data_file:
                import zehavi_data_file_21
                wp_ng_vals = zehavi_data_file_21.get_wp()[0:12]
                bin_edges = zehavi_data_file_21.get_bins()[0:12]
                cov_matrix = zehavi_data_file_21.get_cov()[0:11,0:11]
                ng_err = 0.000005
            invcov = inv(cov_matrix)
            ng = wp_ng_vals[0]
            wp_vals = wp_ng_vals[1:len(wp_ng_vals)]

            return wp_vals, ng, ng_err, bin_edges, invcov
        
        elif self['data'] == "mock":
            import mock_data_2
            wp_vals = mock_data_2.get_wp()
            bin_edges = mock_data_2.get_bin_edges()
            cov_matrix = mock_data_2.get_cov()
            invcov = inv(cov_matrix)
            ng_err = mock_data_2.get_ng_err()
            ng = mock_data_2.get_ng()

            return wp_vals, ng, ng_err, bin_edges, invcov

        elif self['data'] == "guo":
            wp_ng_vals = guo_data_file.get_wp()
            bin_edges = guo_data_file.get_bins()
            cov_matrix = guo_data_file.get_cov()
            invcov = inv(cov_matrix)
            ng_err = 0.00003
            wp_vals = wp_ng_vals[1:len(wp_ng_vals)]
            ng = wp_ng_vals[0]

            return wp_vals, ng, ng_err, bin_edges, invcov
    
class hod_fit(fit_dict):

    def __init__(self,config):
        super().__init__(config)
        self.cur_wp = np.zeros(11)

        if self['sim'] == "bolplanck":
            halocat = CachedHaloCatalog(simname = 'bolplanck')
        elif self['sim'] == "old":
            halocat = CachedHaloCatalog(fname = '/home/lom31/Halo/hlist_1.00231.list.halotools_v0p1.hdf5',update_cached_fname = True)
            halocat.redshift=0.
        elif self['sim']== "smdpl":
            #halocat = CachedHaloCatalog(fname = '/home/lom31/.astropy/cache/halotools/halo_catalogs/SMDPL/rockstar/2019-07-03-18-38-02-9731.dat.my_cosmosim_halos.hdf5',update_cached_fname = True)
            halocat = CachedHaloCatalog(fname='/home/lom31/.astropy/cache/halotools/halo_catalogs/smdpl/rockstar/2019-07-03-18-38-02-9731.dat.my_cosmosim_halos.hdf5',update_cached_fname = True)
            halocat.redshift=0.
        elif self['sim'] == "mdr1":
            halocat = CachedHaloCatalog(fname = '/home/lom31/.astropy/cache/halotools/halo_catalogs/multidark/rockstar/hlist_0.68215.list.halotools_v0p4.hdf5',update_cached_fname = True)
        
        if self['param'] == 'mvir':
            cens_occ_model = Zheng07Cens(threshold=-21)
            cens_prof_model = TrivialPhaseSpace()
            sats_occ_model =  Zheng07Sats(modulate_with_cenocc=True,threshold=-21)
            sats_prof_model = NFWPhaseSpace()
        elif self['param'] == 'vmax':
            cens_occ_model = Zheng07Cens(prim_haloprop_key = 'halo_vmax')
            cens_prof_model = TrivialPhaseSpace()
            sats_occ_model = Zheng07Sats(prim_haloprop_key = 'halo_vmax',
                                         modulate_with_cenocc=True)
            sats_prof_model = NFWPhaseSpace()

        global model_instance

        model_instance = HodModelFactory(centrals_occupation = cens_occ_model,
                                              centrals_profile = cens_prof_model,
                                              satellites_occupation = sats_occ_model,
                                              satellites_profile = sats_prof_model)

        try:
            model_instance.mock.populate()
        except:
            model_instance.populate_mock(halocat)

    def _get_lnlike(self,theta):

        wp_vals, ng, ng_err, bin_edges, invcov = self._get_data()
        
        logMmin, sigma_logM, alpha, logM0, logM1 = theta
        model_instance.param_dict['logMmin'] = logMmin
        model_instance.param_dict['sigma_logM'] = sigma_logM
        model_instance.param_dict['alpha'] = alpha
        model_instance.param_dict['logM0'] = logM0
        model_instance.param_dict['logM1'] = logM1

        if self['sim'] == 'smdpl':
            Lbox = 400.
        elif self['sim'] == 'old':
            Lbox = 250.
        elif self['sim'] == 'bolplanck':
            Lbox = 250.
        elif self['sim'] == 'mdr1':
            Lbox=1000.
        
        model_instance.mock.populate()
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
        #,xbin_refine_factor=3, ybin_refine_factor=3, zbin_refine_factor=2)

        wp_diff = wp_vals-wp_calc['wp']

        ng_diff = ng-model_instance.mock.number_density

        #save current wp calculated value for blobs
        self.cur_wp = wp_calc['wp']
        return -0.5*np.dot(wp_diff, np.dot(invcov, wp_diff)) + -0.5*(ng_diff**2)/(ng_err**2)

    def _get_lnprior(self,theta):

        logMmin_r = self['prior_ranges']['logMmin_r']
        sigma_logM_r = self['prior_ranges']['sigma_logM_r']
        alpha_r = self['prior_ranges']['alpha_r']
        logM0_r = self['prior_ranges']['logM0_r']
        logM1_r = self['prior_ranges']['logM1_r']

        logMmin, sigma_logM, alpha, logM0, logM1 = theta

        if logMmin_r[0]<logMmin<logMmin_r[1] and sigma_logM_r[0]<sigma_logM<sigma_logM_r[1] and alpha_r[0]<alpha<alpha_r[1] and logM0_r[0]<logM0<logM0_r[1] and logM1_r[0]<logM1<logM1_r[1]:     
            return 0.0
        return -np.inf

    def _get_lnprob(self,theta):
        
        lp = self._get_lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf, self.cur_wp

        ll = self._get_lnlike(theta)
        return lp + ll, self.cur_wp

    def _get_pos(self):
            
        pos = [self.guess+1e-4*np.random.randn(self.ndim) for i in range(self.nwalkers)]

        return pos

def main(args):
    # Set up the backend
    # Don't forget to clear it in case the file already exists
    with open(args.config) as fobj:
        config = yaml.load(fobj)
    hod = hod_fit(config)    
    filename = hod.output
    file_ext = hod.file_ext
    nwalkers = hod.nwalkers
    ndim = hod.ndim
    nsteps = hod.nsteps
    pos = hod._get_pos()

    fd = fit_dict(config)
    logger = fd._get_log()
    backend = emcee.backends.HDFBackend(file_ext+filename)
    backend.reset(nwalkers, ndim)
    with Pool(15) as pool:
            
        sampler = emcee.EnsembleSampler(nwalkers, ndim, hod._get_lnprob, 
                                        pool=pool,backend=backend)
        start = time.time()
        sampler.run_mcmc(pos, nsteps, progress=False, store=True)
        end = time.time()
        multi_time = end - start
        print("Multiprocessing took {0:.1f} seconds".format(multi_time))
    logger.info("Final size: {0}".format(backend.iteration))

if __name__=='__main__':
    args = parser.parse_args()
    main(args)
