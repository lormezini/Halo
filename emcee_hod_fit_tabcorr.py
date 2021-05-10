from halotools.sim_manager import CachedHaloCatalog, FakeSim
from halotools.empirical_models import PrebuiltHodModelFactory, Zheng07Cens, Zheng07Sats, TrivialPhaseSpace, NFWPhaseSpace, HodModelFactory
from halotools.mock_observables import return_xyz_formatted_array
import os
#os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import time
from multiprocessing import Pool, cpu_count
import emcee
import yaml
import logging
from numpy.linalg import inv
from warnings import simplefilter
#ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
from tabcorr import TabCorr

import linecache
import os
import tracemalloc

def display_top(snapshot, key_type='lineno', limit=20):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print("#%s: %s:%s: %.1f KiB"
              % (index, frame.filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))

tracemalloc.start()
import gc

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
        gc.collect()

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
        gc.collect()
        return logger

    def _get_data(self):
        if self['data'] == "zehavi":
            data_file = self['data_file']
            if '19' in data_file:
                import zehavi_data_file_19
                wp_ng_vals = zehavi_data_file_19.get_wp()[0:12]
                bin_edges = zehavi_data_file_19.get_bins()[0:12]
                cov_matrix = zehavi_data_file_19.get_cov()[0:11,0:11]
                ng_err = 0.000068
            elif '20_noGW' in data_file:
                import zehavi_data_file_20_noGW
                wp_ng_vals = zehavi_data_file_20_noGW.get_wp()[0:12]
                bin_edges = zehavi_data_file_20_noGW.get_bins()[0:12]
                cov_matrix = zehavi_data_file_20_noGW.get_cov()[0:11,0:11]
                ng_err = 0.00007/np.sqrt(30245)
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

            gc.collect()
            if 'vmax' in self['param']:
                halotab = TabCorr.read('smdpl_vmax_ml.hdf5')
            else:
                halotab = TabCorr.read('smdpl_halo_mvir.hdf5')
            return wp_vals, ng, ng_err, bin_edges, invcov, halotab
        
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
            halocat = CachedHaloCatalog(fname = '/home/lom31/Halo/smdpl.dat.smdpl2.hdf5',update_cached_fname=True)
            halocat.redshift=0.
            ht = halocat.halo_table
            vmax_ml = 10**(np.log10(ht['halo_vmax'])*3.2129514846864926+4.71149353971438)
            ht.add_column(vmax_ml, name='vmax_ml')
        elif self['sim'] == "mdr1":
            halocat = CachedHaloCatalog(fname = '/home/lom31/.astropy/cache/halotools/halo_catalogs/multidark/rockstar/hlist_0.68215.list.halotools_v0p4.hdf5',update_cached_fname = True)
        
        if self['data'] == "zehavi":
            data_file = self['data_file']
            if '19' in data_file:
                Threshold = -19
            if '20' in data_file:
                Threshold = -20
            if '21' in data_file:
                Threshold = -21
            
        if self['param'] == 'mvir':
            cens_occ_model = Zheng07Cens(threshold=Threshold)
            cens_prof_model = TrivialPhaseSpace()
            sats_occ_model =  Zheng07Sats(modulate_with_cenocc=True,threshold=Threshold)
            sats_prof_model = NFWPhaseSpace()

        elif self['param'] == 'vmax':
            cens_occ_model = Zheng07Cens(prim_haloprop_key = 'halo_vmax',
                                         threshold=Threshold)
            cens_prof_model = TrivialPhaseSpace()
            sats_occ_model = Zheng07Sats(prim_haloprop_key = 'halo_vmax',
                                         modulate_with_cenocc=True,
                                         threshold=Threshold)
            sats_prof_model = NFWPhaseSpace()
        elif self['param'] == 'vmax_ml':
            cens_occ_model = Zheng07Cens(prim_haloprop_key = 'vmax_ml',
                                         threshold=Threshold)
            cens_prof_model = TrivialPhaseSpace()
            sats_occ_model = Zheng07Sats(prim_haloprop_key = 'vmax_ml',
                                         modulate_with_cenocc=True,
                                         threshold=Threshold)
            sats_prof_model = NFWPhaseSpace()

        
        gc.collect()
        global model_instance

        model_instance = HodModelFactory(centrals_occupation = cens_occ_model,
                                              centrals_profile = cens_prof_model,
                                              satellites_occupation = sats_occ_model,
                                              satellites_profile = sats_prof_model)


    def _get_lnlike(self,theta):

        wp_vals, ng, ng_err, bin_edges, invcov, halotab = self._get_data()
        
        logMmin, sigma_logM, alpha, logM0, logM1 = theta
        model_instance.param_dict['logMmin'] = logMmin
        model_instance.param_dict['sigma_logM'] = sigma_logM
        model_instance.param_dict['alpha'] = alpha
        model_instance.param_dict['logM0'] = logM0
        model_instance.param_dict['logM1'] = logM1

        ngal, wp = halotab.predict(model_instance)
        wp_diff = wp_vals-wp
        ng_diff = ng-ngal
        #save current wp calculated value for blobs
        self.cur_wp = wp
        gc.collect()

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
    #backend.reset(nwalkers, ndim)
    gc.collect()
    with Pool(15) as pool:
            
        sampler = emcee.EnsembleSampler(nwalkers, ndim, hod._get_lnprob, 
                                        pool=pool,backend=backend)
        #start = time.time()
        sampler.run_mcmc(pos, nsteps, progress=False, store=True)
        #end = time.time()
        #multi_time = end - start
        #print("Multiprocessing took {0:.1f} seconds".format(multi_time))
    logger.info("Final size: {0}".format(backend.iteration))

if __name__=='__main__':
    args = parser.parse_args()
    main(args)
