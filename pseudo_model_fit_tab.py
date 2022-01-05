from halotools.sim_manager import CachedHaloCatalog
from halotools.empirical_models import PrebuiltHodModelFactory, Zheng07Cens, Zheng07Sats, TrivialPhaseSpace, NFWPhaseSpace, HodModelFactory
import numpy as np
import time
from astropy.utils.misc import NumpyRNGContext
from scipy.interpolate import interp1d
from scipy.special import erf,pdtrik
from halotools.utils.array_utils import custom_len
import warnings
from Corrfunc.theory.wp import wp
from halotools.mock_observables import return_xyz_formatted_array
import gc
from numpy.linalg import inv
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import emcee
from os import walk
import logging
from memory_profiler import profile
import resource
from tabcorr import TabCorr

threshold = -21
dname = "zehavi_data_file_21"
param = "pseudo"
output = 'mass_concentration_fit_m21_new_mod_tab.h5'

guess = [0.09, 12.67, 0.19, 1.05, 12.93, 13.93]
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

def bind_required_kwargs(required_kwargs, obj, **kwargs):

    for key in required_kwargs:
        if key in list(kwargs.keys()):
            setattr(obj, key, kwargs[key])
        else:
            class_name = obj.__class__.__name__
            msg = (
                key + ' is a required keyword argument ' +
                'to instantiate the '+class_name+' class'
                )
            raise KeyError(msg)

class OccupationComponent(object):

    def __init__(self, **kwargs):
        
        required_kwargs = ["gal_type", "threshold"]
        bind_required_kwargs(required_kwargs, self, **kwargs)

        try:
            self.prim_haloprop_key = kwargs["prim_haloprop_key"]
            self.sec_haloprop_key = kwargs["sec_haloprop_key"]
        except:
            pass

        try:
            self._upper_occupation_bound = kwargs["upper_occupation_bound"]
        except KeyError:
            msg = "\n``upper_occupation_bound`` is a required keyword argument of OccupationComponent\n"
            raise KeyError(msg)

        self._lower_occupation_bound = 0.0

        self._second_moment = kwargs.get("second_moment", "poisson")

        if not hasattr(self, "param_dict"):
            self.param_dict = {}

        # Enforce the requirement that sub-classes have been configured properly
        required_method_name = "mean_occupation"
        if not hasattr(self, required_method_name):
            raise SyntaxError(
                "Any sub-class of OccupationComponent must "
                "implement a method named %s " % required_method_name
            )

        try:
            self.redshift = kwargs["redshift"]
        except KeyError:
            pass

        # The _methods_to_inherit determines which methods will be directly callable
        # by the composite model built by the HodModelFactory
        try:
            self._methods_to_inherit.extend(["mc_occupation", "mean_occupation"])
        except AttributeError:
            self._methods_to_inherit = ["mc_occupation", "mean_occupation"]

        # The _attrs_to_inherit determines which methods will be directly bound
        # to the composite model built by the HodModelFactory
        try:
            self._attrs_to_inherit.append("threshold")
        except AttributeError:
            self._attrs_to_inherit = ["threshold"]

        if not hasattr(self, "publications"):
            self.publications = []

        # The _mock_generation_calling_sequence determines which methods
        # will be called during mock population, as well as in what order they will be called
        self._mock_generation_calling_sequence = ["mc_occupation"]
        self._galprop_dtypes_to_allocate = np.dtype(
            [("halo_num_" + self.gal_type, "i4")]
        )

    def mc_occupation(self, seed=None, **kwargs):
        
        first_occupation_moment = self.mean_occupation(**kwargs)
        if self._upper_occupation_bound == 1:
            return self._nearest_integer_distribution(
                first_occupation_moment, seed=seed, **kwargs
            )
        elif self._upper_occupation_bound == float("inf"):
            if self._second_moment == "poisson":
                return self._poisson_distribution(
                    first_occupation_moment, seed=seed, **kwargs
                )
            elif self._second_moment == "weighted_nearest_integer":
                return self._weighted_nearest_integer(
                    first_occupation_moment, seed=seed, **kwargs
                )
            else:
                raise ValueError("Unrecognized second moment")
        else:
            msg = (
                "\nYou have chosen to set ``_upper_occupation_bound`` to some value \n"
                "besides 1 or infinity. In such cases, you must also \n"
                "write your own ``mc_occupation`` method that overrides the method in the \n"
                "OccupationComponent super-class\n"
            )
            raise HalotoolsError(msg)


    def _nearest_integer_distribution(
        self, first_occupation_moment, seed=None, **kwargs
    ):
        
        with NumpyRNGContext(seed):
            mc_generator = np.random.random(custom_len(first_occupation_moment))

        result = np.where(mc_generator < first_occupation_moment, 1, 0)
        if "table" in kwargs:
            kwargs["table"]["halo_num_" + self.gal_type] = result
        return result

    def _poisson_distribution(self, first_occupation_moment, seed=None, **kwargs):
        
        # We don't use the built-in Poisson number generator so that when a seed
        # is specified, it preserves the ranks among rvs even when mean is changed.
        with NumpyRNGContext(seed):
            result = np.ceil(
                pdtrik(
                    np.random.rand(*first_occupation_moment.shape),
                    first_occupation_moment,
                )
            ).astype(np.int)
        if "table" in kwargs:
            kwargs["table"]["halo_num_" + self.gal_type] = result
        return result

    def _weighted_nearest_integer(self, first_occupation_moment, seed=None, **kwargs):
        
        nsat_lo = np.floor(first_occupation_moment)
        with NumpyRNGContext(seed):
            uran = np.random.uniform(nsat_lo, nsat_lo + 1, first_occupation_moment.size)
        result = np.where(uran > first_occupation_moment, nsat_lo, nsat_lo + 1)
        if "table" in kwargs:
            kwargs["table"]["halo_num_" + self.gal_type] = result
        return result

halocat = CachedHaloCatalog(fname='/home/lom31/Halo/smdpl.dat.smdpl2.hdf5',update_cached_fname = True)

halocat.redshift=0.
G =  4.302*(10**-9.)
mvir = halocat.halo_table['halo_mvir']
rvir = halocat.halo_table['halo_rvir']
hms = halocat.halo_table['halo_halfmass_scale']
vmax = halocat.halo_table['halo_vmax']
vvir = np.sqrt((G*mvir)/rvir)

cnfw = np.linspace(1, 1400,140000)
vmax_interp = np.sqrt(0.2162166 * 1.0/( (np.log(1+cnfw)/cnfw) - (1/(1+cnfw)) ) )
get_c = interp1d(vmax_interp,cnfw,fill_value='extrapolate')
c = get_c(vmax/vvir)

num_bins = abs(int((np.min(np.log10(mvir))-np.max(np.log10(mvir)))*10))
bins=np.logspace(np.min(np.log10(mvir)),np.max(np.log10(mvir)),num=num_bins)

delta = []
mass = []
con = []
for i in range(1,num_bins):
    mask = np.logical_and(mvir>bins[i-1],mvir<bins[i])
    delta.append((c[mask]-np.mean(c[mask]))/np.std(c[mask]))
    mass.append(mvir[mask])
    con.append(c[mask])
mass = np.concatenate(mass)
delta = np.concatenate(delta)[np.argsort(mass)[::-1]]
con = np.concatenate(con)[np.argsort(mass)[::-1]]
mass = mass[np.argsort(mass)[::-1]]

halocat.halo_table['delta'] = delta

class mod_zheng07Cens(OccupationComponent):
    
    def __init__(self,
            threshold=20,
            prim_haloprop_key="halo_mvir",
            sec_haloprop_key ="delta"
            ):
        
        upper_occupation_bound = 1.0

        # Call the super class constructor, which binds all the
        # arguments to the instance.
        super(mod_zheng07Cens, self).__init__(
            gal_type='centrals',threshold=threshold, upper_occupation_bound=upper_occupation_bound,
            prim_haloprop_key=prim_haloprop_key, sec_haloprop_key = sec_haloprop_key,
            )
        mass_params = self.get_published_parameters(self.threshold)
        self.param_dict = (
                {'logMmin': mass_params['logMmin'],
                'sigma_logM': mass_params['sigma_logM'],
                'a': 0.0}
                )
        self.publications = ['arXiv:0408564', 'arXiv:0703457']
        print(self.param_dict)
    def mean_occupation(self,**kwargs):
        
        # Retrieve the array storing the mass-like variable
        if 'table' in list(kwargs.keys()):
            prop1 = kwargs['table'][self.prim_haloprop_key]
            prop2 = kwargs['table'][self.sec_haloprop_key]
        elif 'prim_haloprop' in list(kwargs.keys()):
            prop1 = np.atleast_1d(kwargs['prim_haloprop'])
            prop2 = np.atleast_1d(kwargs['sec_haloprop'])
        else:
            msg = ("\nYou must pass either a ``table`` or ``prim_haloprop`` argument \n"
                "to the ``mean_occupation`` function of the ``Zheng07Cens`` class.\n")
            print(msg)

        logM = np.log10(prop1) + self.param_dict['a']*prop2
        mean_ncen = 0.5*(1.0 + erf(
            (logM - self.param_dict['logMmin']) / self.param_dict['sigma_logM']))

        return mean_ncen

    def get_published_parameters(self, threshold, publication='Zheng07'):
        
        def get_zheng07_params(threshold):
            # Load tabulated data from Zheng et al. 2007, Table 1
            logMmin_array = [11.35, 11.46, 11.6, 11.75, 12.02, 12.3, 12.79, 13.38, 14.22]
            sigma_logM_array = [0.25, 0.24, 0.26, 0.28, 0.26, 0.21, 0.39, 0.51, 0.77]
            # define the luminosity thresholds corresponding to the above data
            threshold_array = np.arange(-22, -17.5, 0.5)
            threshold_array = threshold_array[::-1]

            threshold_index = np.where(threshold_array == threshold)[0]
            if len(threshold_index) == 0:
                msg = ("\nInput luminosity threshold "
                    "does not match any of the Table 1 values \nof "
                    "Zheng et al. 2007 (arXiv:0703457).\n"
                    "Choosing the best-fit parameters "
                    "associated the default_luminosity_threshold variable \n"
                    "set in the model_defaults module.\n"
                    "You can always manually change the values in ``param_dict``.\n")
                warnings.warn(msg)
                threshold = 20
                threshold_index = np.where(threshold_array == threshold)[0]

            mass_param_dict = (
                {'logMmin': logMmin_array[threshold_index[0]],
                'sigma_logM': sigma_logM_array[threshold_index[0]]}
                )

            return mass_param_dict

        if publication in ['zheng07', 'Zheng07', 'Zheng_etal07', 'zheng_etal07', 'zheng2007', 'Zheng2007']:
            param_dict = get_zheng07_params(threshold)
            return param_dict
        else:
            raise KeyError("For Zheng07Cens, only supported best-fit models are currently Zheng et al. 2007")

class mod_zheng07Sats(OccupationComponent):
    

    def __init__(self,
            threshold=20,
            prim_haloprop_key="halo_mvir", sec_haloprop_key = "delta",
            modulate_with_cenocc=True, cenocc_model=None):
        
        upper_occupation_bound = float("inf")

        # Call the super class constructor, which binds all the
        # arguments to the instance.
        super(mod_zheng07Sats, self).__init__(gal_type='satellites', threshold=threshold,
            upper_occupation_bound=upper_occupation_bound,
            prim_haloprop_key=prim_haloprop_key,
            sec_haloprop_key=sec_haloprop_key,
        )

        mass_params = self.get_published_parameters(self.threshold)
        self.param_dict = (
                {'logM0': mass_params['logM0'],
                'logM1': mass_params['logM1'],
                'alpha': mass_params['alpha']}
                )
        self.publications = ['arXiv:0308519', 'arXiv:0703457']

        if cenocc_model is None:
            cenocc_model = mod_zheng07Cens(
                prim_haloprop_key=prim_haloprop_key, 
                sec_haloprop_key=sec_haloprop_key,
                threshold=threshold)
        else:
            if modulate_with_cenocc is False:
                msg = ("You chose to input a ``cenocc_model``, but you set the \n"
                    "``modulate_with_cenocc`` keyword to False, so your "
                    "``cenocc_model`` will have no impact on the model's behavior.\n"
                    "Be sure this is what you intend before proceeding.\n"
                    "Refer to the Zheng et al. (2007) composite model tutorial for details.\n")
                warnings.warn(msg)

        self.modulate_with_cenocc = modulate_with_cenocc
        if self.modulate_with_cenocc:
            try:
                assert isinstance(cenocc_model, OccupationComponent)
            except AssertionError:
                msg = ("The input ``cenocc_model`` must be an instance of \n"
                    "``OccupationComponent`` or one of its sub-classes.\n")
                print(msg)

            self.central_occupation_model = cenocc_model

            self.param_dict.update(self.central_occupation_model.param_dict)


    def mean_occupation(self,**kwargs):

        if self.modulate_with_cenocc:
            for key, value in list(self.param_dict.items()):
                if key in self.central_occupation_model.param_dict:
                    self.central_occupation_model.param_dict[key] = value

        # Retrieve the array storing the mass-like variable
        if 'table' in list(kwargs.keys()):
            prop1 = kwargs['table'][self.prim_haloprop_key]
            prop2 = kwargs['table'][self.sec_haloprop_key]
        elif 'prim_haloprop' in list(kwargs.keys()):
            prop1 = np.atleast_1d(kwargs['prim_haloprop'])
            prop2 = np.atleast_1d(kwargs['sec_haloprop'])
        else:
            msg = ("\nYou must pass either a ``table`` or ``prim_haloprop`` argument \n"
                "to the ``mean_occupation`` function of the ``Zheng07Sats`` class.\n")
            print(msg)

        M0 = 10.**self.param_dict['logM0']
        M1 = 10.**self.param_dict['logM1']
        a = self.param_dict['a']
        # Call to np.where raises a harmless RuntimeWarning exception if
        # there are entries of input logM for which mean_nsat = 0
        # Evaluating mean_nsat using the catch_warnings context manager
        # suppresses this warning
        mass = 10.**(np.log10(prop1)+a*prop2)
        mean_nsat = np.zeros_like(mass)

        idx_nonzero = np.where(mass - M0 > 0)[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            mean_nsat[idx_nonzero] = ((mass[idx_nonzero] - M0)/M1)**self.param_dict['alpha']

        # If a central occupation model was passed to the constructor,
        # multiply mean_nsat by an overall factor of mean_ncen
        if self.modulate_with_cenocc:
            # compatible with AB models
            mean_ncen = getattr(self.central_occupation_model, "baseline_mean_occupation",\
                                    self.central_occupation_model.mean_occupation)(**kwargs)
            mean_nsat *= mean_ncen

        return mean_nsat

    def get_published_parameters(self, threshold, publication='Zheng07'):

        def get_zheng07_params(threshold):
            # Load tabulated data from Zheng et al. 2007, Table 1
            logM0_array = [11.2, 10.59, 11.49, 11.69, 11.38, 11.84, 11.92, 13.94, 14.0]
            logM1_array = [12.4, 12.68, 12.83, 13.01, 13.31, 13.58, 13.94, 13.91, 14.69]
            alpha_array = [0.83, 0.97, 1.02, 1.06, 1.06, 1.12, 1.15, 1.04, 0.87]
            # define the luminosity thresholds corresponding to the above data
            threshold_array = np.arange(-22, -17.5, 0.5)
            threshold_array = threshold_array[::-1]

            threshold_index = np.where(threshold_array == threshold)[0]

            if len(threshold_index) == 0:
                msg = ("\nInput luminosity threshold "
                    "does not match any of the Table 1 values \nof "
                    "Zheng et al. 2007 (arXiv:0703457).\n"
                    "Choosing the best-fit parameters "
                    "associated the default_luminosity_threshold variable \n"
                    "set in the model_defaults module.\n"
                    "You can always manually change the values in ``param_dict``.\n")
                warnings.warn(msg)
                threshold = model_defaults.default_luminosity_threshold
                threshold_index = np.where(threshold_array == threshold)[0]
                warnings.warn(msg)

            mass_param_dict = (
                {'logM0': logM0_array[threshold_index[0]],
                'logM1': logM1_array[threshold_index[0]],
                'alpha': alpha_array[threshold_index[0]]}
                )
            return mass_param_dict

        if publication in ['zheng07', 'Zheng07', 'Zheng_etal07', 'zheng_etal07', 'zheng2007', 'Zheng2007']:
            param_dict = get_zheng07_params(threshold)
            return param_dict
        else:
            raise KeyError("For Zheng07Sats, only supported best-fit models are currently Zheng et al. 2007")

cens_occ_model = mod_zheng07Cens(prim_haloprop_key = 'halo_mvir', 
                            sec_haloprop_key = 'delta',threshold=-21)
cens_prof_model = TrivialPhaseSpace()

sats_occ_model =  mod_zheng07Sats(prim_haloprop_key = 'halo_mvir', 
                                sec_haloprop_key = 'delta',modulate_with_cenocc=True,
                                  threshold=-21)
sats_prof_model = NFWPhaseSpace()

model_instance = HodModelFactory(centrals_occupation = cens_occ_model,
                                     centrals_profile = cens_prof_model,
                                     satellites_occupation = sats_occ_model,
                                     satellites_profile = sats_prof_model)
model_instance.populate_mock(halocat)
halotab = TabCorr.read('smdpl_pseudo.hdf5')
#@profile
def _get_lnlike(theta):
    a,logMmin,sigma_logM,alpha,logM0, logM1 = theta

    model_instance.param_dict['a'] = a
    model_instance.param_dict['logMmin'] = logMmin
    model_instance.param_dict['sigma_logM'] = sigma_logM
    model_instance.param_dict['alpha'] = alpha
    model_instance.param_dict['logM0'] = logM0
    model_instance.param_dict['logM1'] = logM1

    ngal, wp_calc = halotab.predict(model_instance)
    #est_ngals = model_instance.mock.estimate_ngals()
    #maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    #print(maxrss)
    #if est_ngals/(400**3) > 2*ng:
    #    return -np.inf

    #model_instance.mock.populate()
    #maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    #print(maxrss)
    #Lbox = 400.
    #pos = return_xyz_formatted_array(model_instance.mock.galaxy_table['x'],
    #                            model_instance.mock.galaxy_table['y'],
    #                            model_instance.mock.galaxy_table['z'],
    #                            period = Lbox)
    #x = pos[:,0]
    #y = pos[:,1]
    #z = pos[:,2]
    #velz = model_instance.mock.galaxy_table['vz']
    #pos_zdist = return_xyz_formatted_array(x,y,z,period=Lbox,
    #                velocity=velz,velocity_distortion_dimension='z')

    #pi_max = 60.
    #nthreads = 1

    #wp_calc = wp(Lbox,pi_max,nthreads,bin_edges,pos_zdist[:,0],
    #                    pos_zdist[:,1],pos_zdist[:,2],verbose=False)

    wp_diff = wp_vals-wp_calc
    ng_diff = ng-ngal


    gc.collect()
    
    return -0.5*np.dot(wp_diff, np.dot(invcov, wp_diff)) + -0.5*(ng_diff**2)/(ng_err**2)

def _get_lnprior(theta):
    a,logMmin,sigma_logM,alpha,logM0,logM1 = theta
    if -1.0< a <1.0 and 10.0 < logMmin < 15.0 and 0.0 < sigma_logM < 3.0 and 0.0 < alpha < 4.0 and 10.0 < logM0 < 15.0 and 10.0 < logM1 < 15.0:
        return 0.0
    return -np.inf

def _get_lnprob(theta):
    lp = _get_lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + _get_lnlike(theta)

ndim, nwalkers = 6, 35
nsteps = 50000
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
