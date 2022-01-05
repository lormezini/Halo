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

class mod_zheng07Cens(OccupationComponent):
    
    def __init__(self,
            threshold=20,
            prim_haloprop_key="halo_mvir",
            sec_haloprop_key = "halo_vmax"
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
            prim_haloprop_key="halo_mvir", sec_haloprop_key = "halo_vmax",
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
