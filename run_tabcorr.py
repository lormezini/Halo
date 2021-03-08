import numpy as np
from matplotlib import cm
from matplotlib import colors
import matplotlib.pyplot as plt
from halotools.sim_manager import CachedHaloCatalog
from halotools.mock_observables import wp
from halotools.empirical_models import PrebuiltHodModelFactory,  Zheng07Cens, Zheng07Sats, TrivialPhaseSpace,NFWPhaseSpace, HodModelFactory
from halotools.mock_observables import return_xyz_formatted_array
from astropy.cosmology import FlatLambdaCDM
from tabcorr import TabCorr


halocat = CachedHaloCatalog(fname = '//Users/lmezini/Downloads/hlist_1.00231.list.halotools_v0p1.hdf5',update_cached_fname=True)

#halocat = CachedHaloCatalog(fname='/home/lom31/.astropy/cache/halotools/halo_catalogs/smdpl/rockstar/2019-07-03-18-38-02-9731.dat.my_cosmosim_halos.hdf5',update_cached_fname = True)
cosmo = FlatLambdaCDM(H0=67.77, Om0=0.307115) # smdpl
rp_bins = np.array((0.13159712, 0.20840302, 0.33003624, 0.52265998, 0.82770744, 1.31079410, 2.07583147,3.28737848, 5.20603787, 8.24451170, 13.05637315, 20.67664966))
pi_max =  60.
halotab = TabCorr.tabulate(halocat, 
                 wp, rp_bins, pi_max,
                 cosmology = cosmo,
                 prim_haloprop_key = 'halo_vmax', prim_haloprop_bins = 100,
                 sec_haloprop_percentile_bins = None, project_xyz = True,
                 sats_per_prim_haloprop=3e-12,verbose=True,Num_ptcl_requirement=0.5000055455806451)

halotab.write('smdpl_vmax.hdf5')
