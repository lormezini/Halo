import numpy as np
from halotools.sim_manager import CachedHaloCatalog
from halotools.mock_observables import wp
from halotools.empirical_models import PrebuiltHodModelFactory,  TrivialPhaseSpace,NFWPhaseSpace, HodModelFactory
from halotools.mock_observables import return_xyz_formatted_array
from astropy.utils.misc import NumpyRNGContext
from scipy.interpolate import interp1d
from scipy.special import erf,pdtrik
from halotools.utils.array_utils import custom_len
import warnings
from astropy.cosmology import FlatLambdaCDM
from tabcorr import TabCorr
from scipy.interpolate import interp1d

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


cosmo = FlatLambdaCDM(H0=67.77, Om0=0.307115) # smdpl
rp_bins = np.array((0.13159712, 0.20840302, 0.33003624, 0.52265998, 0.82770744, 1.31079410, 2.07583147,3.28737848, 5.20603787, 8.24451170, 13.05637315, 20.67664966))
pi_max =  60.
halotab = TabCorr.tabulate(halocat, 
                 wp, rp_bins, pi_max,
                 cosmology = cosmo,
                 prim_haloprop_key = 'halo_mvir', sec_haloprop_key_2 = 'delta', 
                 prim_haloprop_bins = 100,
                 sec_haloprop_percentile_bins = None, project_xyz = True,
                           sats_per_prim_haloprop=3e-12,verbose=True,a=a)

halotab.write('smdpl_pseudo.hdf5'.format(a))
