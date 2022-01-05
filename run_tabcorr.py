import numpy as np
from halotools.sim_manager import CachedHaloCatalog
from halotools.mock_observables import wp
from halotools.empirical_models import PrebuiltHodModelFactory,  Zheng07Cens, Zheng07Sats, TrivialPhaseSpace,NFWPhaseSpace, HodModelFactory
from halotools.mock_observables import return_xyz_formatted_array
from astropy.cosmology import FlatLambdaCDM
from tabcorr import TabCorr
from scipy.interpolate import interp1d

import argparse
parser = argparse.ArgumentParser(description='A values')
parser.add_argument('--a',type=int, help='Composite parameter, a, value')
args = parser.parse_args()
a = args.a

halocat = CachedHaloCatalog(fname = 'smdpl.dat.smdpl2.hdf5',update_cached_fname=True)

ht = halocat.halo_table
#m,b = (3.1069839419403174, 5.015822745222037)
#vmax_ml = 10**(np.log10(ht['halo_vmax'])*m+b)
#ht.add_column(vmax_ml,name = 'vmax_ml')

#hmvir = ht['halo_mvir']
#hvmax = ht['vmax_ml']

#a = a/100.0
#halocat.halo_table.add_column((hmvir**a)*(hvmax**(1-a)),name='combo')

G =  4.302*(10**-9.)
mvir = ht['halo_mvir']
rvir = ht['halo_rvir']
vmax = ht['halo_vmax']
vvir = np.sqrt((G*mvir)/rvir)

cnfw = np.linspace(1, 1400,140000)
vmax_interp = np.sqrt(0.2162166 * 1.0/( (np.log(1+cnfw)/cnfw) - (1/(1+cnfw)) ) )
get_c = interp1d(vmax_interp,cnfw,fill_value='extrapolate')
del cnfw
c = get_c(vmax/vvir)

num_bins = abs(int((np.min(np.log10(mvir))-np.max(np.log10(mvir)))*10))
bins=np.logspace(np.min(np.log10(mvir)),np.max(np.log10(mvir)),num=num_bins)

delta = []
mass = []
for i in range(1,num_bins):
    mask = np.logical_and(mvir>bins[i-1],mvir<bins[i])
    delta.append((c[mask]-np.mean(c[mask]))/np.std(c[mask]))
    mass.append(mvir[mask])
mass = np.concatenate(mass)
delta = np.concatenate(delta)[np.argsort(mass)[::-1]]
mass = mass[np.argsort(mass)[::-1]]

a = a/100.0
ht['pseudo'] = mass*10**(a*delta)

cosmo = FlatLambdaCDM(H0=67.77, Om0=0.307115) # smdpl
rp_bins = np.array((0.13159712, 0.20840302, 0.33003624, 0.52265998, 0.82770744, 1.31079410, 2.07583147,3.28737848, 5.20603787, 8.24451170, 13.05637315, 20.67664966))
pi_max =  60.
halotab = TabCorr.tabulate(halocat, 
                 wp, rp_bins, pi_max,
                 cosmology = cosmo,
                 prim_haloprop_key = 'pseudo', prim_haloprop_bins = 100,
                 sec_haloprop_percentile_bins = None, project_xyz = True,
                           sats_per_prim_haloprop=3e-12,verbose=True,a=a)

halotab.write('smdpl_pseudo_a{}.hdf5'.format(a))
