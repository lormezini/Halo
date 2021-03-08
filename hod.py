from halotools.sim_manager import CachedHaloCatalog, FakeSim
from halotools.empirical_models import PrebuiltHodModelFactory, Zheng07Cens, Zheng07Sats, TrivialPhaseSpace, NFWPhaseSpace, HodModelFactory
from halotools.mock_observables import return_xyz_formatted_array
import numpy as np
import astropy

halocat = CachedHaloCatalog(fname = '/Users/lmezini/.astropy/cache/halotools/halo_catalogs/bolplanck/rockstar/hlist_1.00231.list.halotools_v0p4.hdf5',update_cached_fname = True)
halocat.redshift = 0.

model = PrebuiltHodModelFactory('zheng07',threshold=-20,z=0)
model.param_dict['logMmin'] = 12.24
model.param_dict['sigma_logM'] = 0.84
model.param_dict['alpha'] = 1.05
model.param_dict['logM0'] = 12.19583311183421
model.param_dict['logM1'] = 13.19

def get_ngals():
    model.populate_mock(halocat = halocat)
    halo_table = model.mock.halo_table
    gal_table = model.mock.galaxy_table
    data=np.zeros(len(halo_table))
    for j in range(len(halo_table)):
        mask=gal_table['halo_hostid']==halo_table['halo_hostid'][j]
        data[j]+=len(gal_table[mask])
    col = astropy.table.Column(name='Ngals',data=data)
    halo_table.add_column(col,rename_duplicate=True)

    return halo_table

def mass_mask(m_arr, lower, upper):
    return np.logical_and(m_arr<upper, m_arr>=lower)

def bin(halo_table):
    mock_mean_ntot = np.zeros(200)
    mhist,mbins = np.histogram(halo_table['halo_mvir'],np.logspace(10, 15, 201))
    for i in range(200):
        mock_mean_ntot[i] = np.sum(halo_table['Ngals'][mass_mask(halo_table['halo_mvir'], mbins[i], mbins[i+1])])

    return mock_mean_ntot

ntot= np.zeros(200)

for i in range(30):
    halo_table=get_ngals()
    mock_mean_ntot=bin(halo_table)
    ntot+=mock_mean_ntot

np.save('hod.npy', ntot/50.)
