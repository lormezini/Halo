#!/usr/bin/env python
# coding: utf-8

# In[2]:


from halotools.sim_manager import CachedHaloCatalog, FakeSim
from halotools.empirical_models import PrebuiltHodModelFactory, Zheng07Cens, Zheng07Sats, TrivialPhaseSpace, NFWPhaseSpace, HodModelFactory
from halotools.mock_observables import return_xyz_formatted_array,wp
from halotools import utils
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool, cpu_count
import emcee
#import corner
from Corrfunc.theory.wp import wp
import MCMC_data_file
from numpy.linalg import inv
import scipy.optimize as op
from scipy.stats import chi2
import scipy.stats as stats
import random
import warnings
warnings.filterwarnings("ignore")
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')
from scipy.special import gamma
from scipy.stats import chisquare


# In[3]:


from astropy.table import Table 

__all__ = ('add_new_table_column', )

def add_new_table_column(table, new_colname, new_coltype, grouping_key,  
    aggregation_function, colnames_needed_by_function, 
    sorting_keys = None, table_is_already_sorted = False):

    assert type(table) == Table
    assert new_colname not in table.keys()
    assert grouping_key in table.keys()
    assert callable(aggregation_function) is True
    _ = iter(colnames_needed_by_function)
    for colname in colnames_needed_by_function:
        assert colname in table.keys()

    if sorting_keys == None:
        sorting_keys = [grouping_key]

    _ = iter(sorting_keys)
    for colname in sorting_keys:
        assert colname in table.keys()
    else:
        assert sorting_keys[0] == grouping_key
        
    if table_is_already_sorted is False:
        table.sort(sorting_keys)

    group_ids_data, idx_groups_data, group_richness_data = np.unique(
        table[grouping_key].data, 
        return_index = True, return_counts = True)

    dt = np.dtype([(new_colname, new_coltype)])
    
    result = np.zeros(len(table), dtype=dt[new_colname])

    func_arglist = [table[key].data for key in colnames_needed_by_function]

    for igroup, host_halo_id in enumerate(group_ids_data):
        first_igroup_idx = idx_groups_data[igroup]
        last_igroup_idx = first_igroup_idx + group_richness_data[igroup]
        group_data_list = [arg[first_igroup_idx:last_igroup_idx] for arg in func_arglist]
        result[first_igroup_idx:last_igroup_idx] = aggregation_function(*group_data_list)

    table[new_colname] = result


# In[4]:


threshold = 20
dname = "zehavi_data_file_20"
param = "mvir"


# In[5]:


if '21' in dname:
    wp_ng_vals = zehavi_data_file_21.get_wp()[0:12]
    bin_edges = zehavi_data_file_21.get_bins()[0:12]
    cov_matrix = zehavi_data_file_21.get_cov()[0:11,0:11]
    err = np.array([cov_matrix[i,i] for i in range(len(cov_matrix))])
    bin_cen = (bin_edges[1:]+bin_edges[:-1])/2.
if '19' in dname:
    print('19')
    import zehavi_data_file_19
    wp_ng_vals = zehavi_data_file_19.get_wp()[0:12]
    bin_edges = zehavi_data_file_19.get_bins()[0:12]
    cov_matrix = zehavi_data_file_19.get_cov()[0:11,0:11]
    err = np.array([cov_matrix[i,i] for i in range(len(cov_matrix))])    
    bin_cen = (bin_edges[1:]+bin_edges[:-1])/2.
if '20' in dname:
    print('20')
    import zehavi_data_file_20
    wp_ng_vals = zehavi_data_file_20.get_wp()[0:12]
    bin_edges = zehavi_data_file_20.get_bins()[0:12]
    cov_matrix = zehavi_data_file_20.get_cov()[0:11,0:11]
    err = np.array([cov_matrix[i,i] for i in range(len(cov_matrix))])    
    bin_cen = (bin_edges[1:]+bin_edges[:-1])/2.


# In[6]:


def calc_chi2(obs,exp,var):
    chi2s = []
    diff = obs-exp
    chi2 = np.sum(diff**2/var)
    chi2s.append(round(chi2,7))
        
    return chi2s


# In[10]:


for i in range(90,96):
    a = i*0.01
    print(a)
    new_colname = 'combo'
    new_coltype = 'f4'
    grouping_key = 'halo_hostid'
    def lin_combo_mass_c(mass, c):
        return 10**(a*np.log10(mass)+(1-a)*np.log10(c))
    aggregation_function = lin_combo_mass_c
    colnames_needed_by_function = ['halo_mvir','halo_nfw_conc']
    sorting_keys = ['halo_hostid', 'halo_upid']
    
    halocat = CachedHaloCatalog(fname='/Users/lmezini/.astropy/cache/halotools/halo_catalogs/SMDPL/rockstar/2019-07-03-18-38-02-9731.dat.my_cosmosim_halos.hdf5')
    add_new_table_column(halocat.halo_table, new_colname, new_coltype, grouping_key, 
                     aggregation_function, colnames_needed_by_function, 
                     sorting_keys=sorting_keys)

    cens_occ_model = Zheng07Cens(prim_haloprop_key = 'combo',threshold=threshold)
    cens_prof_model = TrivialPhaseSpace()

    sats_occ_model =  Zheng07Sats(prim_haloprop_key = 'combo', modulate_with_cenocc=True,threshold=threshold)
    sats_prof_model = NFWPhaseSpace()
    
    halocat.redshift = 0.
    pi_max = 60.
    Lbox = 400.

    model_instance = HodModelFactory(centrals_occupation = cens_occ_model, centrals_profile = cens_prof_model, 
                                 satellites_occupation = sats_occ_model, satellites_profile = sats_prof_model)
    model_instance.param_dict['logMmin'] = 11.96
    model_instance.param_dict['sigma_logM'] = 0.38
    model_instance.param_dict['alpha'] = 1.16
    model_instance.param_dict['logM0'] = 13.28-1.7
    model_instance.param_dict['logM1'] = 13.28

    try:
        model_instance.mock.populate()
    except:
        model_instance.populate_mock(halocat)
        
    halo_table = model_instance.mock.halo_table
    pos = return_xyz_formatted_array(model_instance.mock.galaxy_table['x'], model_instance.mock.galaxy_table['y'],
                                 model_instance.mock.galaxy_table['z'],period = Lbox)
    x = pos[:,0]
    y = pos[:,1]
    z = pos[:,2]
    velz = model_instance.mock.galaxy_table['vz']
    pos_zdist = return_xyz_formatted_array(x,y,z, period = Lbox, velocity=velz, velocity_distortion_dimension='z')
    mod = wp(Lbox,pi_max,1,bin_edges,pos_zdist[:,0],pos_zdist[:,1],pos_zdist[:,2],
                verbose=True)
    print(calc_chi2(mod['wp'],wp_ng_vals[1:len(wp_ng_vals)],err))
    plt.errorbar(bin_cen,wp_ng_vals[1:len(wp_ng_vals)],yerr=np.sqrt(err),fmt='o',markersize=2,capsize=4,label='data')
    plt.plot(bin_cen,mod['wp'],markersize=2,label=str(a))

    #plt.plot(bin_cen,oldfunc,label='Old Corrfunc')
    #plt.title("")
    plt.legend()
    plt.ylabel('wp')
    plt.xlabel('rp')
    plt.tick_params(right=True)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid()
    plt.show()


# In[5]:


halocat = CachedHaloCatalog(fname='/Users/lmezini/.astropy/cache/halotools/halo_catalogs/SMDPL/rockstar/2019-07-03-18-38-02-9731.dat.my_cosmosim_halos.hdf5')


# In[8]:


plt.clf()
m = halocat.halo_table['halo_mvir']
c = halocat.halo_table['halo_nfw_conc']
x = np.logspace(np.log10(min(m)),np.log10(max(m)),100)
plt.scatter(m,c)
plt.plot(x,a+b*np.log10(x),c = 'r')
plt.xscale('log')
plt.yscale('log')


# In[7]:


m = halocat.halo_table['halo_mvir']
c = halocat.halo_table['halo_nfw_conc']
a,b= np.polyfit(np.log10(m), np.log10(c),1)
a,b


# In[13]:


plt.plot(m,0.5*np.log10(m)+(1-0.5)*0.5*m+0.05)
plt.yscale('log')
plt.xscale('log')
plt.xlabel('mass')


# In[50]:


min(0.8*np.log10(m)+(1-0.8)*np.log10(c))


# In[ ]:




