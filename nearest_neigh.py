from halotools.sim_manager import CachedHaloCatalog, FakeSim
from halotools.empirical_models import PrebuiltHodModelFactory, Zheng07Cens, Zheng07Sats, TrivialPhaseSpace, NFWPhaseSpace, HodModelFactory
from halotools.mock_observables import return_xyz_formatted_array,wp
import numpy as np
import time
from multiprocessing import Pool, cpu_count
import emcee
from Corrfunc.theory.wp import wp
from numpy.linalg import inv
import scipy.optimize as op
from scipy.stats import chi2
import scipy.stats as stats
import random
import warnings
warnings.filterwarnings("ignore")
from scipy.special import gamma
from scipy.stats import chisquare, gaussian_kde
from sklearn.neighbors import NearestNeighbors, KDTree
import gc
from tabcorr import TabCorr

fname = "/home/lom31/emcee_fits_tabcorr/mar_2021/zehavi_smdpl_vmax_m19_tabcorr_2.h5"
dname ='zehavi_data_file_19'
param = 'vmax_ml'
threshold = -19
out_file = "best_fit_lnprob_{}_m{}.npz".format(param,threshold)

if '21' in dname:
    import zehavi_data_file_21
    wp_ng_vals = zehavi_data_file_21.get_wp()[0:12]
    bin_edges = zehavi_data_file_21.get_bins()[0:12]
    cov_matrix = zehavi_data_file_21.get_cov()[0:11,0:11]
    err = np.array([cov_matrix[i,i] for i in range(len(cov_matrix))])
    bin_cen = (bin_edges[1:]+bin_edges[:-1])/2.
    invcov = inv(cov_matrix)
    ng_cov = 0.000005
if '19' in dname:
    print('19')
    import zehavi_data_file_19
    wp_ng_vals = zehavi_data_file_19.get_wp()[0:12]
    bin_edges = zehavi_data_file_19.get_bins()[0:12]
    cov_matrix = zehavi_data_file_19.get_cov()[0:11,0:11]
    err = np.array([cov_matrix[i,i] for i in range(len(cov_matrix))])    
    bin_cen = (bin_edges[1:]+bin_edges[:-1])/2.
    invcov = inv(cov_matrix)
    ng_cov = 0.000068
if '20' in dname:
    print('20')
    import zehavi_data_file_20
    wp_ng_vals = zehavi_data_file_20.get_wp()[0:12]
    bin_edges = zehavi_data_file_20.get_bins()[0:12]
    cov_matrix = zehavi_data_file_20.get_cov()[0:11,0:11]
    err = np.array([cov_matrix[i,i] for i in range(len(cov_matrix))])    
    bin_cen = (bin_edges[1:]+bin_edges[:-1])/2.
    invcov = inv(cov_matrix)
    ng_cov = 0.00007
if 'mock' in dname:
    wp_ng_vals = mock_data_2.get_wp()
    bin_edges = mock_data_2.get_bin_edges()
    cov_matrix = mock_data_2.get_cov()
    err = np.array([cov_matrix[i,i] for i in range(len(cov_matrix))])
    bin_cen = (bin_edges[1:]+bin_edges[:-1])/2.


files = [fname]
s = []
log_prob_s = []
wps = []
for f in files: 
    reader = emcee.backends.HDFBackend(f, read_only=True)
    s.append(reader.get_chain(discard=1000, flat=False, thin=1))
    log_prob_s.append(reader.get_log_prob(discard=1000, flat=False, thin=1))
    wps.append(reader.get_blobs(discard=1000))

m,b = (0.3260532250619037, -1.6354239489541218)
if param == 'vmax_ml':
    s[0][:,:,0]=s[0][:,:,0]*(m)+(b)
    s[0][:,:,3]=s[0][:,:,3]*(m)+(b)
    s[0][:,:,4]=s[0][:,:,4]*(m)+(b)
    s[0][:,:,1]=s[0][:,:,1]*m #sigma
    s[0][:,:,2]=s[0][:,:,2]/m #alpha

if len(files)>1:
    print('greater')
    samples = s[0]
    log_prob_samples = log_prob_s[0]
    wp_samples = wps[0]
    for i in range(len(files)):
        if i+1 < len(s):
            samples = np.concatenate((samples,s[i+1]))
            log_prob_samples = np.concatenate((log_prob_samples,
                                               log_prob_s[i+1]))
            wp_samples = np.concatenate((wp_samples,wps[i+1]))
else:
    samples = s[0]
    log_prob_samples = log_prob_s[0]
    wp_samples = wps[0]

# # Nearest Neighbor########
#downsample, 20 chains, every 50th step
#reshape so that it is nchains*length x nparams
dsamples = samples[:,:20,:][::50,:]
dsamples=samples[:,:20,:][::50,:].reshape(dsamples.shape[0]*dsamples.shape[1],5)


####Prepare data for nearest neighbor search
print('Median values:')
print(' logMvirmin, ${\sigma}logMvir, $alpha$, $logMvir_0$, $logMvir1$]')
params_median = np.median(dsamples, axis=0)
params_std = np.std(dsamples, axis=0)
print('  ', params_median[[0, 1, 2, 3, 4]])
print('+-', params_std[[0, 1, 2, 3, 4]])
print()

# Find the best-fit parameters in segments of the chain
params_1_sigma_range = (np.percentile(dsamples, 84, axis=0) - np.percentile(dsamples, 16, axis=0))/2

print('1-sigma range:')
print(' logMvirmin, ${\sigma}logMvir, $alpha$, $logMvir_0$, $logMvir1$]')
print('  ', params_1_sigma_range[[0, 1, 2, 3, 4]])


data = dsamples[:, :6]

# Reflect against hard prior boundaries
# sigma_logM
# determine this from corner plots and see which parameters are cut off by prior bounds
"""
data_new = np.copy(data)
data_new[:, 1] = -data_new[:, 1]
data = np.concatenate([data, data_new], axis=0)
"""
data_new = np.copy(data)
data_new[:, 3] = 1.0 - (data_new[:, 3]-1.0)
data = np.concatenate([data, data_new], axis=0)

# rescaling the dimensions
data_rescale = data/params_1_sigma_range[None, :6]

# Find the distance to the n-th nearest neighbor
neigh = NearestNeighbors(n_neighbors=501, radius=np.inf, algorithm='kd_tree')
neigh.fit(data_rescale)
dist, _ = neigh.kneighbors(data_rescale, return_distance=True)

# Remove the reflected points
maxdist = np.max(dist, axis=1)
idx_best = np.argsort(maxdist)
mask = idx_best < len(dsamples)
print("orig npoints/including reflected npoints: ",np.sum(mask)/len(idx_best))
idx_best = idx_best[mask]

# select the top n_best points
n_best = 500
idx_best = idx_best[:n_best]
gc.collect()

#plt.hist(maxdist, 100, range=(np.min(maxdist)-0.2, np.percentile(maxdist, 99)))
#plt.hist(maxdist[idx_best], 100, range=(np.min(maxdist)-0.2, np.percentile(maxdist, 99)))
#plt.xlabel('distance')
#plt.show()

#ndim=5
#fig = corner.corner(dsamples[idx_best].reshape((-1,ndim)),
#        labels=["$logVmaxMin$", "${\sigma}logVmax$", "$alpha$", "$logVmax_0$", "$logVmax$"],
#        show_titles=True,title_kwargs={"fontsize": 10},levels=(0.68, 0.95,),plot_datapoints=False,
#        fill_contours=False,plot_density=True,quantiles=(0.16, 0.84), color='b',bins=15)#, levels=(1-np.exp(-0.5),))


#fig = corner.corner(dsamples.reshape((-1,ndim)),
#        labels=["$logVmaxMin$", "${\sigma}logVmax$", "$alpha$", "$logVmax_0$", "$logVmax$"],
#        show_titles=True,title_kwargs={"fontsize": 10},quantiles=(0.16, 0.84),bins=20)#, levels=(1-np.exp(-0.5),))


# Find average Chi2

if param == 'mvir':
    print('mvir')
    cens_occ_model = Zheng07Cens(threshold=threshold)
    cens_prof_model = TrivialPhaseSpace()

    sats_occ_model =  Zheng07Sats(modulate_with_cenocc=True,threshold=threshold)
    sats_prof_model = NFWPhaseSpace()
    halotab = TabCorr.read('smdpl_halo_mvir.hdf5')

elif param == 'vmax':
    print('vmax')
    cens_occ_model = Zheng07Cens(prim_haloprop_key = 'halo_vmax',threshold=threshold)
    cens_prof_model = TrivialPhaseSpace()

    sats_occ_model =  Zheng07Sats(prim_haloprop_key = 'halo_vmax', modulate_with_cenocc=True,threshold=threshold)
    sats_prof_model = NFWPhaseSpace()

elif param == 'vmax_ml':
    cens_occ_model = Zheng07Cens(prim_haloprop_key = 'vmax_ml',
                                 threshold=threshold)
    cens_prof_model = TrivialPhaseSpace()
    sats_occ_model =  Zheng07Sats(prim_haloprop_key = 'vmax_ml', 
                                  modulate_with_cenocc=True,
                                  threshold=threshold)
    sats_prof_model = NFWPhaseSpace()
    halotab = TabCorr.read('smdpl_vmax_ml.hdf5')
    
#halocat = CachedHaloCatalog(simname='bolshoi',redshift = 0.0)
#halocat = CachedHaloCatalog(fname = '/Users/lmezini/.astropy/cache/halotools/halo_catalogs/bolplanck/rockstar/hlist_1.00231.list.halotools_v0p4.hdf5',update_cached_fname = True)
#halocat = CachedHaloCatalog(fname = '/Users/lmezini/Downloads/hlist_1.00231.list.halotools_v0p1.hdf5',update_cached_fname = True)
halocat = CachedHaloCatalog(fname='/home/lom31/home/Halo/smdpl.dat.smdpl2.hdf5',update_cached_fname = True)
if param == 'vmax_ml':
    ht = halocat.halo_table
    vmax_ml = 10**(np.log10(ht['halo_vmax'])*3.1069839419403174+5.015822745222037)
    ht.add_column(vmax_ml, name='vmax_ml')
halocat.redshift = 0.
pi_max = 60.
Lbox = 400.

model_instance = HodModelFactory(centrals_occupation = cens_occ_model,
                                 centrals_profile = cens_prof_model, 
                                 satellites_occupation = sats_occ_model,
                                 satellites_profile = sats_prof_model)
model_instance.populate_mock(halocat)
nthreads = 4

# number of repeats
nrepeat = 100

lnlike_all = np.zeros((len(idx_best), nrepeat))

time_last = time.time()
for index in range(len(idx_best)):
    
    model_instance.param_dict['logMmin'] = dsamples[idx_best[index]][0]
    model_instance.param_dict['sigma_logM'] = dsamples[idx_best[index]][1]
    model_instance.param_dict['alpha'] = dsamples[idx_best[index]][2]
    model_instance.param_dict['logM0'] = dsamples[idx_best[index]][3]
    model_instance.param_dict['logM1'] = dsamples[idx_best[index]][4]

    # Repeatedly populate galaxies and average lnL
    for repeat_index in range(nrepeat):
        gc.collect()
        """
        try:
            model_instance.mock.populate()
        except:
            model_instance.populate_mock(halocat)

        number_gal = len(model_instance.mock.galaxy_table)
        
        pos = return_xyz_formatted_array(model_instance.mock.galaxy_table['x'], 
                                        model_instance.mock.galaxy_table['y'],
                                        model_instance.mock.galaxy_table['z'],
                                        period = Lbox)
        x = pos[:,0]
        y = pos[:,1]
        z = pos[:,2]
        velz = model_instance.mock.galaxy_table['vz']
        pos_zdist = return_xyz_formatted_array(x,y,z, period = Lbox, 
                        velocity=velz, velocity_distortion_dimension='z')

        mod = wp(Lbox,pi_max,1,bin_edges,pos_zdist[:,0],pos_zdist[:,1],pos_zdist[:,2],
                    verbose=True)
        
        number_gal = len(model_instance.mock.galaxy_table)
        """

        ngal, model_wp = halotab.predict(model_instance)

        # log likelihood
        wp_dev = model_wp - wp_ng_vals[1:len(wp_ng_vals)]
        wplike = -0.5*np.dot(np.dot(wp_dev, invcov), wp_dev)

        # log likelihood from number density
        #ngal = number_gal/(model_instance.mock.Lbox[0]**3)
        #ng_theory_error = ngal/np.sqrt(number_gal)
        nglike = -0.5*((ngal-wp_ng_vals[0])**2/(ng_cov**2))#+ng_theory_error**2))

        lnlike_all[index, repeat_index] = np.exp(wplike + nglike)

        #####################################################################
    
    time_now = time.time()
    print('{}  {:.0f} sec'.format(index, time_now-time_last))
    time_last = time_now

np.savez_compressed(out_file, idx=idx_best, lnlike_all=lnlike_all)


results = np.load(out_file)
lnlike_all = results['lnlike_all']


def hlmean(data, multiplier=None, verbose=True):
    ndata = len(data)
    if ndata==0 and verbose:
        print('H-L mean: empty array!!!')
    if ndata < 200:
        pairmean = np.zeros(int(ndata*(ndata+1)/2))
        index = 0
        for i in range(ndata):
            for j in range(i,ndata):
                pairmean[index] = (data[i]+data[j])/2
                index += 1
    else:
        if multiplier==None:
            nsamp = 200 * ndata
        else:
            nsamp = multiplier * ndata
        idx = np.floor(np.random.rand(nsamp,2)*ndata)
        idx = idx.astype(np.int64,copy=False)
        pairmean = np.sum(data[idx],axis=1)/2.
    return(np.median(pairmean))

#lnlike_hlmean = np.array([hlmean(lnlike_all[index]) for index in range(len(lnlike_all))])
lnlike_mean = np.mean(lnlike_all, axis=1)
idx_single_best = idx_best[np.argsort(lnlike_mean)[-1]]


#plt.figure(figsize=(5, 5))
#plt.errorbar(np.arange(20), lnlike_hlmean[np.argsort(lnlike_hlmean)[-20:]], 
#             yerr=np.std(lnlike_all[np.argsort(lnlike_hlmean)[-20:]], axis=1)/np.sqrt(nrepeat), markersize=10, fmt='x', marker='.')
#plt.xlabel('hlmean')
# plt.ylabel('mean')
#plt.show()

params_best = dsamples[idx_single_best]
params_best_plus = np.percentile(dsamples, 84, axis=0) - params_best
params_best_minus = params_best - np.percentile(dsamples, 16, axis=0)
print('Best-fit values:')
print('  [alpha  logM1  sigma_logM  logM0  logMmin  pzerr_rescale  bias]')
print('  ', params_best[[0, 1, 2, 3, 4]])
print(' +', params_best_plus[[0, 1, 2, 3, 4]])
print(' -', params_best_minus[[0, 1, 2, 3, 4]])


# Top 10 points
idx_best_10 = idx_best[np.argsort(lnlike_mean)[-10:]]
nthreads = 4

# number of repeats
nrepeat = 1000

lnlike_all = np.zeros((len(idx_best_10), nrepeat))

time_last = time.time()
for index in range(len(idx_best_10)):
    model_instance.param_dict['logMmin'] = dsamples[idx_best[index]][0]
    model_instance.param_dict['sigma_logM'] = dsamples[idx_best[index]][1]
    model_instance.param_dict['alpha'] = dsamples[idx_best[index]][2]
    model_instance.param_dict['logM0'] = dsamples[idx_best[index]][3]
    model_instance.param_dict['logM1'] = dsamples[idx_best[index]][4]

    # Repeatedly populate galaxies and average lnL
    for repeat_index in range(nrepeat):
        gc.collect()

        try:
            model_instance.mock.populate()
        except:
            model_instance.populate_mock(halocat)

        """
        number_gal = len(model_instance.mock.galaxy_table)
        
        pos = return_xyz_formatted_array(model_instance.mock.galaxy_table['x'], 
                                        model_instance.mock.galaxy_table['y'],
                                        model_instance.mock.galaxy_table['z'],
                                         period = Lbox)
        x = pos[:,0]
        y = pos[:,1]
        z = pos[:,2]
        velz = model_instance.mock.galaxy_table['vz']
        pos_zdist = return_xyz_formatted_array(x,y,z, period = Lbox, 
                        velocity=velz, velocity_distortion_dimension='z')

        mod = wp(Lbox,pi_max,1,bin_edges,pos_zdist[:,0],pos_zdist[:,1],pos_zdist[:,2],
                    verbose=True)
        model_wp = mod['wp']
        # log likelihood from wprp
        wp_dev = model_wp - wp_ng_vals[1:len(wp_ng_vals)]
        wplike = -0.5*np.dot(np.dot(wp_dev, invcov), wp_dev)

        # log likelihood from number density
        ngal = number_gal/(model_instance.mock.Lbox[0]**3)
        ng_theory_error = ngal/np.sqrt(number_gal)
        nglike = -0.5*((ngal-wp_ng_vals[0])**2/(ng_cov**2))#+ng_theory_error**2))

        lnlike_all[index, repeat_index] = np.exp(wplike + nglike)
        """
        ngal, model_wp = halotab.predict(model_instance)

        # log likelihood
        wp_dev = model_wp - wp_ng_vals[1:len(wp_ng_vals)]
        wplike = -0.5*np.dot(np.dot(wp_dev, invcov), wp_dev)

        nglike = -0.5*((ngal-wp_ng_vals[0])**2/(ng_cov**2))
        lnlike_all[index, repeat_index] = np.exp(wplike + nglike)

        #####################################################################
    
    time_now = time.time()
    print('{}  {:.0f} sec'.format(index, time_now-time_last))
    time_last = time_now

np.savez_compressed('top10_'+out_file, 
                    idx=idx_best_10, lnlike_all=lnlike_all)


results = np.load('top10_'+out_file)
lnlike_all = results['lnlike_all']

np.random.seed(53)
#lnlike_hlmean = np.array([hlmean(lnlike_all[index], multiplier=50000) for index in range(len(lnlike_all))])
lnlike_mean = np.mean(lnlike_all, axis=1)

idx_single_best = idx_best_10[np.argsort(lnlike_mean)[-1]]

#plt.figure(figsize=(5, 5))
#plt.errorbar(np.arange(len(lnlike_hlmean)), lnlike_hlmean[np.argsort(lnlike_hlmean)], 
#             yerr=np.std(lnlike_all[np.argsort(lnlike_hlmean)], axis=1)/np.sqrt(nrepeat), 
#             markersize=10, fmt='x', marker='.')
# plt.xlabel('mean')
#plt.ylabel('hlmean')
#plt.show()


# In[35]:


params_best = dsamples[idx_single_best]
params_best_plus = np.percentile(dsamples, 84, axis=0) - params_best
params_best_minus = params_best - np.percentile(dsamples, 16, axis=0)
print('Best-fit values:')
print('  [alpha  logM1  sigma_logM  logM0  logMmin  pzerr_rescale  bias]')
print('  ', params_best[[0, 1, 2, 3, 4]])
print(' +', params_best_plus[[0, 1, 2, 3, 4]])
print(' -', params_best_minus[[0, 1, 2, 3, 4]])


# number of repeats
nrepeat = 100

# Initialize arrays
lnlike_all = np.zeros((1, nrepeat))
model_wp_all = np.zeros((1, nrepeat, len(bin_cen)))
number_gal_all = np.zeros((1, nrepeat))


nthreads = 4

for index in [0]:
    
    print(index)
    model_instance.param_dict['logMmin'] = dsamples[idx_best[index]][0]
    model_instance.param_dict['sigma_logM'] = dsamples[idx_best[index]][1]
    model_instance.param_dict['alpha'] = dsamples[idx_best[index]][2]
    model_instance.param_dict['logM0'] = dsamples[idx_best[index]][3]
    model_instance.param_dict['logM1'] = dsamples[idx_best[index]][4]

    # Repeatedly populate galaxies and average lnL
    for repeat_index in range(nrepeat):
        gc.collect()
        try:
            model_instance.mock.populate()
        except:
            model_instance.populate_mock(halocat)

        """
        number_gal = len(model_instance.mock.galaxy_table)
        
        pos = return_xyz_formatted_array(model_instance.mock.galaxy_table['x'], 
                                        model_instance.mock.galaxy_table['y'],
                                        model_instance.mock.galaxy_table['z'],
                                        period = Lbox)
        x = pos[:,0]
        y = pos[:,1]
        z = pos[:,2]
        velz = model_instance.mock.galaxy_table['vz']
        pos_zdist = return_xyz_formatted_array(x,y,z, period = Lbox, 
                        velocity=velz, velocity_distortion_dimension='z')

        mod = wp(Lbox,pi_max,1,bin_edges,pos_zdist[:,0],pos_zdist[:,1],pos_zdist[:,2],
                    verbose=True)
        model_wp = mod['wp']
        # log likelihood from wprp
        wp_dev = model_wp - wp_ng_vals[1:len(wp_ng_vals)]
        wplike = -0.5*np.dot(np.dot(wp_dev, invcov), wp_dev)

        # log likelihood from number density
        ngal = number_gal/(model_instance.mock.Lbox[0]**3)
        ng_theory_error = ngal/np.sqrt(number_gal)
        nglike = -0.5*((ngal-wp_ng_vals[0])**2/(ng_cov**2))#+ng_theory_error**2))

        lnlike_all[index, repeat_index] = np.exp(wplike + nglike)
        """
        ngal, model_wp = halotab.predict(model_instance)

        # log likelihood                                                        
        wp_dev = model_wp - wp_ng_vals[1:len(wp_ng_vals)]
        wplike = -0.5*np.dot(np.dot(wp_dev, invcov), wp_dev)
                     
        nglike = -0.5*((ngal-wp_ng_vals[0])**2/(ng_cov**2))
        lnlike_all[index, repeat_index] = np.exp(wplike + nglike)

        #####################################################################
    
    time_now = time.time()
    print('{}  {:.0f} sec'.format(index, time_now-time_last))
    time_last = time_now
    
np.savez_compressed('single_best_results_{}_m{}.npz'.format(param,threshold), 
                    lnlike_all=lnlike_all, bin_edges=bin_edges,
                    model_wp_all=model_wp_all, number_gal_all=number_gal_all)


#results = np.load('single_best_mock_results_{}_m{}.npz'.format(param,threshold))
#lnlike_all = results['lnlike_all']






