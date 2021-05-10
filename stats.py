from halotools.sim_manager import CachedHaloCatalog, FakeSim
from halotools.empirical_models import PrebuiltHodModelFactory, Zheng07Cens, Zheng07Sats, TrivialPhaseSpace, NFWPhaseSpace, HodModelFactory
from halotools.mock_observables import return_xyz_formatted_array
import numpy as np
import matplotlib.pyplot as plt
#import time
from multiprocessing import Pool, cpu_count
import emcee
import corner
from Corrfunc.theory.wp import wp
#import MCMC_data_file
from numpy.linalg import inv
#import scipy.optimize as op
from scipy.stats import chi2
import scipy.stats as stats
import random
import warnings
warnings.filterwarnings("ignore")
from scipy.special import gamma
from math import floor,ceil
from tabcorr import TabCorr


fname = 'zehavi_smdpl_vmax_m21_tabcorr_2.h5'
param ='vmax_ml'

def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i


def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf

# Automated windowing procedure following Sokal (1989)
def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

# Following the suggestion from Goodman & Weare (2010)
def autocorr_gw2010(y, c=5.0):
    f = autocorr_func_1d(np.mean(y, axis=0))
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]


def autocorr_new(y, c=5.0):
    f = np.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)
    f /= len(y)
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]

files = [fname]#,fname2]

#files = [fname4,fname5]
s = []
log_prob_s = []
wp = []

for f in files: 
    reader = emcee.backends.HDFBackend(f, read_only=True)
    s.append(reader.get_chain(discard=5000, flat=False, thin=1))
    log_prob_s.append(reader.get_log_prob(discard=5000, flat=False, thin=1))
    wp.append(reader.get_blobs(discard=5000))


print("shape log prob: {}".format(np.shape(log_prob_s)))
if param == 'vmax_ml':
    m,b = (0.3260532250619037, -1.6354239489541218)
    s[0][:,:,0]=s[0][:,:,0]*(m)+(b)
    s[0][:,:,3]=s[0][:,:,3]*(m)+(b)
    s[0][:,:,4]=s[0][:,:,4]*(m)+(b)
    s[0][:,:,1]=s[0][:,:,1]*m #sigma
    s[0][:,:,2]=s[0][:,:,2]/m #alpha

if len(files)>1:
    print('greater')
    samples = s[0]
    log_prob_samples = log_prob_s[0]
    wp_samples = wp[0]
    for i in range(len(files)):
        if i+1 < len(s):
            samples = np.concatenate((samples,s[i+1]))
            log_prob_samples = np.concatenate((log_prob_samples,log_prob_s[i+1]))
            wp_samples = np.concatenate((wp_samples,wp[i+1]))

else:
    samples = s[0]
    log_prob_samples = log_prob_s[0]
    wp_samples = wp[0]

    max_n = np.shape(samples)[0]#
    print(max_n)#samples.iteration
    # We'll track how the average autocorrelation time estimate changes
    index = 0
    autocorr = np.empty(max_n)

    # This will be useful to testing convergence
    old_tau = np.inf

    tau = reader.get_autocorr_time(tol=0)
    autocorr[index] = np.mean(tau)
    index += 1

    # Check convergence
    converged = (tau * 100 < reader.iteration)
    converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
    old_tau = tau
    print(converged,tau)
    
for j in range(5):
    chain = samples[:,:,j].T
    N = np.exp(np.linspace(np.log(100), np.log(chain.shape[1]), 20)).astype(int)
    gw2010 = np.empty(len(N))
    new = np.empty(len(N))
    for i, n in enumerate(N):
        gw2010[i] = autocorr_gw2010(chain[:, :n])
        new[i] = autocorr_new(chain[:, :n])

    # Plot the comparisons
    #plt.loglog(N, gw2010, "o-", label="G\&W 2010")
    plt.loglog(N, new, "o-", label=str(j))
ylim = plt.gca().get_ylim()
plt.plot(N, N / 50.0, "--k", label=r"$\tau = N/50$")
plt.ylim(ylim)
plt.xlabel("number of samples, $N$")
plt.ylabel(r"$\tau$ estimates")
plt.title(fname)
plt.legend(fontsize=14)
plt.savefig('autocorr_'+fname[:-3]+'.pdf')


sort_chi2 = np.sort(-2*log_prob_samples,axis=None)
plt.hist(sort_chi2[0::50],bins=50,histtype='step')
mean = np.mean(sort_chi2[0::50])
plt.axvline(mean, label=str(round(mean,4)))
plt.legend()
plt.title('Chi2s ({})'.format(fname[:-3]))
plt.savefig('chi2s_hist_'+fname[:-3]+'.pdf')

#calculate chi2 per degree of freedom
deg_of_frdm = 7
chi_per_deg = -2.0*log_prob_samples/deg_of_frdm
print("chi2 per dof:", chi_per_deg.min())
print('median,max,std: ',np.median(-2.0*log_prob_samples),np.max(-2.0*log_prob_samples),np.std(-2.0*log_prob_samples))
row,col = np.where(chi_per_deg<10.)
#randomly select 20 sets of parameters with chi2 per dof less than 6.5
choices = np.random.choice(samples[row,col].shape[0],20,replace=False)
param_string_sampling = samples[row,col][choices]
#print("Sets of parameters with chi2 per dof < 6.5")
param_sampling = []
for i in param_string_sampling:
    param_sampling.append(list(np.array(i)))

M0_vals = np.zeros(len(param_sampling))
M1_vals = np.zeros(len(param_sampling))
sigma_vals = np.zeros(len(param_sampling))
alpha_vals = np.zeros(len(param_sampling))
logMmin_vals = np.zeros(len(param_sampling))
for i in range(len(param_sampling)):
    logMmin_vals[i]+=param_sampling[i][0]
    alpha_vals[i]+=param_sampling[i][2]
    sigma_vals[i]+=param_sampling[i][1]
    M1_vals[i]+=param_sampling[i][4]
    M0_vals[i]+=param_sampling[i][3]
lo = np.median(M1_vals)-np.percentile(M1_vals,16)
hi = np.percentile(M1_vals,84)-np.median(M1_vals)
print(f"median M1 val: {round(np.median(M1_vals),4)} (+{round(lo,4)},-{round(hi,4)})")
lo = np.median(M0_vals)-np.percentile(M0_vals,16)
hi = np.percentile(M0_vals,84)-np.median(M0_vals)
print(f"median M0 val: {round(np.median(M0_vals),4)} (+{round(lo,4)},-{round(hi,4)})")
lo = np.median(logMmin_vals)-np.percentile(logMmin_vals,16)
hi = np.percentile(logMmin_vals,84)-np.median(logMmin_vals)
print(f"median logMmin val: {round(np.median(logMmin_vals),4)} (+{round(lo,4)},-{round(hi,4)})")
lo = np.median(alpha_vals)-np.percentile(alpha_vals,16)
hi = np.percentile(alpha_vals,84)-np.median(alpha_vals)
print(f"median alpha val: {round(np.median(alpha_vals),4)} (+{round(lo,4)},-{round(hi,4)})")
lo = np.median(sigma_vals)-np.percentile(sigma_vals,16)
hi = np.percentile(sigma_vals,84)-np.median(sigma_vals)
print(f"median sigma val: {round(np.median(sigma_vals),4)} (+{round(lo,4)},-{round(hi,4)})")


plt.plot(log_prob_samples[:,:],alpha=0.5)
#plt.ylim(-600,20)
plt.title("steps vs logliklihood ({})".format(fname[:-3]))
max_ll = log_prob_samples[:,:].max()
#plt.ylim(-20,0)
F"Max logliklihood: {max_ll}"
plt.savefig('steps_v_llike'+fname[:-3]+'.pdf')

#For one chain, bin chi2s in bins of 1000 steps and take the median and std value per bin
intvl = 1000
chain_num = 1
n = int(len(log_prob_samples)/intvl)
steps = np.linspace(0,n*intvl,n)
meds = np.zeros(n)
stds = np.zeros(n)
for i in range(n):
    meds[i]+=np.median(-2*log_prob_samples[i*intvl:(i+1)*intvl,chain_num])
    stds[i]+=np.std(-2*log_prob_samples[i*intvl:(i+1)*intvl,chain_num])

plt.errorbar(steps,meds,yerr=1.23*stds)
m,b = np.polyfit(steps,meds,deg=1)[0],np.polyfit(steps,meds,deg=1)[1]
plt.plot(steps,m*steps+b,label='m= {}'.format(round(m,7)))
plt.ylabel('chi2')
plt.xlabel('steps')
plt.title('Medians per 1000 steps ({})'.format(fname[:-3]))
plt.legend()
plt.savefig('med_chi2_'+fname[:-3]+'.pdf')

#fractional error of the same data as above
plt.plot(steps,1.23*stds/meds)
m,b = np.polyfit(steps,1.23*stds/meds,deg=1)[0],np.polyfit(steps,1.23*stds/meds,deg=1)[1]
plt.plot(steps,m*steps+b,label='m ={}'.format(round(m,10)))
plt.ylabel('fractional err')
plt.xlabel('steps')
plt.title('Fractional error per 1000 steps ({})'.format(fname[:-3]))
plt.legend()
plt.savefig('med_chi2_err_'+fname[:-3]+'.pdf')

#plot first and last 10000 chi2 values for a single walker
steps = np.linspace(0,10_000,10_000)
chain_num = 1
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('First 10K vs last 10K steps for single walker ({})'.format(fname[:-3]))
ax1.plot(steps, -2*log_prob_samples[0:len(steps),chain_num])
ax2.plot(steps, -2*log_prob_samples[-len(steps):len(log_prob_samples),chain_num])
for ax in fig.get_axes():
    ax.label_outer()
plt.savefig('begin_end_chains_'+fname[:-3]+'.pdf')

j = 4 #walker number
c = ['#e34a33']#['#fee8c8','#fdbb84','#e34a33']
intvl = 1000 #steps to take median over
n = int(len(samples)/intvl)
steps = np.linspace(0,n*intvl,n)

fig, axs = plt.subplots(5,figsize=(15,15),gridspec_kw={'hspace': 0},sharex=True)
meds = np.zeros(n)
stds = np.zeros(n)
for i in range(n):
    meds[i]+=np.median(samples[0:(i+1)*intvl,j,0])
    stds[i]+=np.std(samples[0:(i+1)*intvl,j,0],ddof=1)
axs[0].plot(steps,meds,c= c[0])
axs[0].fill_between(steps, meds-2*stds, meds+2*stds,color = '#e34a33',alpha = 0.5)
axs[0].set(ylabel='log{}Min'.format(param))
axs[0].set(title='Cumulative Med per param w/ 2$\sigma$ Range ({})'.format(fname[:-3]))

meds = np.zeros(n)
stds = np.zeros(n)
for i in range(n):
    meds[i]+=np.median(samples[0:(i+1)*intvl,j,1])
    stds[i]+=np.std(samples[0:(i+1)*intvl,j,1],ddof=1)
axs[1].plot(steps,meds,c= c[0])
axs[1].fill_between(steps, meds-2*stds, meds+2*stds,color = '#e34a33',alpha = 0.5)
axs[1].set(ylabel='sigmalog{}'.format(param))


meds = np.zeros(n)
stds = np.zeros(n)
for i in range(n):
    meds[i]+=np.median(samples[0:(i+1)*intvl,j,2])
    stds[i]+=np.std(samples[0:(i+1)*intvl,j,2],ddof=1)
axs[2].plot(steps,meds,c= c[0])
axs[2].fill_between(steps, meds-2*stds, meds+2*stds,color = '#e34a33',alpha = 0.5)
axs[2].set(ylabel='alpha')


meds = np.zeros(n)
stds = np.zeros(n)
for i in range(n):
    meds[i]+=np.median(samples[0:(i+1)*intvl,j,3])
    stds[i]+=np.std(samples[0:(i+1)*intvl,j,3],ddof=1)
axs[3].plot(steps,meds,c= c[0])
axs[3].fill_between(steps, meds-2*stds, meds+2*stds,color = '#e34a33',alpha = 0.5)
axs[3].set(ylabel='log{}0'.format(param))

meds = np.zeros(n)
stds = np.zeros(n)
for i in range(n):
    meds[i]+=np.median(samples[0:(i+1)*intvl,j,4])
    stds[i]+=np.std(samples[0:(i+1)*intvl,j,4],ddof=1)
axs[4].plot(steps,meds,c= c[0])
axs[4].fill_between(steps, meds-2*stds, meds+2*stds,color = '#e34a33',alpha = 0.5)
axs[4].set(ylabel='log{}1'.format(param))


plt.savefig('param_v_step_'+fname[:-3]+'.pdf')

ndim=5
fig = corner.corner(samples.reshape((-1,ndim)),
        labels=["$logVmaxMin$", "${\sigma}logVmax$", "$alpha$", "$logVmax_0$", "$logVmax$"],
        show_titles=True,title_kwargs={"fontsize": 10},quantiles=(0.16, 0.84))#, levels=(1-np.exp(-0.5),))
fig.suptitle('{}'.format(fname[:-3]))
plt.savefig('corner_'+fname[:-3]+'.pdf')
