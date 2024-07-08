import os
# turn off underlying numpy paralellisation
os.environ["OMP_NUM_THREADS"] = "1"
from load_data import *
from scipy.interpolate import interp1d
from model import roeloffs, get_temperature_z, evaluate_model_quakes_lin
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
import emcee
import corner
from math import pi
# uniform PRIOR for emcee
def uniform_prior_for_emcee(params, bounds):
    for i, p in enumerate(params):
        # print(f"parameter nr {i} value {p}")
        # print(f"parameter nr {i} bounds {bounds[0][i], bounds[1][i]}")
        if p < bounds[0][i]:
            return(-np.inf)
        if p > bounds[1][i]:
            return(-np.inf)
    return(0.0)

# neg. log. LIKELIHOOD for emcee
def log_likelihood_for_emcee(params, ind_vars, data, data_err):
    ll = 0.0

    # get the synthetics for these parameters
    synth = evaluate_model_quakes_lin(ind_vars, params)
    
    # data could be several dimensions
    if np.ndim(data) == 1:
        data = np.array(data, ndmin=2)
        data_err = np.array(data_err, ndmin=2)

    for ixd, d in enumerate(data):

        # here, I had some adjustments because the Weaver theoretical error is quite small
        # this is potentially an underestimation of the real error, and it makes it hard to find a fitting model
        # log_f parameter is used to account for possible underestimate of observational uncertainty.
        # uncertainty (data_err) is otherwise based on Weaver et al
        # not sure if we need them or can live without them.
        # if error_is_underestimated_g:
        #     sigma2 = (g * data_err[ixd]) ** 2
        # elif error_is_underestimated_logf:
        #     sigma2 = (data_err[ixd] + np.exp(log_f)) ** 2
        # else:
        #     sigma2 = data_err[ixd] ** 2

        # try using the maximum data error, worst case
        #sigma2 = np.ones(data_err.shape) * data_err.max()
        sigma2 = data_err[ixd] ** 2
        res = synth - d
        ll += -0.5 * np.sum(res ** 2 / sigma2) - np.sum(np.log(np.sqrt(2. * pi * sigma2)))

    return ll


# neg. log. probability for emcee
def log_probability_for_emcee(params, ind_vars, bounds, data, cov):

    prior = uniform_prior_for_emcee(params, bounds)
    llh = log_likelihood_for_emcee(params, ind_vars, data, cov)
    #print("Prior, ", prior)
    #print("loglikelihood, ", llh)
    if not np.isfinite(prior):
        return(-np.inf)
    else:
        return(llh + prior)




# Fixed model parameters and starting values
# earthquake-related parameters
# =============================

# fixed throughout inversion:
# origin time of the earthquake
qtimes = [UTCDateTime("2017-09-19T18:14:40")] # , UTCDateTime("2020-06-23T15:29:05")]

# starting values and bounds for inversion params:
# list of the log10 of maximum relaxation times (one per earthquake)
log10_tau_maxs = [6] #[np.log10(86400. * 365 * 10), np.log10(86400. * 365 * 0.5)]
lower_log10_tau_maxs = [0]  # log 10 of lower bound for relaxation time, e.g. 3 is 1000 seconds
upper_log10_tau_maxs = [12]  # 10**10 seconds is about 317 years, accordingly for 31 and 3 years
# list of the velocity drops (one per earthquake).
# unitless (i.e. 0.1 is 10%)
log10_drops = [-6] #[0.1, 0.002]
lower_log10_drops = [-12,]
upper_log10_drops = [0]


# linear increase parameters
# ==========================
# starting values and bounds for inversion params:
log10_slope = -6 #np.log10(0.0018 / (86400. * 365.)) # in rate per year, starting value
lower_log10_slope = -12
upper_log10_slope = 0
log10_offset = -6  # offset, starting value
lower_log10_offset = -12
upper_log10_offset = 0

# inversion setup
# ===============
# how many chains
n_initializations = 12
# how many processes
n_multiprocess = 1
# how many iterations until checking for convergence
n_iterations = 50_000
# how many iterations overall
max_iterations = 150_000
# Burn-in phase: at the beginning the sampler varies widely until it finds
# a high-likelihood area. burn-in samples will be discarded
n_burnin = 30_000
# output directory
output_dir = "inversion_results"


# put here your data loading ============================
tstamps, dvv_qc, dvv_error = load_dvv_data(plot=False)
# timestamps in seconds, dvv unit-free (not percent)
# ======================================================

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# independent variables in this model: time, earthquake integer timestamps of origin time
independent_variables = [tstamps, qtimes]

# initial values
parameters = [*log10_tau_maxs, *log10_drops, log10_slope, log10_offset]

# bounds
lower_bounds = [*lower_log10_tau_maxs, *lower_log10_drops, lower_log10_slope, lower_log10_offset]
upper_bounds = [*upper_log10_tau_maxs, *upper_log10_drops, upper_log10_slope, upper_log10_offset]

bounds = [lower_bounds, upper_bounds]
init_perturbation = [*[0.1], *[0.1], 0.1, 0.1]


# get the initial position from the max. likeligood model and perturb by small random nrs
init_pos = parameters
print("initial values: ", init_pos)

# run the inversion
perturb_initial_values = np.random.randn(n_initializations, len(init_pos)) *\
    np.array(init_perturbation)

position = np.array([init_pos]*n_initializations) + perturb_initial_values
for ixp, p in enumerate(position):
    plt.plot(p)
    plt.plot(perturb_initial_values[ixp], "--")
plt.show()

nwalkers, ndim = (n_initializations, len(init_pos))

with multiprocessing.Pool(n_multiprocess) as pool:
    # Initialize the sampler
    # here you can edit the types of moves etc.
    ind_vars = independent_variables
    cov = dvv_error
    data = dvv_qc
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_for_emcee,
                                    moves=[(emcee.moves.StretchMove(a=2.0), 0.5),
                                            (emcee.moves.DESnookerMove(), 0.5)],
                                    args=(ind_vars, bounds, data, cov),
                                    pool=pool)
    iterations_performed = 0
    have_tau = False
    while True:
        try:
            position = sampler.run_mcmc(position, n_iterations, progress=True, skip_initial_state_check=True)
            iterations_performed += n_iterations
        except ValueError:  # badly conditioned starting point
            position = sampler.get_last_sample()
            position.coords += np.random.randn(n_initializations, len(init_pos)) *\
                               np.array(init_perturbation[0: len(init_pos)])
        try:
            tau = sampler.get_autocorr_time(discard=n_burnin)
            foname = output_dir + "/tau.txt"
            with open(foname, "w") as fh:
                fh.write(tau)
            thin = int(np.max(tau)) // 2
            print("Tau could be estimated, tau: ", np.max(tau))
            have_tau = True
            break
        except:
            print("Apparently no convergence yet, adding another {} samples.".format(n_iterations))
        if iterations_performed >= max_iterations:
            # give up :~(
            break

if not have_tau:
    thin = 2
    #continue


# get and save the samples
all_samples_temp = sampler.get_chain(discard=n_burnin)
# save the "clean" ensemble: Post burn-in, flat, decimated by 1/2 * autocorrelation time.
flat_samples = sampler.get_chain(flat=True, discard=n_burnin, thin=thin)
log_prob_samples = sampler.get_log_prob(discard=n_burnin, flat=True, thin=thin)

foname = output_dir + "/probability_{}{}.npy".format("test_inversion", {True: "", False: "_noconvergence"}[have_tau])
np.save(foname, log_prob_samples)
foname = (output_dir + "/samples_{}{}.npy".format("test_inversion", {True: "", False: "_noconvergence"}[have_tau]))
np.save(foname, flat_samples)

# get the median and percentile models and save
mcmcout = []
for ixp in range(ndim):
    mcmcout.append(np.percentile(flat_samples[:, ixp], [16, 50, 84]))
mcmcout = np.array(mcmcout)
foname = (output_dir + "/percs_{}{}.npy".format("test_inversion", {True: "", False: "_noconvergence"}[have_tau]))
np.save(foname, mcmcout)

# Plot the chains
fig, axes = plt.subplots(ndim, figsize=(10, 2*ndim), sharex=True)
labels = ["tau1",  "drop1",  "slope", "offset", "log prob"]
for ixparam in range(ndim):
    ax = axes[ixparam]
    ax.plot(all_samples_temp[:, :, ixparam], "k", alpha=0.3)
    ax.set_xlim(0, len(all_samples_temp))
    ax.set_ylabel(labels[ixparam])
    ax.yaxis.set_label_coords(-0.1, 0.5)
axes[-1].set_xlabel("Step number")
foname = (output_dir + "/MCMC_chains_{}{}.png".format("test_inversion", {True: "", False: "_noconvergence"}[have_tau]))
fig.savefig(foname)
plt.close()


# Create a corner plot and save
flat_samples = sampler.get_chain(flat=True, discard=n_burnin, thin=thin)
log_prob_samples = sampler.get_log_prob(discard=n_burnin, flat=True, thin=thin)
print(flat_samples.shape)
print(log_prob_samples.shape)
print(log_prob_samples)
if flat_samples.shape[0] > 50:
    samples_probs = np.concatenate((flat_samples, log_prob_samples[:, None]), axis=1)
    print(samples_probs.shape)

    labels += ["log prob"]
    fig = corner.corner(
        samples_probs, labels=labels
    );
    foname = (output_dir + "/MCMC_{}{}.png".format("test_inversion", {True: "", False: "_noconvergence"}[have_tau]))
    fig.savefig(foname)
    plt.close()

