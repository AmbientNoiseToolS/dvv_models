#!/usr/bin/env python
# coding: utf-8
# imports
import numpy as np
from obspy import UTCDateTime
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.interpolate import interp1d
from scipy.fftpack import next_fast_len
from utils.model_tools_kurama import logheal_llc
from scipy.special import erf, erfc


# ### Model for earthquake healing
# accelerated implementation using low-level callback by Kurama Okubo 
# Okubo, K., Delbridge, B. G., & Denolle, M. A. (2024). Monitoring velocity change over 20 years at Parkfield. Journal of Geophysical Research: Solid Earth, 129, e2023JB028084. https://doi.org/10.1029/2023JB028084 
def func_healing(independent_vars, params, time_quake="2017,09,19,18,14,00"):
    # full implementation of Snieder's healing model from Snieder et al. 2017
    # but faster
    # (c) by Kurama Okubo
    # update by Laura: quake is shifted to nearest time sample (necessary for correct normalization of the relaxation integral)
    t = independent_vars[0]
    if len(independent_vars) == 2:
        time_quake = independent_vars[1]

    tau_min = 0.1
    tau_max = params[0]
    drop_eq = params[1]
    tquake = UTCDateTime(time_quake).timestamp

    ix_eq_on_timeaxis = np.argmin((t - tquake)**2)
    tquake = t[ix_eq_on_timeaxis]
    
    tax = t - tquake
    ixt = tax >= 0
    tax[~ixt] = 0.0

    dv_quake = np.zeros(len(tax))

    # separate function accelerated by c and low level callback for scipy quad
    dv_quake[ixt] = [logheal_llc(tt, tau_min, tau_max, drop_eq) for tt in tax[ixt]]
    dv_quake /= np.log(tau_max/tau_min)
   
    return(dv_quake)

# for more than 1 quake in the timeseries
def func_healing_list(independent_vars, params):
    t = independent_vars[0]
    quakes = independent_vars[1]

    dv_quakes = np.zeros(len(t))
    tau_max_list = params[0]
    drop_eq_list = params[1]

    for i in range(len(quakes)):
        dv_quakes += func_healing([t], [tau_max_list[i], drop_eq_list[i]], time_quake=quakes[i])
    return(dv_quakes)


# ### Model for pore pressure effect on dvv
#################################################
# Hydrology: 1-D poroelastic response to rainfall
#################################################
def get_effective_pressure(rhophi, z, rhos):
    p = np.zeros(len(z))
    dz = np.zeros(len(z))
    dz[:-1] = z[1:] - z[:-1]
    dz[-1] = dz[-2]

    p[1: ] += np.cumsum(rhos * 9.81 * dz)[:-1]  # overburden

    # parameter rhophi: rho water * porosity
    p[1: ] -= z[1:] * rhophi[1:] * 9.81 # roughly estimated pore pressure -- just set to hydrostatic pressure here
    return(p)

def model_SW_dsdp(p_in, waterlevel=100.):
    # 1 / vs  del v_s / del p: Derivative of shear wave velocity to effective pressure
    # identical for Walton smooth model and Hertz-Mindlin model
    # input: 
    # p_in (array of int or float): effective pressure (hydrostatic - pore)
    # waterlevel: to avoid 0 division at the free surface. Note that results are sensitive to this parameter.
    # output: 1/vs del vs / del p

    p_in[p_in < waterlevel] = waterlevel
    p_in[p_in<=0] = p_in[p_in > 0].min()
    sens = 1. / (6. * p_in)
    return(sens)

def roeloffs_1depth(t, rain, r, B_skemp, nu, diff,
                    rho, g, waterlevel, model, nfft=None):
    # evaluate Roeloff's response function for a specific depth r
    # input:
    # t: time vector in seconds
    # rain: precipitation time series in m
    # r: depth in m
    # B_skemp: Skempton's coefficient (no unit)
    # nu: Poisson ratio (no unit)
    # diff: Hydraulic diffusivity, m^2/s
    # rho: Density of water (kg / m^3)
    # g: gravitational acceleration (N / kg)
    # waterlevel: to avoid zero division at the surface. Results are not sensitive to the choice of waterlevel
    # model: drained, undrained or both (see Roeloffs, 1988 paper)
    # output: Pore pressure time series at depth r

    # use nfft to try an increase convolution speed
    if nfft is None:
        nfft = len(t)

    dp = rho * g * rain
    dt = t[1] - t[0]  # delta t, sampling (e.g. 1 day)
    diffterm = 4. * diff * np.arange(len(t)) * dt
    diffterm[0] = waterlevel
    diffterm = r / np.sqrt(diffterm)
    
    resp = erf(diffterm)
    rp = np.zeros(nfft)
    rp[0: len(resp)] = resp
    P_ud = np.convolve(rp, dp, "full")[0: len(dp)]
    
    resp = erfc(diffterm)
    rp = np.zeros(nfft)
    rp[0: len(resp)] = resp
    P_d = np.convolve(rp, dp, "full")[0: len(dp)]
    if model == "both":
        P = P_d + B_skemp * (1 + nu) / (3. - 3. * nu) * P_ud
    elif model == "drained":
        P = P_d
    elif model == "undrained":
        P = B_skemp * (1 + nu) / (3. - 3. * nu) * P_ud
    else:
        raise ValueError("Unknown model for Roeloff's poroelastic response. Model must be \"drained\" or \"undrained\" or \"both\".")
    return P

def roeloffs(t, rain, r, B_skemp, nu, diff, rho=1000.0, g=9.81, waterlevel=1.e-12, model="both"):
    s_rain = np.zeros((len(t), len(r)))
    fftN = next_fast_len(len(t))
    for i, depth in enumerate(r):
        p = roeloffs_1depth(t, rain, depth, B_skemp, nu, diff,
                            rho, g, waterlevel, model, nfft=fftN)
        s_rain[:, i] = p
    return(s_rain)


def func_rain(independent_vars, params):
    # This function does the bookkeeping for predicting dv/v from pore pressure change.
    z = independent_vars[0]
    dp_rain = independent_vars[1]
    rhos = independent_vars[2]
    phis = independent_vars[3]
    kernel = independent_vars[4]

    dz = np.zeros(len(z))
    dz[:-1] = z[1:] - z[:-1]
    dz[-1] = dz[-2]

    waterlevel = params[0]

    rhophi = 1000.0 * phis
    p = get_effective_pressure(rhophi, z, rhos)
    stress_sensitivity = model_SW_dsdp(p, waterlevel)
    dv_rain = np.dot(-dp_rain, stress_sensitivity * kernel * dz)

    return(dv_rain)


# ### Thermoelastic effect
#################################################
# Thermoelastic effect following Richter et al., 2015
#################################################

def diff_temp_term(t0_surface, t, z, n, diff, w0=2.*np.pi/(365.25*86400.0)):
    gamma = np.sqrt(n * w0 / (2. * diff))
    ts = t0_surface * np.exp(1.j * (n * t * w0 - gamma * z) - gamma * z)
    return(np.real(ts))

def cn(n, t, y, tau=86400.0 * 365.25):
    c = y * np.exp(-1.j * 2 * n * np.pi * t / tau)
    return c.sum()/c.size


def get_temperature_z(t, T_surface, z, thermal_diffusivity,
                      n_fourier_components=6):
    
    T_surface -= T_surface.mean()

    # get Fourier series representation of temperature
    fcoeffs = np.array([cn(n, t - t.min(), T_surface, tau=86400.0 * 365.25) \
        for n in range(n_fourier_components)])

    # get diffusion result
    difftemp = np.zeros((len(t), len(z)))
    for ix, zz in enumerate(z):
        for n, fc in enumerate(fcoeffs):
            difftemp[:, ix] += np.array([diff_temp_term(fc, tt, zz, n, thermal_diffusivity) \
            for tt in t - t.min()])

    # return diffusion result
    return(difftemp)

def func_temp(independent_vars, params):

    t = independent_vars[0]
    z = independent_vars[1]
    kernel = independent_vars[2]
    dp_temp = independent_vars[3]

    dz = np.zeros(len(z))
    dz[:-1] = z[1:] - z[:-1]
    dz[-1] = dz[-1]

    assert dz[0] > 0.0
    sensitivity_factor = params[0]
    dv_temp = sensitivity_factor * np.dot(dp_temp, kernel * dz)
    return(dv_temp)


#################################################
# Linear velocity increase / decrease
#################################################
def func_lin(independent_vars, params):
    # linear trend
    t = independent_vars[0]
    slope = params[0]
    const = params[1]
    dv_y = slope * (t - t.min()) + const
    return(dv_y)


# ### Superposition of several model terms for the final dvv timeseries
def evaluate_model_quakes_rain_temp_lin(ind_vars, params, return_all=False):

    # independent variables in this model: time, depth, surface wave sensitivity kernel (depths equal to depth array)
    # density, porosity, rain in m, temperature in degrees and earthquake integer timestamps of origin time
    t = ind_vars[0]
    z = ind_vars[1]
    z_T = ind_vars[2]
    kernel_vs = ind_vars[3]
    kernel_vs_T = ind_vars[4]
    rho = ind_vars[5]
    phi = ind_vars[6]
    dp_rain = ind_vars[7]
    dp_temp = ind_vars[8]
    quakes_timestamps = ind_vars[9]
    
    # Parameters: earthquake maximum relaxation times (as many as earthquake timestamps)
    # velocity drops (as many as earthquake timestamps), decadic log of pressure at the surface in Pascal), 
    # decadic log of temperature sensitivity
    
    tau_maxs = [10. ** p for p in params[0: len(quakes_timestamps)]]
    drops = params[len(quakes_timestamps): 2 * len(quakes_timestamps)]
    p0 = 10. ** params[2 * len(quakes_timestamps)]
    tsens = 10. ** params[2 * len(quakes_timestamps) + 1]
    slope = params[2 * len(quakes_timestamps) + 2]
    offset = params[2 * len(quakes_timestamps) + 3]
    
    dv_rain = func_rain([z, dp_rain, rho, phi, kernel_vs], [p0])
    dv_temp = func_temp([t, z_T, kernel_vs_T, dp_temp], [tsens])
    dv_lin = func_lin([t], [slope, offset])

    dv_quake = np.zeros(len(t))
    for ixq, q in enumerate(quakes_timestamps):
        dv_quake += func_healing([t], [tau_maxs[ixq], drops[ixq]], time_quake=q)
    
    if return_all:
        return(dv_rain + dv_temp + dv_quake + dv_lin, [dv_rain, dv_temp, dv_quake, dv_lin])
    else:
        return(dv_rain + dv_temp + dv_quake + dv_lin)


# ### only quake / healing model and linear (make it easier to troubleshoot the mcmc)
def evaluate_model_quakes_lin(ind_vars, params, return_all=False):

    # independent variables in this model: time, depth, surface wave sensitivity kernel (depths equal to depth array)
    # density, porosity, rain in m, temperature in degrees and earthquake integer timestamps of origin time
    t = ind_vars[0]
    quakes_timestamps = ind_vars[1]
    
    # Parameters: earthquake maximum relaxation times (as many as earthquake timestamps)
    # velocity drops (as many as earthquake timestamps), decadic log of pressure at the surface in Pascal), 
    # decadic log of temperature sensitivity
    
    tau_maxs = [10. ** p for p in params[0: len(quakes_timestamps)]]
    drops = [100 ** p for p in params[len(quakes_timestamps): 2 * len(quakes_timestamps)]]
    dv_quake = np.zeros(len(t))
    for ixq, q in enumerate(quakes_timestamps):
        dv_quake += func_healing([t], [tau_maxs[ixq], drops[ixq]], time_quake=q)
    
    slope = 10 ** params[2 * len(quakes_timestamps)]
    offset = 10 ** params[2 * len(quakes_timestamps) + 1]
    dv_lin = func_lin([t], [slope, offset])

    return(dv_quake + dv_lin)


# ### additional convenience functions
# Function to replace NaN values with nearest non-NaN values
# by ChatGPT
# prompt: I have a numpy array in Python that contains several NaN-values, and I would like to replace them by the nearest finite value.
def replace_nan_with_nearest(arr, interpolation_method="nearest"):
    indices = np.arange(len(arr))
    valid = ~np.isnan(arr)
    
    # Nearest-neighbor interpolation
    f = interp1d(indices[valid], arr[valid], bounds_error=False,
                 fill_value="extrapolate", kind=interpolation_method)
    
    arr[~valid] = f(indices[~valid])
    return arr
