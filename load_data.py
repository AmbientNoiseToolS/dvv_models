import numpy as np
import matplotlib.pyplot as plt
from obspy import UTCDateTime
from scipy.interpolate import interp1d

# # Start here by loading all the data
# Load vs sensitivity kernel for Rayleigh waves
# Load sensitivity kernel
def get_sensitivity_kernel(plot):
    data_dc_ds = np.loadtxt('cdmx_example_data/kernels_psv.unm.f=0.500.c=1117.595', skiprows=3)

    depth_in_meters = np.abs(6371000. - data_dc_ds[:, 0])
    K_vs = data_dc_ds[:, -2] + data_dc_ds[:, -3]

    depth_in_meters = depth_in_meters[::-1]
    K_vs = K_vs[::-1]

    if plot:
        plt.plot(K_vs, depth_in_meters, linewidth=2) #

        plt.xlabel('Sensitivity (dC/dVs)', fontsize=12)
        plt.ylabel('Depth')

        plt.ylim([3000., 0.])

        plt.grid()
        plt.show()
    return(depth_in_meters, K_vs)


# #### Load density
# load the density profile or define a density
def load_density():
    data_dc_ds = np.loadtxt('cdmx_example_data/kernels_psv.unm.f=0.500.c=1117.595', skiprows=3)
    rho = np.ones(data_dc_ds.shape[0]) * 2000.
    return(rho)

# #### Define the meteoreological data: timestamp, temperature, rain
# define a function to get meteo data
# needed: Timestamp,  temperature, rain
def get_met_data(plot):
    timestamps = np.load("cdmx_example_data/timestamps_unm_BHZ_BHN_fmin0.5Hz.npy")
    rain = np.load("cdmx_example_data/rain_unm_BHZ_BHN_fmin0.5Hz.npy")
    temperature = np.load("cdmx_example_data/temp_unm_BHZ_BHN_fmin0.5Hz.npy")

    if plot:
        plt.plot(timestamps, rain * 1000, "b")
        plt.ylabel("Rain / mm")
        plt.xticks(rotation=45)
        xt = plt.xticks()
        plt.xticks(xt[0][1:-1], [UTCDateTime(xtt).strftime('%Y-%m') for xtt in xt[0]][1:-1])
        plt.grid()
        plt.show()
    
        plt.plot(timestamps, temperature, "r")
        plt.grid()
        plt.xticks(rotation=45)
        xt = plt.xticks()
        plt.xticks(xt[0][1:-1], [UTCDateTime(xtt).strftime('%Y-%m') for xtt in xt[0]][1:-1])
        plt.ylabel("Temperature / degree C")
        plt.show()
        
    return(timestamps, rain, temperature)


# #### Load dvv data
# needed: timestamps, dvv, correlation coefficient (error is actually only needed for inversion)
def load_dvv_data(plot):
    dvv = np.load("cdmx_example_data/data_unm_BHZ_BHN_fmin0.5Hz.npy")[0]
    dvv_err = np.load("cdmx_example_data/sigma_unm_BHZ_BHN_fmin0.5Hz.npy")[0]
    tstamps = np.load("cdmx_example_data/timestamps_unm_BHZ_BHN_fmin0.5Hz.npy")

    # to run the model faster, either downsample the data
    # or cut off years, or both
    # meteo data are missing prior to about 2002
    tstamps_low = np.linspace(UTCDateTime("2016,150").timestamp, tstamps.max(), 19 * 12)
    f = interp1d(tstamps, dvv, kind="cubic")
    dvv_low = f(tstamps_low)
    f = interp1d(tstamps, dvv_err, kind="cubic")
    err_low = f(tstamps_low)

    # here we could also drop data below a certain correlation coefficient etc
    # (has been done already for this data)
    dvv_qc = dvv - dvv[0]
    dvv_qc_low = dvv_low - dvv_low[0]

    if plot:
        plt.plot(tstamps, dvv_qc * 100, "k")
        plt.plot(tstamps_low, dvv_qc_low * 100, "orange")
        plt.grid()
        plt.xticks(rotation=45)
        xt = plt.xticks()
        plt.xticks(xt[0][1:-1], [UTCDateTime(xtt).strftime('%Y-%m') for xtt in xt[0]][1:-1])
        plt.ylabel("dv/v / %")
        plt.show()

    tstamps = tstamps_low
    dvv_qc = dvv_qc_low
    return(tstamps, dvv_qc, err_low)
