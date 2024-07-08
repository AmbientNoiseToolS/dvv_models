import numpy as np
import matplotlib.pyplot as plt
from load_data import *
from model import roeloffs, get_temperature_z, evaluate_model_quakes_lin



qtimes = [UTCDateTime("2017-09-19T18:14:40")]
tstamps, dvv_qc, dvv_error = load_dvv_data(plot=False)
a = np.load("inversion_results/percs_test_inversion_noconvergence.npy")
independent_variables = [tstamps, qtimes]

# plot the models
plt.plot(tstamps, dvv_qc, "k", linewidth=2)
plt.plot(tstamps, evaluate_model_quakes_lin(independent_variables, a[:, 0]), "b--", linewidth=1)
plt.plot(tstamps, evaluate_model_quakes_lin(independent_variables, a[:, 1]), "b", linewidth=1)
plt.plot(tstamps, evaluate_model_quakes_lin(independent_variables, a[:, 2]), "b--", linewidth=1)
plt.xlabel("Time stamp (s)")
plt.ylabel("dv/v (-)")
plt.show()

