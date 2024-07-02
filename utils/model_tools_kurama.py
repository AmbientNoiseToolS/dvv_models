# 2022.02.02 Kurama Okubo

# modules used for Low level callback function duting integration
import os, ctypes
import numpy as np
from scipy import integrate, LowLevelCallable
# Integration with low level callback function
# read shared library
lib_int = ctypes.CDLL(os.path.abspath('./utils/healing_int.so'))
lib_int.f.restype = ctypes.c_double
lib_int.f.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p)
    
def logheal_llc(ts, taumin, taumax, S):

    if ts < 0:
        return(0)
    else:
        # using Low-level caling function    
        c = ctypes.c_double(ts) # This is the argument of time t as void * userdata
        user_data = ctypes.cast(ctypes.pointer(c), ctypes.c_void_p)
        int1_llc = LowLevelCallable(lib_int.f, user_data) # in this way, only void* is available as argument
        
        return -S * integrate.quad(int1_llc, taumin, taumax)[0]

def GWL_SSW06(t, precip, phi, a):
    dt = np.diff(t).mean()
    Nprecip = len(precip)
    expij = [np.exp(-a * dt * x) for x in range(Nprecip)]
    GWL = np.convolve(expij, precip, mode='full')[:Nprecip] / phi
    return GWL