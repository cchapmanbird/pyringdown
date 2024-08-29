from functools import partial
import numba
import math
from pyringdown.utility import ensure_arrays
import numpy as np

@partial(numba.jit, fastmath=True)
def _sa_kernel(output, nwave, nt, time, A, b, fN, phi0, offset):
    for i in numba.prange(nwave):
        beta = b[i]/2
        omega = 2 * math.pi * fN[i]
        omega_star = math.sqrt(omega*omega - beta*beta)
        for j in numba.prange(nt):
            output[i, j] = math.exp(-beta * time[j]) * A[i] * math.cos(omega_star * time[j] + phi0[i]) + offset[i]
    
def small_angle_approx(t, A, b, fN, phi0, offset=None, dtype=np.float64):
    if offset is None:
        offset = np.zeros_like(A)
    t, A, b, fN, phi0, offset = ensure_arrays(t, A, b, fN, phi0, offset)

    nwave, nt = A.size, t.size
    output = np.empty((nwave, nt), dtype=dtype)
    _sa_kernel(output, nwave, nt, t, A, b, fN, phi0, offset)
    return output
