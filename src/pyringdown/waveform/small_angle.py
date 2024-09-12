from functools import partial
import numba
import math
import cmath
from pyringdown.utility import ensure_arrays
import numpy as np

@partial(numba.jit, fastmath=True)
def _sa_td_kernel(output, nwave, nt, time, A, b, fN, phi0, offset, m_drift):
    for i in numba.prange(nwave):
        beta = b[i]/2
        omega = 2 * math.pi * fN[i]
        omega_star = math.sqrt(omega*omega - beta*beta)
        for j in numba.prange(nt):
            output[i, j] = math.exp(-beta * time[j]) * A[i] * math.cos(omega_star * time[j] + phi0[i]) + m_drift[i]*time[j] + offset[i]
    
def small_angle_approx_td(t, A, b, fN, phi0, offset=None, m_drift=None, dtype=np.float64):
    if offset is None:
        offset = np.zeros_like(A)
    if m_drift is None:
        m_drift = np.zeros_like(A)
    t, A, b, fN, phi0, offset, m_drift = ensure_arrays(t, A, b, fN, phi0, offset, m_drift)

    nwave, nt = A.size, t.size
    output = np.empty((nwave, nt), dtype=dtype)
    _sa_td_kernel(output, nwave, nt, t, A, b, fN, phi0, offset, m_drift)
    return output

@partial(numba.jit, fastmath=True)
def _sa_fd_kernel(output, nwave, nf, freqs, A, b, fN, phi0):
    for i in numba.prange(nwave):
        beta = b[i]/2
        eiphi = cmath.exp(1j*phi0[i])
        omega = 2 * math.pi * fN[i]
        omega_star = math.sqrt(omega*omega - beta*beta)
        for j in numba.prange(nf):
            output[i, j] = A[i] / 2 * (eiphi / (beta + 1j*(2*math.pi*freqs[j]-omega_star)) +  eiphi.conjugate() / (beta + 1j*(2*math.pi*freqs[j]+omega_star)))

def small_angle_approx_fd(f, A, b, fN, phi0, dtype=np.complex128):

    f, A, b, fN, phi0 = ensure_arrays(f, A, b, fN, phi0)

    nwave, nf = A.size, f.size
    output = np.empty((nwave, nf), dtype=dtype)
    _sa_fd_kernel(output, nwave, nf, f, A, b, fN, phi0)
    return output
