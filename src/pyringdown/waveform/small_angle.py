from functools import partial
import numba
import math
import cmath
from pyringdown.utility import ensure_arrays
import numpy as np
from typing import Union, Optional

# @partial(numba.jit, fastmath=False)
# def _sa_td_kernel(output, nwave, nt, time, A, b, fN, theta0, phi0, offset, m_drift):
#     for i in numba.prange(nwave):
#         beta = b[i]/2
#         omega = 2 * math.pi * fN[i]
#         omega_star = math.sqrt(omega*omega - beta*beta)
#         for j in numba.prange(nt):
#             theta1 = theta0[i] * math.exp(-beta * time[j]) * math.cos(omega_star * time[j] + phi0[i]) / 2.
#             output[i, j] = A[i] * math.sin(theta1) + m_drift[i]*time[j] + offset[i]

@partial(numba.jit, fastmath=True)
def _sa_td_kernel(output, nwave, nt, time, A, b, fN, phi0, offset, m_drift):
    for i in numba.prange(nwave):
        beta = b[i]/2
        omega = 2 * math.pi * fN[i]
        omega_star = math.sqrt(omega*omega - beta*beta)
        for j in numba.prange(nt):
            output[i, j] = A[i] * math.exp(-beta * time[j]) * math.cos(omega_star * time[j] + phi0[i]) + m_drift[i]*time[j] + offset[i]


# @partial(numba.jit, fastmath=True)
# def _sa_td_kernel_series(output, nwave, nt, time, A, b, fN, theta0, phi0, offset, m_drift, nterms):
#     for i in numba.prange(nwave):
#         beta = b[i]/2
#         omega = 2 * math.pi * fN[i]
#         omega_star = math.sqrt(omega*omega - beta*beta)

#         for j in numba.prange(nt):
#             theta1 = theta0[i] * math.exp(-beta * time[j]) #* math.cos(omega_star * time[j] + phi0[i]) / 2.

#             eps_part = math.sqrt(math.cos(theta1/2))
#             epsilon = 0.5 * (1 - eps_part) / (1 + eps_part)
#             q = epsilon + 2*math.pow(epsilon,5) + 15*math.pow(epsilon,9) + 150*math.pow(epsilon,13) + 1707*pow(epsilon, 17) + 20910*pow(epsilon,21)
            
#             part = 0
#             for k in numba.prange(nterms):
#                 n = 1+2*k
#                 part += ((pow(-1, k)/n) * (pow(q, n/2)/(1+pow(q,n))) * math.cos(n*omega_star*time[j]+ phi0[i]))

#             output[i, j] = A[i]*math.sin(8*part/2) + m_drift[i]*time[j] + offset[i]
    

# def small_angle_approx_td(t, A, b, fN, theta0, phi0, offset=None, m_drift=None, nterms=1, dtype=np.float64):
#     if offset is None:
#         offset = np.zeros_like(A)
#     if m_drift is None:
#         m_drift = np.zeros_like(A)
#     t, A, b, fN, theta0, phi0, offset, m_drift = ensure_arrays(t, A, b, fN, theta0, phi0, offset, m_drift)

#     nwave, nt = A.size, t.size
#     output = np.empty((nwave, nt), dtype=dtype)
#     # nterms=5
#     if nterms == 1:
#         _sa_td_kernel(output, nwave, nt, t, A, b, fN, theta0, phi0, offset, m_drift)
#     else:
#         _sa_td_kernel_series(output, nwave, nt, t, A, b, fN, theta0, phi0, offset, m_drift, nterms=nterms)
#     return output

def small_angle_approx_td(
        t: Union[float, np.ndarray], 
        A: Union[float, np.ndarray], 
        b: Union[float, np.ndarray], 
        fN: Union[float, np.ndarray], 
        phi0: Union[float, np.ndarray], 
        offset: Optional[Union[float, np.ndarray]]=None, 
        m_drift: Optional[Union[float, np.ndarray]]=None, 
        dtype: np.dtype=np.float64):
    """
    Small-angle approximation for theta(t), including damping via a linear friction term.
    Returns in the time domain.

    Args:
        t: Time at which to evaluate the waveform.
        A: Amplitude of oscillation.
        b: Damping coefficient.
        fN: The natural frequency of the oscillation (which equals the oscillation frequency for `b=0`).
        phi0: Phase offset of the oscillation.
        offset: An additive offset to be applied to the waveform before returning.
        m_drift: The gradient of an additive linear drift term to be applied to the waveform before returning.
        dtype: The numerical precision with which to compute the waveform (defaults to double precision).
    """

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

def small_angle_approx_fd(
        f: Union[float, np.ndarray], 
        A: Union[float, np.ndarray], 
        b: Union[float, np.ndarray], 
        fN: Union[float, np.ndarray], 
        phi0: Union[float, np.ndarray], 
        dtype: np.dtype=np.complex128):
    """
    Small-angle approximation for theta(t), including damping via a linear friction term.
    Returns in the frequency domain.

    NB: One should be mindful of spectral leakage effects when performing analysis of time domain data in the frequency domain.

    Args:
        f: Frequency at which to evaluate the waveform.
        A: Amplitude of oscillation.
        b: Damping coefficient.
        fN: The natural frequency of the oscillation (which equals the oscillation frequency for `b=0`).
        phi0: Phase offset of the oscillation.
        dtype: The numerical precision with which to compute the waveform (defaults to double precision complex).
    """

    f, A, b, fN, phi0 = ensure_arrays(f, A, b, fN, phi0)

    nwave, nf = A.size, f.size
    output = np.empty((nwave, nf), dtype=dtype)
    _sa_fd_kernel(output, nwave, nf, f, A, b, fN, phi0)
    return output
