from nessai.model import Model
import numpy as np
from pyringdown.waveform.small_angle import small_angle_approx_td, small_angle_approx_fd
from pyringdown.utility import ensure_arrays
import numba
from functools import partial
import math
from scipy.signal.windows import get_window
from scipy.ndimage import convolve1d
from scipy.optimize import differential_evolution

@partial(numba.jit, fastmath=True)
def _loglike_td_kernel(output, nlike, nt, data, waveforms, sigma2):
    for i in range(nlike):
        like_here = 0
        for j in range(nt):
            sigma2_here = sigma2[i,j]
            logsigma = math.log(2 * math.pi * sigma2_here)
            dminh = data[j] - waveforms[i,j]
            like_here -= 0.5*((dminh * dminh)/sigma2_here + logsigma)
        output[i] = like_here

    return output

class TimeDomainModel(Model):
    def __init__(self, bounds, data, fs=None, downsample=False):
        all_params = self.get_all_parameter_names()
        self.names = list(bounds.keys())
        for nm in self.names:
            if nm not in all_params:
                raise ValueError(f'Parameter "{nm} not in list of supported parameters: {all_params}')

        self.bounds = bounds

        self._vectorised_likelihood = True

        if fs is not None:
            self.data = data
            self.fs = fs
        else:
            self.fs = data[0]
            self.data = data[2:]

        if downsample:
            self.fs /= downsample
            self.data = self.data[::downsample]

        self.dt = 1/self.fs
        self.nt = len(self.data)
        self.t_eval = np.arange(self.nt)/self.fs

    def get_all_parameter_names(self):
        return ["fN", "b", "A", "phi0", "offset", "sigma", "sigma_A"]

    def log_prior(self, x):
        # TODO user defined priors from dict
        log_p = np.log(self.in_bounds(x), dtype="float")
        for n in self.names:
            log_p -= np.log(self.bounds[n][1] - self.bounds[n][0])
        return log_p

    def waveform(self, *args):
        return small_angle_approx_td(self.t_eval, *args)
    
    def log_likelihood(self, x):
        wave = small_angle_approx_td(self.t_eval, x['A'], x['b'], x['fN'], x['phi0'], x['offset'], m_drift=None)

        if 'sigma_A' in self.names:
            sigma2 = ((wave*x['sigma_A'][:,None])**2 + x['sigma'][:,None]**2)
        else:
            sigma2 = np.repeat(x['sigma'][:,None]**2, self.nt, axis=1)

        nlike = len(x['A'])
        logl = np.zeros(nlike, dtype=np.float64)

        _loglike_td_kernel(logl, nlike, self.nt, self.data, wave, sigma2)
        return logl

@partial(numba.jit, fastmath=True)
def _loglike_fd_kernel(output, nlike, nf, data, waveforms, sigma2, df):
    for i in range(nlike):
        like_here = 0
        for j in range(nf):
            sigma2_here = sigma2[i,j]
            logsigma = math.log(math.pi * sigma2_here)
            dminh = data[j] - waveforms[i,j]
            like_here -= 0.5*(4 * df * (dminh.conjugate() * dminh).real/sigma2_here + 2*logsigma)
        output[i] = like_here

    return output

def maximum_likelihood_optimisation(model, optim_kwargs=None, return_result_object=False):
    if optim_kwargs is None:
        optim_kwargs = {}
    
    def loglike_wrapper(x):
        names = model.names
        inps = {nm: ensure_arrays(x[i])[0] for i, nm in enumerate(names)}
        return -model.log_likelihood(inps)

    input_bounds = list(model.bounds.values())

    result = differential_evolution(loglike_wrapper, input_bounds, vectorized=True, updating="deferred", **optim_kwargs)

    if return_result_object:
        return result
    else:
        return result.x

class FrequencyDomainModel(Model):
    def __init__(self, bounds, data, frequency_band=None, window=None, fs=None):
        all_params = self.get_all_parameter_names()
        self.names = list(bounds.keys())
        for nm in self.names:
            if nm not in all_params:
                raise ValueError(f'Parameter "{nm} not in list of supported parameters: {all_params}')

        self.bounds = bounds

        self._vectorised_likelihood = True

        if fs is not None:
            self.td_data = data
            self.fs = fs
        else:
            self.fs = data[0]
            self.td_data = data[2:]

        self.dt = 1/self.fs
        self.nt = len(self.td_data)
        self.t_eval = np.arange(self.nt)/self.fs
        self.freqs = np.fft.rfftfreq(self.nt, self.dt)

        if window is None:
            self.window = np.ones(self.nt)
        else:
            self.window = get_window(window, self.nt)

        self.td_data = self.td_data * self.window
        
        self.data = np.fft.rfft(self.td_data) #/ self.fs

        # noisesigma = 1e-2 * (1/2)**0.5
        # noise = np.random.normal(scale=noisesigma, size=self.freqs.size) + 1j*(np.random.normal(scale=noisesigma, size=self.freqs.size))
        # self.data += noise

        self.data /= self.fs

        if frequency_band is not None:
            self.fmin_ind = np.argmin(abs(self.freqs - frequency_band[0]))
            self.fmax_ind = np.argmin(abs(self.freqs - frequency_band[1]))

            self.data = self.data[self.fmin_ind:self.fmax_ind]
            self.freqs = self.freqs[self.fmin_ind:self.fmax_ind]

            self.fmin, self.fmax = self.freqs[0], self.freqs[-1]

        else:
            self.fmin, self.fmax = 0., self.freqs[-1]

        self.nf = self.freqs.size
        self.df = self.freqs[1] - self.freqs[0]

        # window_fft = np.fft.rfft(self.window, norm="forward")
        # self.convolution_kernel = np.concatenate((np.flip(window_fft[:self.nf]), window_fft[1:self.nf]))  # the kernel for template-convolving.
        # self.convolution_kernel = np.concatenate((np.flip(window_fft[:self.nf]), window_fft[1:self.nf]))  # the kernel for template-convolving.



    def get_all_parameter_names(self):
        return ["fN", "b", "A", "phi0", "offset", "sigma"]

    def log_prior(self, x):
        # TODO user defined priors from dict
        log_p = np.log(self.in_bounds(x), dtype="float")
        for n in self.names:
            log_p -= np.log(self.bounds[n][1] - self.bounds[n][0])
        return log_p

    def waveform(self, *args):
        return small_angle_approx_fd(self.freqs, *args)
    
    def log_likelihood(self, x):
        # wave = small_angle_approx_fd(self.freqs, x['A'], x['b'], x['fN'], x['phi0'])
        wave = small_angle_approx_td(self.t_eval, x['A'], x['b'], x['fN'], x['phi0'], np.zeros_like(x['phi0']))
        wave *= self.window[None,:]
        wave = np.fft.rfft(wave, axis=-1)[:,self.fmin_ind:self.fmax_ind] / self.fs
        # breakpoint()
        # convolve1d(wave, weights=self.convolution_kernel, output=wave)
        # breakpoint()
        sigma2 = np.repeat(x['sigma'][:,None]**2, self.nf, axis=1)

        nlike = len(x['A'])
        logl = np.zeros(nlike)

        _loglike_fd_kernel(logl, nlike, self.nf, self.data, wave, sigma2, self.df)
        return logl
