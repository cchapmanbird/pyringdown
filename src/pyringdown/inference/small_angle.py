from nessai.model import Model
import numpy as np
from pyringdown.waveform.small_angle import small_angle_approx
import numba
from functools import partial
import math

@partial(numba.jit, fastmath=True)
def _loglike_kernel(output, nlike, nt, data, waveforms, sigma2):
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

    def waveform(self, x):
        wave = small_angle_approx(self.t_eval, x['A'], x['b'], x['fN'], x['phi0'], x['offset'])
        return wave
    
    def log_likelihood(self, x):
        wave = small_angle_approx(self.t_eval, x['A'], x['b'], x['fN'], x['phi0'], x['offset'])

        if 'sigma_A' in self.names:
            sigma2 = ((wave*x['sigma_A'][:,None])**2 + x['sigma'][:,None]**2)
        else:
            sigma2 = np.repeat(x['sigma'][:,None]**2, self.nt, axis=1)

        nlike = len(x['A'])
        logl = np.zeros(nlike, dtype=np.float32)

        _loglike_kernel(logl, nlike, self.nt, self.data, wave, sigma2)
        return logl
