import equinox as eqx
from ..waveform.interface import TDWaveform, FDWaveform
import jax.numpy as jnp
from typing import Optional, Union

class Likelihood(eqx.Module):
    """
    Likelihood baseclass.

    """
    
    fixed_parameters: dict

    def __init__(self, fixed_parameters: Optional[dict] = None):

        if fixed_parameters is None:
            fixed_parameters = {}
        
        self.fixed_parameters = fixed_parameters

    def fill_parameters(self, parameters: dict):
        parameters.update(self.fixed_parameters)

    def log_likelihood(self, parameters: dict):
        raise NotImplementedError

    def __call__(self, parameters: dict):
        """
        Evaluate the likelihood of the waveform given the parameters, under the assumption of Gaussian noise.
        
        Args:
            parameters: Parameters for the waveform and noise models. See `self.waveform.parameters` for details.
        
        Returns:
            The log-likelihood given the waveform parameters and noise variance.
        """

        self.fill_parameters(parameters)
        return self.log_likelihood(parameters)

class TDLikelihood(Likelihood):
    """
    Time-domain likelihood class.
    """

    waveform: Union[TDWaveform, FDWaveform]
    data: Optional[jnp.ndarray]

    def __init__(self, waveform_model: TDWaveform, data: Optional[jnp.ndarray] = None, fixed_parameters: Optional[dict] = None):
        """
        Args:
            waveform: An instance of a TDWaveform class.
            data: Optional data to compare against the waveform. If provided, must match the length of the waveform.
            fixed_parameters: Optional dictionary of parameters and values to keep fixed during likelihood evaluation.
        """
        super().__init__(fixed_parameters=fixed_parameters)
        self.waveform = waveform_model

        self.data = None
        if data is not None:
            self.set_data(data)

    def set_data(self, data):
        assert data.size == self.waveform.Nt, "Data and waveform length must match."
        self.data = data

    @property
    def parameters(self):
        return self.waveform.parameters + ['noise_variance',]
    
    def log_likelihood(self, parameters):
        """
        Evaluate the likelihood of the waveform given the parameters and noise variance, under the assumption
        of stationary white Gaussian noise in the time domain.
        
        Args:
            parameters: Parameters for the waveform and noise models. See `self.waveform.parameters` for details.
        
        Returns:
            The log-likelihood given the waveform parameters and noise variance.
        """

        waveform = self.waveform.get_td_waveform(parameters)
        residual = waveform - self.data
        likelihood = -0.5 * (jnp.sum((residual ** 2) / parameters['noise_variance']) + jnp.log(2 * jnp.pi * parameters['noise_variance'] * self.waveform.Nt))
        return likelihood

class FDLikelihood(Likelihood):
    """
    Frequency-domain likelihood class.
    """

    waveform: Union[TDWaveform, FDWaveform]
    td_data: Optional[jnp.ndarray]
    fd_data: Optional[jnp.ndarray]
    phase_maximise: bool
    amplitude_maximise: bool
    estimate_noise: bool
    
    def __init__(self, waveform_model: TDWaveform, td_data: Optional[jnp.ndarray] = None, fd_data: Optional[jnp.ndarray] = None, phase_maximise:bool=False, amplitude_maximise:bool=False, estimate_noise:bool=False, fixed_parameters: Optional[dict] = None):
        """
        Args:
            waveform: An instance of a TDWaveform class.
            td_data: Optional time-domain data to compare against the waveform. If provided, must match the length of the waveform.
            fd_data: Optional frequency-domain data to compare against the waveform. If provided, must match the length of the waveform.
            phase_maximise: If True, maximise likelihood over a phase offset.
            amplitude_maximise: If True, maximise likelihood over an overall amplitude scaling.
            estimate_noise: If True, include the noise PSD as a parameter to be estimated. Cannot be used with semicoherent likelihood.
            fixed_parameters: Optional dictionary of parameters and values to keep fixed during likelihood evaluation.
        """
        super().__init__(fixed_parameters=fixed_parameters)
        self.waveform = waveform_model

        self.set_data(td_data=td_data, fd_data=fd_data)

        self.phase_maximise = phase_maximise
        self.amplitude_maximise = amplitude_maximise

        if estimate_noise and amplitude_maximise:
            raise ValueError("Cannot estimate noise and maximise over amplitude at the same time.")
    
        self.estimate_noise = estimate_noise

    def set_data(self, td_data=None, fd_data=None):
        if td_data is not None:
            self.fd_data = self.td_data_to_truncated_fd_data(td_data)
            self.td_data = td_data
        elif fd_data is not None:
            assert fd_data.size == self.waveform.Nf, "Data and waveform length must match."
            self.fd_data = fd_data
            self.td_data = None
        else:
            raise ValueError("Must provide either time-domain or frequency-domain data.")

    def td_data_to_truncated_fd_data(self, data):
        fd_data = jnp.fft.rfft(data) * self.waveform.dt
        f_min_ind = self.waveform.f_min / (self.waveform.df)
        return fd_data[int(f_min_ind):int(f_min_ind)+self.waveform.Nf]
        
    @property
    def parameters(self):
        return self.waveform.parameters + ['noise_psd',]
    
    def _inner(self, a, b, psd):
        if self.phase_maximise:
            return (4 / self.waveform.T) * jnp.abs(jnp.sum((a.conj() * b) / psd))
        else:
            return (4 / self.waveform.T) * (jnp.sum((a.conj() * b).real / psd))

    def log_likelihood(self, parameters):
        """
        Evaluate the likelihood of the waveform given the parameters and noise variance, under the assumption
        of stationary white Gaussian noise in the frequency domain.
        
        Args:
            parameters: Parameters for the waveform and noise models. See `self.waveform.parameters` for details.
        
        Returns:
            The log-likelihood given the waveform parameters and noise variance.
        """

        waveform = self.waveform.get_fd_waveform(parameters)

        d_h = self._inner(self.fd_data, waveform, parameters['noise_psd'])
        h_h = self._inner(waveform, waveform, parameters['noise_psd'])

        if self.amplitude_maximise:
            likelihood = d_h / h_h**0.5 / 2
        else:
            likelihood = d_h - 0.5 * h_h
            if self.estimate_noise:
                d_d = self._inner(self.fd_data, self.fd_data, parameters['noise_psd'])
                likelihood = likelihood -0.5 * (d_d + jnp.log(8 * jnp.pi * parameters['noise_psd']) * self.fd_data.size)

        return likelihood