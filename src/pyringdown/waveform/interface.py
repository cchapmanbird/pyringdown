import equinox as eqx
import jax.numpy as jnp
from typing import Callable, Optional
from .utils import get_sampling_parameters_from_partial_inputs

class TDWaveform(eqx.Module):
    """General interface for waveform evaluation."""

    waveform_function: Callable
    Nt: int
    dt: float
    T: float
    Nf: int
    t: jnp.ndarray

    def __init__(self, waveform_function: Callable, dt: float, T: float):
        """
        Args:
            waveform_function (Callable): Function to evaluate the waveform. 
                Expected call signature is `waveform_function(t, parameters)`.
            dt (float): Time step size.
            T (float): Total duration of the waveform.
            **kwargs: Additional keyword arguments, for flexibility.
        """
        self.waveform_function = waveform_function

        self.Nt = int(T / dt)
        self.dt = dt
        self.T = T

        self.Nf = jnp.fft.rfftfreq(self.Nt, self.dt).size  # +1 to include f_max

        self.t = jnp.arange(self.Nt) * dt

    @property
    def parameters(self):
        """Return the parameters of the waveform function."""
        return self.waveform_function.parameters if hasattr(self.waveform_function, 'parameters') else []

    def get_td_waveform(self, parameters):
        return self.waveform_function(self.t, parameters)
    
    def get_fd_waveform(self, parameters):
        td_waveform = self.get_td_waveform(parameters)
        return jnp.fft.rfft(td_waveform) * self.dt


class FDWaveform(eqx.Module):
    """General interface for waveform evaluation."""

    waveform_function: Callable
    Nf: int
    df: float
    dt: float
    T: float
    f_min: float
    f_max: float
    f: jnp.ndarray
    Nf_full: int

    def __init__(
            self, 
            waveform_function: Callable, 
            f: Optional[jnp.ndarray] = None, 
            f_max: Optional[float] = None, 
            df: Optional[float] = None, 
            f_min: Optional[float] = 0., 
            dt: Optional[float] = None, 
            T: Optional[float] = None,
        ):
        """
        Args:
            waveform_function (Callable): Function to evaluate the waveform. 
                Expected call signature is `waveform_function(t, parameters)`.
            Nt (int): Number of time steps. Must be provided along with either `dt` or `T`.
            dt (float): Time step size. Must be provided along with either `Nt` or `T`.
            T (float): Total duration of the waveform. Must be provided along with either `Nt` or `dt`.
            **kwargs: Additional keyword arguments, for flexibility.
        """
        self.waveform_function = waveform_function

        if f is not None:
            assert jnp.all(jnp.diff(f) > 0), "Frequencies must be in ascending order."
            assert jnp.all(jnp.diff(f) == jnp.diff(f)[0]), "Frequencies must be evenly spaced."

            self.f = f
            self.f_min = f[0]
            self.f_max = f[-1]
            self.df = f[1] - f[0]
            self.Nf = f.size

            self.dt = dt if dt is not None else 1 / (2 * self.f_max)
            self.T = T if T is not None else 1 / self.df

        else:
            if (f_max is None) and (dt is None):
                raise ValueError("Must provide either f_max or dt if f is not provided.")

            self.df = df
            self.dt = dt if dt is not None else 1 / (2 * f_max)
            self.T = T if T is not None else 1 / self.df

            f_all = jnp.fft.rfftfreq(int(self.T / self.dt), self.dt)
            if f_min is None:
                self.f_min = f_all[0]
            else:        
                self.f_min = f_all[jnp.argmin(jnp.abs(f_all - f_min))]
            if f_max is None:
                self.f_max = f_all[-1]
            else:
                self.f_max = f_all[jnp.argmin(jnp.abs(f_all - f_max))]

            self.Nf = int(jnp.round((self.f_max - self.f_min) / self.df)) + 1  # +1 to include f_max
            self.f = jnp.linspace(self.f_min, self.f_max, self.Nf)

        self.Nf_full = jnp.fft.rfftfreq(int(self.T / self.dt), self.dt).size

    @property
    def parameters(self):
        """Return the parameters of the waveform function."""
        return self.waveform_function.parameters if hasattr(self.waveform_function, 'parameters') else []

    def get_fd_waveform(self, parameters):
        return self.waveform_function(self.f, parameters, self.T)
    
    def get_td_waveform(self, parameters):
        fd_waveform = self.get_fd_waveform(parameters)
        return jnp.fft.irfft(fd_waveform) / self.dt