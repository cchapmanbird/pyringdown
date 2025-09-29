from nessai.model import Model
from .likelihood import Likelihood
import numpy as np

from functools import partial
from jax import vmap, jit
import jax.numpy as jnp

class NessaiModel(Model):

    def __init__(self, likelihood: Likelihood, bounds: dict):
        # Names of parameters to sample
        self.names = list(set(list(likelihood.parameters)) - set(list(likelihood.fixed_parameters.keys())))
        # Prior bounds for each parameter
        self.bounds = bounds

        # check that all parameters are accounted for
        if set(list(bounds.keys())) != set(self.names):
            raise ValueError("Parameters in likelihood and bounds do not match")
        
        self.likelihood = likelihood
        self._log_likelihood = vmap(jit(self.likelihood.__call__))

        self.vectorised_likelihood = True

    def log_prior(self, x):
        """
        Returns log of prior given a live point assuming uniform
        priors on each parameter.
        """
        # Check if values are in bounds, returns True/False
        # Then take the log to get 0/-inf and make sure the dtype is float
        log_p = np.log(self.in_bounds(x), dtype="float")
        # Iterate through each parameter (x and y)
        # since the live points are a structured array we can
        # get each value using just the name
        for n in self.names:
            log_p -= np.log(self.bounds[n][1] - self.bounds[n][0])
        return log_p

    def log_likelihood(self, x):
        """
        Returns log likelihood of given live points using the likelihood object.
        """
        return np.asarray(self._log_likelihood({n: jnp.atleast_1d(x[n]) for n in self.names}))
