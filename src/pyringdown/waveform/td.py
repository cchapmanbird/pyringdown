from jax import jit
import jax.numpy as jnp

# @jit
# def small_angle(t: jnp.ndarray, parameters: dict) -> jnp.ndarray:
#     beta = parameters['b'] / 2
#     omega_star = jnp.sqrt((2 * jnp.pi * parameters['f_N'])**2 - beta**2)
#     return parameters['A'] * jnp.exp(- beta * t) * jnp.cos(omega_star * t + parameters['phi0'])


@jit
def small_angle(t: jnp.ndarray, parameters: dict) -> jnp.ndarray:
    omega = 2 * jnp.pi * parameters['f_N']
    # omega_star = jnp.sqrt(omega**2 - parameters['gamma']**2)
    return parameters['A'] * jnp.exp(- parameters['gamma'] * t) * jnp.cos(omega * t + parameters['phi0'])

small_angle.parameters = [
    'A',  # Amplitude
    'gamma',  # Damping factor in Hertz
    'f_N',  # Frequency in Hertz
    'phi0'  # Initial phase
]

@jit
def double_small_angle(t: jnp.ndarray, parameters: dict) -> jnp.ndarray:
    # beta = parameters['b_1'] / 2
    omega = 2 * jnp.pi * parameters['f_N_1']
    # omega_star = jnp.sqrt((2 * jnp.pi * parameters['f_N_1'])**2 - beta**2)
    out = parameters['A_1'] * jnp.exp(- parameters['gamma_1'] * t) * jnp.cos(omega * t + parameters['phi0_1'])

    # beta = parameters['b_2'] / 2
    omega = 2 * jnp.pi * parameters['f_N_2']
    # omega_star = jnp.sqrt((2 * jnp.pi * parameters['f_N_2'])**2 - beta**2)
    out += parameters['A_2'] * jnp.exp(- parameters['gamma_2'] * t) * jnp.cos(omega * t + parameters['phi0_2'])
    return out

double_small_angle.parameters = [
    'A_1',  # Amplitude
    'gamma_1',  # Damping factor
    'f_N_1',  # Natural frequency
    'phi0_1',  # Initial phase
    'A_2',  # Amplitude
    'gamma_2',  # Damping factor
    'f_N_2',  # Natural frequency
    'phi0_2'  # Initial phase
]