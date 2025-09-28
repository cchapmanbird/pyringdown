from jax import jit
import jax.numpy as jnp

# @jit
# def small_angle(f: jnp.ndarray, parameters: dict, duration: float) -> jnp.ndarray:
#     omega = 2 * jnp.pi * f
#     omega_N = 2 * jnp.pi * parameters['f_N']
#     g_omega = parameters['gamma'] + 1j * omega

#     temp =  parameters['A'] / 2 * jnp.exp(1j * parameters['phi0'] - g_omega * duration) * \
#         (g_omega * (jnp.exp(g_omega * duration) - jnp.cos(omega_N * duration)) + omega_N * jnp.sin(omega_N * duration)) \
#         / (g_omega ** 2 + omega_N ** 2)
    
#     return temp + temp.conj()

@jit 
def _fd_sa_kern(om, omN, g, T, phi0):
    return (1 - jnp.exp(-(g - 1j * (om - omN)) * T)) / (g + 1j*(om - omN)) * jnp.exp(1j * phi0)

@jit
def small_angle(f: jnp.ndarray, parameters: dict, duration: float) -> jnp.ndarray:
    omega = 2 * jnp.pi * f
    omega_N = 2 * jnp.pi * parameters['f_N']
    g = parameters['gamma']
    phi0 = parameters['phi0']

    return parameters['A'] / 2  * (
        _fd_sa_kern(omega, omega_N, g, duration, phi0) + 
        _fd_sa_kern(-omega, omega_N, g, duration, phi0).conj()
    )
    
small_angle.parameters = [
    'A',  # Amplitude
    'gamma',  # Damping factor
    'f_N',  # Natural frequency
    'phi0'  # Initial phase'
]

@jit
def double_small_angle(f: jnp.ndarray, parameters: dict, duration: float) -> jnp.ndarray:
    omega = 2 * jnp.pi * f
    omega_N = 2 * jnp.pi * parameters['f_N_1']
    
    g = parameters['gamma_1']
    phi0 = parameters['phi0_1']

    temp = parameters['A_1'] / 2  * (
        _fd_sa_kern(omega, omega_N, g, duration, phi0) + 
        _fd_sa_kern(-omega, omega_N, g, duration, phi0).conj()
    )

    omega_N = 2 * jnp.pi * parameters['f_N_2']
    g = parameters['gamma_2']
    phi0 = parameters['phi0_2']

    return temp + parameters['A_2'] / 2  * (
            _fd_sa_kern(omega, omega_N, g, duration, phi0) + 
            _fd_sa_kern(-omega, omega_N, g, duration, phi0).conj()
        )

double_small_angle.parameters = [
    'A_1',  # Amplitude
    'gamma_1',  # Damping factor
    'f_N_1',  # Natural frequency
    'phi0_1',  # Initial phase'
    'A_2',  # Amplitude
    'gamma_2',  # Damping factor
    'f_N_2',  # Natural frequency
    'phi0_2'  # Initial phase'
]