from jax import jit
import jax.numpy as jnp

# @jit
# def small_angle(f: jnp.ndarray, parameters: dict, duration: float) -> jnp.ndarray:
#     beta = parameters['gamma']
#     omega = 2 * jnp.pi * f
#     omega_star = jnp.sqrt((2 * jnp.pi * parameters['f_N'])**2 - beta**2)
    
#     prefac = parameters['A'] / 2 * jnp.exp(1j * parameters['phi0'])

#     prefac1 = (-beta + 1j * (omega_star - omega))
#     num1 = prefac * (jnp.exp(duration * prefac1) - 1) / prefac1

#     prefac2 = (-beta - 1j * (omega_star + omega))
#     num2 = prefac.conjugate() * (jnp.exp(duration * prefac2) - 1) / prefac2

#     return num1 + num2

@jit
def small_angle(f: jnp.ndarray, parameters: dict, duration: float) -> jnp.ndarray:
    omega = 2 * jnp.pi * f
    omega_N = 2 * jnp.pi * parameters['f_N']
    g_omega = parameters['gamma'] + 1j * omega

    return parameters['A'] * jnp.exp(1j * parameters['phi0'] - g_omega * duration) * \
        (g_omega * (jnp.exp(g_omega * duration) - jnp.cos(omega_N * duration)) + omega_N * jnp.sin(omega_N * duration)) \
        / (g_omega ** 2 + omega_N ** 2)

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
    g_omega = parameters['gamma_1'] + 1j * omega

    out = parameters['A_1'] * jnp.exp(1j * parameters['phi0_1'] - g_omega * duration) * \
        (g_omega * (jnp.exp(g_omega * duration) - jnp.cos(omega_N * duration)) + omega_N * jnp.sin(omega_N * duration)) \
        / (g_omega ** 2 + omega_N ** 2)

    omega_N = 2 * jnp.pi * parameters['f_N_2']
    g_omega = parameters['gamma_2'] + 1j * omega

    out += parameters['A_2'] * jnp.exp(1j * parameters['phi0_2'] - g_omega * duration) * \
        (g_omega * (jnp.exp(g_omega * duration) - jnp.cos(omega_N * duration)) + omega_N * jnp.sin(omega_N * duration)) \
        / (g_omega ** 2 + omega_N ** 2)

    return out

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