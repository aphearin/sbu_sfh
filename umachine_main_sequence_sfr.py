"""
"""
import numpy as np


def mean_sfr(vmax, z, **kwargs):
    V = vmax/10**logV(z, **kwargs)
    v = vmax/V
    term1 = 1./(v**alpha(z, **kwargs) + v**beta(z, **kwargs))

    exp_arg = -np.log10(v)**2/(2.*delta(z, **kwargs))
    term2 = 10**logGamma(z, **kwargs)*np.exp(exp_arg)

    return 10**logEpsilon(z, **kwargs)*(term1 + term2)


def logV(z, logV_0=2.151, logV_a=-1.658, logV_lnz=1.68, logV_z=-0.233, **kwargs):
    a = 1./(1. + z)
    log10_V = logV_0 + logV_a*(1. - a) + logV_lnz*np.log(1.+z) + logV_z*z
    return log10_V


def logEpsilon(z, epsilon_0=0.109, epsilon_a=-3.441, epsilon_lnz=5.079, epsilon_z=-0.781, **kwargs):
    a = 1./(1. + z)
    log10_epsilon = epsilon_0 + epsilon_a*(1.-a) + epsilon_lnz*np.log(1.+z) + epsilon_z*z
    return log10_epsilon


def alpha(z, **kwargs):
    a = 1./(1. + z)
    return kwargs['alpha_0'] + kwargs['alpha_a']*(a - 1.) + kwargs['alpha_z']*z


def beta(z, **kwargs):
    a = 1./(1. + z)
    return kwargs['beta_0'] + kwargs['beta_a']*(a - 1.) + kwargs['beta_z']*z


def logGamma(z, **kwargs):
    a = 1./(1. + z)
    return kwargs['gamma_0'] + kwargs['gamma_a']*(a - 1.) + kwargs['gamma_z']*z


def delta(z, **kwargs):
    return kwargs['delta_0']


def scatter(vmax, z, **kwargs):
    return kwargs['sf_scatter']
