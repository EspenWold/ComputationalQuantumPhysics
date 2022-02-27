from math import exp


def harmonic_oscillator_wf(r, alpha):
    return exp(-0.5 * alpha * alpha * r * r)


def harmonic_oscillator_local_energy(r, alpha):
    return 0.5 * r * r * (1 - alpha ** 4) + 0.5 * alpha * alpha
