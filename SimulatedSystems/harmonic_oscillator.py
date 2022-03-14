from random import random, seed, normalvariate
from math import sqrt, exp
import numpy as np
from numpy import ndarray


def exact_energy_1d(alpha):
    return 0.5 * (1/(4*alpha) + alpha)


def exact_energy_2d(alpha):
    return 1/(4*alpha) + alpha


def exact_energy_3d(alpha):
    return 1.5 * (1/(4*alpha) + alpha)


def harmonic_oscillator_wf(pos, alpha):
    r2 = sum([coord**2 for coord in pos])
    return exp(- alpha * r2)


def harmonic_oscillator_local_energy(pos, alpha, dims):
    r2 = sum([coord**2 for coord in pos])
    return 0.5 * r2 * (1 - 4 * alpha ** 2) + dims * alpha


def harmonic_oscillator_quantum_force(r, alpha):
    return -4 * alpha * r


def mc_sampler_harmonic_oscillator_metropolis(alphas: ndarray,
                                              num_particles: int,
                                              dimensions_per_particle: int,
                                              importance_sampling: bool,
                                              outfile):
    (num_variations,) = alphas.shape
    values = np.zeros(num_variations)
    variances = np.zeros(num_variations)

    # Cycles and diffusion constants
    number_of_cycles = 10000
    D = 0.5
    time_step = 0.05

    new_position_per_particle: ndarray = np.zeros((num_particles, dimensions_per_particle), np.double)
    old_position_per_particle: ndarray = np.zeros((num_particles, dimensions_per_particle), np.double)
    if importance_sampling:
        new_qf_per_particle: ndarray = np.zeros((num_particles, dimensions_per_particle), np.double)
        old_qf_per_particle: ndarray = np.zeros((num_particles, dimensions_per_particle), np.double)
    new_wf_per_particle: ndarray = np.zeros(num_particles, np.double)
    old_wf_per_particle: ndarray = np.zeros(num_particles, np.double)

    seed()
    for n in range(len(alphas)):
        alpha = alphas[n]

        # Initial position
        for particle_index in range(num_particles):
            for dimension_index in range(dimensions_per_particle):
                old_position_per_particle[particle_index][dimension_index] = normalvariate(0.0, 1.0) * sqrt(time_step)

            old_wf_per_particle[particle_index] = harmonic_oscillator_wf(
                old_position_per_particle[particle_index], alpha)
            if importance_sampling:
                old_qf_per_particle[particle_index] = harmonic_oscillator_quantum_force(
                    old_position_per_particle[particle_index], alpha)

        energy_estimate = energy_squared_estimate = 0.0
        # Loop over MCMC cycles
        for cycle in range(number_of_cycles):

            # Trial position by single particle moves
            for pi in range(num_particles):
                for di in range(dimensions_per_particle):
                    new_position = old_position_per_particle[pi][di] + normalvariate(0.0, 1.0) * sqrt(time_step)
                    if importance_sampling:
                        new_position += old_qf_per_particle[pi][di] * time_step * D
                    new_position_per_particle[pi][di] = new_position

                new_wf_per_particle[pi] = harmonic_oscillator_wf(new_position_per_particle[pi], alpha)

                # Metropolis acceptance ratio
                acceptance_ratio = (new_wf_per_particle[pi] / old_wf_per_particle[pi]) ** 2

                # Add importance sampling greens function factor
                if importance_sampling:
                    new_qf_per_particle[pi] = harmonic_oscillator_quantum_force(new_position_per_particle[pi], alpha)
                    greens_function_exponent = 0
                    for di in range(dimensions_per_particle):
                        greens_function_exponent += 0.5 * (old_qf_per_particle[pi][di] + new_qf_per_particle[pi][di]) * (
                                old_position_per_particle[pi][di] - new_position_per_particle[pi][di] +
                                0.5 * D * time_step * (old_qf_per_particle[pi][di] - new_qf_per_particle[pi][di])
                        )
                    greens_function_factor = exp(greens_function_exponent)
                    acceptance_ratio *= greens_function_factor

                if random() < acceptance_ratio:
                    for di in range(dimensions_per_particle):
                        old_position_per_particle[pi][di] = new_position_per_particle[pi][di]
                        if importance_sampling:
                            old_qf_per_particle[pi][di] = new_qf_per_particle[pi][di]
                    old_wf_per_particle[pi] = new_wf_per_particle[pi]

            sampled_value = 0
            for pos in old_position_per_particle:
                sampled_value += harmonic_oscillator_local_energy(pos, alpha, dimensions_per_particle)
            energy_estimate += sampled_value
            energy_squared_estimate += sampled_value ** 2

        # We calculate mean, variance and error ...
        energy_estimate /= number_of_cycles
        energy_squared_estimate /= number_of_cycles
        variance = energy_squared_estimate - energy_estimate ** 2
        error = sqrt(variance / number_of_cycles)
        values[n] = energy_estimate
        variances[n] = variance

        output_string = '%f %f %f %f\n' % (alpha, energy_estimate, variance, error)
        # print(output_string)
        outfile.write(output_string)

    return values, variances
