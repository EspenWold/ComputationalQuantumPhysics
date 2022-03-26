import time
from random import random, seed, normalvariate, randrange
from math import sqrt, exp
import numpy as np
from numpy import ndarray


def exact_energy_1d(alpha):
    return 0.5 * (1 / (4 * alpha) + alpha)


def exact_variance_1d(alpha):
    return 0.5 * (1 / (4 * alpha) - alpha) ** 2


def exact_energy_2d(alpha):
    return 1 / (4 * alpha) + alpha


def exact_variance_2d(alpha):
    return (1 / (4 * alpha) - alpha) ** 2


def exact_energy_3d(alpha):
    return 1.5 * (1 / (4 * alpha) + alpha)


def exact_variance_3d(alpha):
    return 1.5 * (1 / (4 * alpha) - alpha) ** 2


def harmonic_oscillator_wf(pos, alpha):
    r2 = sum([coord ** 2 for coord in pos])
    return exp(- alpha * r2)


def harmonic_oscillator_local_energy(pos, alpha, dims):
    r2 = sum([coord ** 2 for coord in pos])
    return 0.5 * r2 * (1 - 4 * alpha ** 2) + dims * alpha


def harmonic_oscillator_local_energy_numerical(pos, alpha, dims, wf_0, h, h2_inverse):
    r2 = sum([coord ** 2 for coord in pos])
    tweaked_pos = pos.copy()
    kinetic_term = 2 * dims * wf_0
    for di in range(dims):
        tweaked_pos[di] += h
        kinetic_term -= harmonic_oscillator_wf(tweaked_pos, alpha)
        tweaked_pos[di] -= 2 * h
        kinetic_term -= harmonic_oscillator_wf(tweaked_pos, alpha)
        tweaked_pos[di] += h

    kinetic_term *= h2_inverse
    return 0.5 * (kinetic_term / wf_0 + r2)


def harmonic_oscillator_quantum_force(r, alpha):
    return -4 * alpha * r


def harmonic_oscillator_normalised_wf_gradient(pos):
    return - sum([coord ** 2 for coord in pos])


def mc_sampler_harmonic_oscillator(alphas: ndarray,
                                              num_particles: int,
                                              dimensions_per_particle: int,
                                              num_cycles: int,
                                              numerical: bool,
                                              importance_sampling: bool,
                                              outfile):
    (num_variations,) = alphas.shape
    values = np.zeros(num_variations)
    variances = np.zeros(num_variations)

    # Cycles and diffusion constants
    number_of_cycles = num_cycles
    D = 0.5
    time_step = 0.5

    # Parameters for numerical gradient
    h = 0.01
    h2_inverse = 1 / (h ** 2)

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

        # Initiate energy estimate over all particles
        sampled_energy_value = 0
        for (pi, pos) in enumerate(old_position_per_particle):
            if numerical:  # By numerical differentiation
                sampled_energy_value += harmonic_oscillator_local_energy_numerical(pos, alpha, dimensions_per_particle,
                                                                                   old_wf_per_particle[pi], h,
                                                                                   h2_inverse)
            else:  # By using the analytical expression for the local energy
                sampled_energy_value += harmonic_oscillator_local_energy(pos, alpha, dimensions_per_particle)

        # Loop over MCMC cycles
        count = 0
        for cycle in range(number_of_cycles):
            # Trial position by moving a single random particle
            pi = randrange(num_particles)

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
                    greens_function_exponent += 0.5 * (
                            old_qf_per_particle[pi][di] + new_qf_per_particle[pi][di]) * (
                                                        old_position_per_particle[pi][di] -
                                                        new_position_per_particle[pi][di] +
                                                        0.5 * D * time_step * (old_qf_per_particle[pi][di] -
                                                                               new_qf_per_particle[pi][di])
                                                )
                greens_function_factor = exp(greens_function_exponent)
                acceptance_ratio *= greens_function_factor

            if random() < acceptance_ratio:
                # Update single particle contribution to the energy
                if numerical:  # By numerical differentiation
                    sampled_energy_value -= harmonic_oscillator_local_energy_numerical(old_position_per_particle[pi],
                                                                                       alpha,
                                                                                       dimensions_per_particle,
                                                                                       old_wf_per_particle[pi], h,
                                                                                       h2_inverse)
                    sampled_energy_value += harmonic_oscillator_local_energy_numerical(new_position_per_particle[pi],
                                                                                       alpha,
                                                                                       dimensions_per_particle,
                                                                                       new_wf_per_particle[pi], h,
                                                                                       h2_inverse)
                else:  # By using the analytical expression for the local energy
                    sampled_energy_value -= harmonic_oscillator_local_energy(old_position_per_particle[pi], alpha,
                                                                             dimensions_per_particle)
                    sampled_energy_value += harmonic_oscillator_local_energy(new_position_per_particle[pi], alpha,
                                                                             dimensions_per_particle)

                # Update single particle position
                for di in range(dimensions_per_particle):
                    old_position_per_particle[pi][di] = new_position_per_particle[pi][di]
                    if importance_sampling:
                        old_qf_per_particle[pi][di] = new_qf_per_particle[pi][di]

                # Update single particle contribution to the wavefunction
                old_wf_per_particle[pi] = new_wf_per_particle[pi]
                count += 1

            energy_estimate += sampled_energy_value
            energy_squared_estimate += sampled_energy_value ** 2

        print('Acceptance rate', count / number_of_cycles)
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


def harmonic_oscillator_gradient_descent(initial_alpha: float,
                                         num_particles: int,
                                         dimensions_per_particle: int,
                                         importance_sampling: bool):
    learning_rate = 0.15

    # MCMC constants
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
    alpha = initial_alpha
    energy_estimate = energy_squared_estimate = 0.0
    norm_grad_estimate = grad_by_energy_estimate = 0.0
    for n in range(10):
        # Update our variational parameters by using the estimated gradient
        variational_gradient = 2 * (grad_by_energy_estimate - energy_estimate * norm_grad_estimate)
        alpha = alpha - learning_rate * variational_gradient

        # Initial position
        for particle_index in range(num_particles):
            for dimension_index in range(dimensions_per_particle):
                old_position_per_particle[particle_index][dimension_index] = normalvariate(0.0, 1.0) * sqrt(time_step)

            old_wf_per_particle[particle_index] = harmonic_oscillator_wf(
                old_position_per_particle[particle_index], alpha)
            if importance_sampling:
                old_qf_per_particle[particle_index] = harmonic_oscillator_quantum_force(
                    old_position_per_particle[particle_index], alpha)

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
                        greens_function_exponent += 0.5 * (old_qf_per_particle[pi][di] + new_qf_per_particle[pi][di]) \
                                                    * (old_position_per_particle[pi][di]
                                                       - new_position_per_particle[pi][di]
                                                       + 0.5 * D * time_step * (old_qf_per_particle[pi][di] -
                                                                                new_qf_per_particle[pi][di])
                                                       )
                    greens_function_factor = exp(greens_function_exponent)
                    acceptance_ratio *= greens_function_factor

                if random() < acceptance_ratio:
                    for di in range(dimensions_per_particle):
                        old_position_per_particle[pi][di] = new_position_per_particle[pi][di]
                        if importance_sampling:
                            old_qf_per_particle[pi][di] = new_qf_per_particle[pi][di]
                    old_wf_per_particle[pi] = new_wf_per_particle[pi]

            sampled_energy_value = 0
            sampled_gradient_value = 0
            for pos in old_position_per_particle:
                sampled_energy_value += harmonic_oscillator_local_energy(pos, alpha, dimensions_per_particle)
                sampled_gradient_value += harmonic_oscillator_normalised_wf_gradient(pos)
            energy_estimate += sampled_energy_value
            energy_squared_estimate += sampled_energy_value ** 2
            norm_grad_estimate += sampled_gradient_value
            grad_by_energy_estimate += sampled_energy_value * sampled_gradient_value

        # We calculate our estimates and variance
        energy_estimate /= number_of_cycles
        energy_squared_estimate /= number_of_cycles
        variance = energy_squared_estimate - energy_estimate ** 2
        norm_grad_estimate /= number_of_cycles
        grad_by_energy_estimate /= number_of_cycles

        print(alpha, variational_gradient, energy_estimate, norm_grad_estimate, grad_by_energy_estimate)

    return alpha
