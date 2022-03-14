from random import random, seed, normalvariate
from math import sqrt, exp
import numpy as np
from numpy import ndarray


def mc_sampler_separable_wavefunction(wavefunction_oracle,
                                      local_energy_oracle,
                                      quantum_force_oracle,
                                      dimensions_per_particle: int,
                                      num_particles: int,
                                      variational_values: ndarray,
                                      outfile):
    (num_variations, variational_dimensions) = variational_values.shape
    values = np.zeros(num_variations)
    variances = np.zeros(num_variations)

    # Cycles and diffusion constants
    number_of_cycles = 10000
    D = 0.5
    time_step = 0.05

    old_position_per_particle: ndarray = np.zeros((num_particles, dimensions_per_particle), np.double)
    new_position_per_particle: ndarray = np.zeros((num_particles, dimensions_per_particle), np.double)

    seed()
    for variational_index in range(num_variations):
        variational_vector = variational_values[variational_index]

        # Initial position
        for particle_index in range(num_particles):
            for dimension_index in range(dimensions_per_particle):
                old_position_per_particle[particle_index][dimension_index] = normalvariate(0.0, 1.0) * sqrt(time_step)
        old_wavefunction = wavefunction_oracle(old_position_per_particle, variational_vector)
        old_quantum_force = quantum_force_oracle(old_position_per_particle, variational_vector)

        energy_estimate = energy_squared_estimate = 0.0
        # Loop over MCMC cycles
        for cycle in range(number_of_cycles):

            # Trial position by single particle moves
            for pi in range(num_particles):
                for di in range(dimensions_per_particle):
                    new_position_per_particle[pi][di] = old_position_per_particle[pi][di] \
                                                        + normalvariate(0.0, 1.0) * sqrt(time_step) \
                                                        + old_quantum_force[pi][di] * time_step * D

                new_wavefunction = wavefunction_oracle(new_position_per_particle, variational_vector)
                new_quantum_force = quantum_force_oracle(new_position_per_particle, variational_vector)

                greens_function_exponent = 0
                for di in range(dimensions_per_particle):
                    greens_function_exponent += 0.5 * (old_quantum_force[pi][di] + new_quantum_force[pi][di]) * (
                            old_position_per_particle[pi][di] - new_position_per_particle[pi][di] +
                            0.5 * D * time_step * (old_quantum_force[pi][di] - new_quantum_force[pi][di])
                    )

                greens_function_factor = exp(greens_function_exponent)

                # Metropolis-Hastings test to see whether we accept the move
                if random() < greens_function_factor * (new_wavefunction**2 / old_wavefunction**2):
                    for di in range(dimensions_per_particle):
                        old_position_per_particle[pi][di] = new_position_per_particle[pi][di]
                        old_quantum_force[pi][di] = new_quantum_force[pi][di]
                    old_wavefunction = new_wavefunction

            sampled_value = local_energy_oracle(old_position_per_particle, variational_vector)
            energy_estimate += sampled_value
            energy_squared_estimate += sampled_value ** 2

        # We calculate mean, variance and error ...
        energy_estimate /= number_of_cycles
        energy_squared_estimate /= number_of_cycles
        variance = energy_squared_estimate - energy_estimate ** 2
        error = sqrt(variance / number_of_cycles)
        values[variational_index] = energy_estimate
        variances[variational_index] = variance

        output_string = ''.join(['%f ' % (var) for var in variational_vector]) + '|' + ' %f %f %f\n' % (
            energy_estimate, variance, error)
        # print(output_string)
        outfile.write(output_string)

    return values, variances


def mc_sampler_system_agnostic(probability_oracle,
                               value_oracle,
                               drift_vector_oracle,
                               system_dimensions: int,
                               variational_values: ndarray,
                               outfile):
    (num_variations, variational_dimensions) = variational_values.shape
    values = np.zeros(num_variations)
    variances = np.zeros(num_variations)

    # Cycles and diffusion constants
    number_of_cycles = 10000
    D = 0.5
    time_step = 5

    old_position: ndarray = np.zeros(system_dimensions, np.double)
    new_position: ndarray = np.zeros(system_dimensions, np.double)

    seed()
    accepted_moves = 0
    for variational_index in range(num_variations):
        variational_vector = variational_values[variational_index]

        # Initial position
        for i in range(system_dimensions):
            old_position[i] = normalvariate(0.0, 1.0) * sqrt(time_step)
        old_position = np.array([-15, 5])
        old_probability = probability_oracle(old_position, variational_vector)
        old_drift_vector = drift_vector_oracle(old_position, variational_vector)

        value_estimate = value_squared_estimate = 0.0
        # Loop over MCMC cycles
        for cycle in range(number_of_cycles):

            # Trial position in configuration space
            for i in range(system_dimensions):
                new_position[i] = old_position[i] + normalvariate(0.0, 1.0) * \
                                  sqrt(time_step) \
                                  + old_drift_vector[i] * time_step * D

            new_probability = probability_oracle(new_position, variational_vector)
            new_drift_vector = drift_vector_oracle(new_position, variational_vector)

            greens_function_exponent = 0
            for i in range(system_dimensions):
                greens_function_exponent += \
                    0.5 * (old_drift_vector[i] + new_drift_vector[i]) * (
                            old_position[i] - new_position[i] +
                            0.5 * D * time_step * (old_drift_vector[i] - new_drift_vector[i])
                    )

            greens_function_factor = exp(greens_function_exponent)

            # Metropolis test to see whether we accept the move
            if random() < greens_function_factor * (new_probability / old_probability):
                old_position = np.copy(new_position)
                old_probability = new_probability
                old_drift_vector = np.copy(new_drift_vector)
                accepted_moves += 1

            sampled_value = value_oracle(old_position, variational_vector)
            value_estimate += sampled_value
            value_squared_estimate += sampled_value ** 2

        print("Acceptance rate:", accepted_moves/number_of_cycles)
        # We calculate mean, variance and error ...
        value_estimate /= number_of_cycles
        value_squared_estimate /= number_of_cycles
        variance = value_squared_estimate - value_estimate ** 2
        error = sqrt(variance / number_of_cycles)
        values[variational_index] = value_estimate
        variances[variational_index] = variance

        output_string = ''.join(['%f ' % (var) for var in variational_vector]) + '|' + ' %f %f %f\n' % (
            value_estimate, variance, error)
        # print(output_string)
        outfile.write(output_string)

    return values, variances
