from random import random, seed
from math import sqrt
import numpy as np


def mc_sampler_1_dim(trial_wave_function, local_energy_function, max_variations, outfile):
    energies = np.zeros(max_variations)
    variances = np.zeros(max_variations)
    alpha_values = np.zeros(max_variations)

    num_mc_cycles = 100000
    step_size = 1.0

    seed()
    alpha = 0.4
    for ia in range(max_variations):
        alpha += .05
        alpha_values[ia] = alpha
        energy = energy_squared = 0.0
        # Initial position
        old_position = step_size * (random() - .5)
        old_wave_function = trial_wave_function(old_position, alpha)
        # Loop over MC cycles
        for cycle in range(num_mc_cycles):
            # Trial position
            new_position = old_position + step_size * (random() - .5)
            new_wave_function = trial_wave_function(new_position, alpha)
            # Metropolis test to see whether we accept the move
            if random() <= new_wave_function ** 2 / old_wave_function ** 2:
                old_position = new_position
                old_wave_function = new_wave_function
            local_energy = local_energy_function(old_position, alpha)
            energy += local_energy
            energy_squared += local_energy ** 2
        # We calculate mean, variance and error
        energy /= num_mc_cycles
        energy_squared /= num_mc_cycles
        variance = energy_squared - energy ** 2
        error = sqrt(variance / num_mc_cycles)
        energies[ia] = energy
        variances[ia] = variance
        outfile.write('%f %f %f %f \n' % (alpha, energy, variance, error))
    return energies, alpha_values, variances


def mc_sampler_2_dims(trial_wave_function, local_energy_function, num_particles, dimensions, max_variations, outfile):
    alpha_values = np.zeros(max_variations)
    beta_values = np.zeros(max_variations)
    energies = np.zeros((max_variations, max_variations))
    variances = np.zeros((max_variations, max_variations))

    NumberMCcycles = 25000
    StepSize = 1.0
    PositionOld = np.zeros((num_particles, dimensions), np.double)
    PositionNew = np.zeros((num_particles, dimensions), np.double)

    seed()
    alpha = 0.9
    for ia in range(max_variations):
        alpha += .025
        alpha_values[ia] = alpha
        beta = 0.0
        for jb in range(max_variations):
            beta += .01
            beta_values[jb] = beta
            energy = energy2 = 0.0
            # Initial position
            for i in range(num_particles):
                for j in range(dimensions):
                    PositionOld[i, j] = StepSize * (random() - .5)
            wfold = trial_wave_function(PositionOld, alpha, beta)

            # Loop over MC MCcycles
            for cycle in range(NumberMCcycles):
                # Trial position moving one particle at the time
                for i in range(num_particles):
                    for j in range(dimensions):
                        PositionNew[i, j] = PositionOld[i, j] + StepSize * (random() - .5)
                    wfnew = trial_wave_function(PositionNew, alpha, beta)

                    # Metropolis test to see whether we accept the move
                    if random() < wfnew ** 2 / wfold ** 2:
                        for j in range(dimensions):
                            PositionOld[i, j] = PositionNew[i, j]
                        wfold = wfnew
                DeltaE = local_energy_function(PositionOld, alpha, beta)
                energy += DeltaE
                energy2 += DeltaE ** 2
            # We calculate mean, variance and error ...
            energy /= NumberMCcycles
            energy2 /= NumberMCcycles
            variance = energy2 - energy ** 2
            error = sqrt(variance / NumberMCcycles)
            energies[ia, jb] = energy
            variances[ia, jb] = variance
            outfile.write('%f %f %f %f %f\n' % (alpha, beta, energy, variance, error))
    return energies, variances, alpha_values, beta_values


def mc_sampler_any_dims(trial_wave_function, local_energy_function, system_dimensions: int,
                        variational_values: np.array, outfile):
    (num_variations, variational_dimensions) = variational_values.shape
    energies = np.zeros(num_variations)
    variances = np.zeros(num_variations)

    number_of_cycles = 50000
    step_size = 1.0
    old_position = np.zeros(system_dimensions, np.double)
    new_position = np.zeros(system_dimensions, np.double)

    seed()
    for variational_index in range(num_variations):
        variational_vector = variational_values[variational_index]

        # Initial position
        for i in range(system_dimensions):
            old_position[i] = step_size * (random() - .5)
        wf_old = trial_wave_function(old_position, variational_vector)

        energy = energy2 = 0.0
        # Loop over MCMC cycles
        for cycle in range(number_of_cycles):
            # Trial position in configuration space
            for i in range(system_dimensions):
                new_position[i] = old_position[i] + step_size * (random() - .5)
            wf_new = trial_wave_function(new_position, variational_vector)

            # Metropolis test to see whether we accept the move
            if random() < wf_new ** 2 / wf_old ** 2:
                old_position = np.copy(new_position)
                wf_old = wf_new
            delta_e = local_energy_function(old_position, variational_vector)
            energy += delta_e
            energy2 += delta_e ** 2

        # We calculate mean, variance and error ...
        energy /= number_of_cycles
        energy2 /= number_of_cycles
        variance = energy2 - energy ** 2
        error = sqrt(variance / number_of_cycles)
        energies[variational_index] = energy
        variances[variational_index] = variance

        # output_string = ''.join(['%f ' % (var) for var in variational_vector]) + '|' + ' %f %f %f\n' % (energy, variance, error)
        # print(output_string)
        # outfile.write('%f %f %f %f %f\n' % ([var for var in variational_vector], energy, variance, error))

    return energies, variances
