from random import random, seed, normalvariate, randrange
from math import sqrt, exp
import numpy as np
from numba import int32, boolean, float64
from numba.experimental import jitclass


# spec = [
#     ('sigma_squared', int32),
#     ('wf_squared', boolean),
#     ('num_particles', int32),
#     ('dimensions', int32),
#     ('a', float64[:]),
#     ('b', float64[:]),
#     ('w', float64[:, :]),
# ]
#
#
# @jitclass(spec)
class BMWaveFunctionModel:
    def __init__(self, wf_squared: bool, num_particles: int, dimensions: int, num_hidden_nodes: int):
        self.sigma_squared = 1
        self.wf_squared = wf_squared
        self.num_particles = num_particles
        self.dimensions = dimensions
        self.a: np.ndarray = np.random.normal(loc=0.0, scale=0.001, size=num_particles * dimensions)
        self.b: np.ndarray = np.random.normal(loc=0.0, scale=0.001, size=num_hidden_nodes)
        self.w: np.ndarray = np.random.normal(loc=0.0, scale=0.001, size=(num_particles * dimensions, num_hidden_nodes))

    def get_model_specs(self):
        num_parameters = self.a.size + self.b.size + self.w.size
        return self.num_particles, self.dimensions, num_parameters

    def get_parameters(self):
        return np.concatenate((self.a, self.b, self.w.flatten()))

    def set_parameters(self, flattened_parameter_array: np.ndarray):
        (coords, h_nodes) = self.w.shape
        [flat_a, flat_b, flat_w] = np.split(flattened_parameter_array, [coords, coords + h_nodes])
        self.a = np.copy(flat_a)
        self.b = np.copy(flat_b)
        self.w = np.copy(flat_w.reshape(self.w.shape))

    def get_wf_value(self, positions: np.ndarray):
        gaussian_factor_exponent = - 0.5 * np.linalg.norm((positions - self.a)) ** 2 / self.sigma_squared
        if self.wf_squared:
            gaussian_factor_exponent /= 2
        hidden_layer_exponent_vector = (self.b + np.matmul(positions, self.w) / self.sigma_squared)
        product_vector = 1 + np.exp(hidden_layer_exponent_vector)
        if self.wf_squared:
            product_vector = np.sqrt(product_vector)
        return np.exp(gaussian_factor_exponent) * np.prod(product_vector)

    def log_derivatives_position(self, positions: np.ndarray):
        xa_vec = (positions - self.a)
        exponent_vec = (self.b + np.matmul(positions, self.w) / self.sigma_squared)

        first_frac_vec = 1 / (1 + np.exp(-exponent_vec))
        first = (np.matmul(self.w, first_frac_vec) - xa_vec) / self.sigma_squared

        w_squared = np.square(self.w)
        second_frac_vec = np.exp(exponent_vec) / (1 + np.exp(exponent_vec)) ** 2
        second = -1 / self.sigma_squared + (np.matmul(w_squared, second_frac_vec)) / self.sigma_squared ** 2
        factor = 0.5 if self.wf_squared else 1

        return [factor * first, factor * second]

    def log_derivatives_parameters(self, positions: np.ndarray):
        a_derivatives = (positions - self.a) / self.sigma_squared
        b_derivatives = 1 / (1 + np.exp(-(self.b + np.matmul(positions, self.w) / self.sigma_squared)))
        w_derivatives = np.outer(positions / self.sigma_squared, b_derivatives)
        factor = 0.5 if self.wf_squared else 1
        return factor * np.concatenate([a_derivatives, b_derivatives, w_derivatives.flatten()])


def local_energy(wf_log_derivative, wf_log_second_derivative, positions, omega, particle_distances,
                 interaction):
    interaction_part = 0
    if interaction:
        interaction_part = np.sum(1 / particle_distances)
    potential_part = 0.5 * omega * np.sum(np.square(positions))
    kinetic_part = -0.5 * (np.sum(wf_log_second_derivative) + np.sum(np.square(wf_log_derivative)))
    return potential_part + kinetic_part + interaction_part
    # return 0.5 * (omega * np.sum(np.square(positions)) - np.sum(wf_log_second_derivative) - np.sum(np.square(
    #     wf_log_derivative))) + interaction_part


def single_run_model_mcmc_sampler(wf_model: BMWaveFunctionModel, number_of_cycles: int,
                                  gradient: bool, interactions: bool,
                                  print_to_file: bool, outfile, positions_file):
    # MCMC constants
    D = 0.5
    time_step = 0.5

    # System constants
    (num_particles, dimensions_per_particle, num_parameters) = wf_model.get_model_specs()
    omega = 1

    old_position_per_particle: np.ndarray = np.zeros((num_particles, dimensions_per_particle), np.double)
    new_position_per_particle: np.ndarray = np.zeros((num_particles, dimensions_per_particle), np.double)
    particle_distances: np.ndarray = np.zeros(int(0.5 * (num_particles - 1) * num_particles), np.double)

    # Set random initial position
    for particle_index in range(num_particles):
        for dimension_index in range(dimensions_per_particle):
            old_position_per_particle[particle_index][dimension_index] = normalvariate(0.0, 1.0) * sqrt(time_step)

    # Update matrix with initial particle distances
    index = 0
    for i in range(num_particles):
        for j in range(i + 1, num_particles):
            vec = np.subtract(old_position_per_particle[i], old_position_per_particle[j])
            particle_distances[index] = np.linalg.norm(vec)
            index += 1

    # Calculate initial wavefunction and quantum force
    old_wf_value = wf_model.get_wf_value(old_position_per_particle.flatten())
    old_quantum_force = 2 * wf_model.log_derivatives_position(old_position_per_particle.flatten())[0].reshape(
        num_particles, dimensions_per_particle)

    energy_sample = energy_estimate = 0.0
    gradient_sample = norm_grad_estimate = grad_by_energy_estimate = np.zeros(num_parameters)
    count = 0
    progress = 0
    seed()
    # Loop over MCMC cycles
    for cycle in range(number_of_cycles):

        # Trial position by moving all particles
        for pi in range(num_particles):
            for di in range(dimensions_per_particle):
                new_position_per_particle[pi][di] = old_position_per_particle[pi][di] + normalvariate(0.0, 1.0) * sqrt(
                    time_step) + old_quantum_force[pi][di] * time_step * D

        new_wf_value = wf_model.get_wf_value(new_position_per_particle.flatten())

        # Metropolis acceptance ratio
        acceptance_ratio = (new_wf_value / old_wf_value) ** 2

        # Add importance sampling greens function factor
        new_quantum_force = 2 * wf_model.log_derivatives_position(new_position_per_particle.flatten())[0].reshape(
            num_particles, dimensions_per_particle)

        greens_function_exponent = 0
        for pi in range(num_particles):
            for di in range(dimensions_per_particle):
                greens_function_exponent += 0.5 * (old_quantum_force[pi][di] + new_quantum_force[pi][di]) \
                                            * (old_position_per_particle[pi][di] - new_position_per_particle[pi][di]
                                               + 0.5 * D * time_step * (
                                                       old_quantum_force[pi][di] - new_quantum_force[pi][di]))
        greens_function_factor = exp(greens_function_exponent)
        acceptance_ratio *= greens_function_factor

        if random() < acceptance_ratio:
            old_position_per_particle = new_position_per_particle.copy()
            old_wf_value = new_wf_value.copy()
            old_quantum_force = new_quantum_force.copy()
            index = 0
            for i in range(num_particles):
                for j in range(i + 1, num_particles):
                    vec = np.subtract(old_position_per_particle[i], old_position_per_particle[j])
                    particle_distances[index] = np.linalg.norm(vec)
                    index += 1
            count += 1

        wf_derivatives = wf_model.log_derivatives_position(old_position_per_particle.flatten())
        energy_sample = local_energy(wf_derivatives[0], wf_derivatives[1],
                                     old_position_per_particle.flatten(), omega, particle_distances, interactions)

        if gradient:
            gradient_sample = wf_model.log_derivatives_parameters(old_position_per_particle.flatten())

        progress += 1
        if print_to_file:
            if progress % 50000 == 0:
                print("--------------------------------")
                print("Progress: ", 100 * (progress / number_of_cycles), " percent")
                print("Energy estimate:", energy_estimate / progress)
            outfile.write('%f\n' % energy_sample)
            for i in range(num_particles):
                string = ''
                for j in range(dimensions_per_particle):
                    string += '%f ' % old_position_per_particle[i][j]
                string += '\n'
                positions_file.write(string)

        energy_estimate += energy_sample
        norm_grad_estimate += gradient_sample
        grad_by_energy_estimate += energy_sample * gradient_sample

    print("Acceptance rate: ", count / number_of_cycles)
    return energy_estimate / number_of_cycles, \
           norm_grad_estimate / number_of_cycles, \
           grad_by_energy_estimate / number_of_cycles


def gradient_descent(wf_model: BMWaveFunctionModel, interactions: bool):
    learning_rate = 10 ** (-3)
    number_of_cycles = 10000

    (num_particles, dimensions_per_particle, num_parameters) = wf_model.get_model_specs()
    energy_estimate = 0.0
    norm_grad_estimate = grad_by_energy_estimate = np.zeros(num_parameters)
    for n in range(100):
        # Update our variational parameters by using the estimated gradient
        variational_gradient = 2 * (grad_by_energy_estimate - energy_estimate * norm_grad_estimate)
        wf_model.set_parameters(wf_model.get_parameters() - learning_rate * variational_gradient)
        print("Step ", n + 1)
        # Run a single round of MCMC
        energy_estimate, norm_grad_estimate, grad_by_energy_estimate = single_run_model_mcmc_sampler(
            wf_model=wf_model,
            number_of_cycles=number_of_cycles,
            gradient=True,
            interactions=interactions,
            print_to_file=False,
            outfile=False,  # OK because the outfile is never accessed with print set to false
            positions_file=False)  # OK because the outfile is never accessed with print set to false
        print("Energy: ", energy_estimate,
              " Gradient: ", np.linalg.norm(2 * (grad_by_energy_estimate - energy_estimate * norm_grad_estimate))
              )
        print(wf_model.get_parameters())

    return wf_model
