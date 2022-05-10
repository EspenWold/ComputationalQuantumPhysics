from random import random, seed, normalvariate, randrange
from math import sqrt, exp
import numpy as np
from numba import jit


@jit(nopython=True)
def single_particle_wf_contribution(pos, alpha, beta, a, distances, interactions):
    single_particle_term = exp(- alpha * (pos[0] ** 2 + pos[1] ** 2 + beta * pos[2] ** 2))

    if not interactions:  # Ignore correlation terms
        return single_particle_term

    if min(distances) <= a:
        print("Particle collision! Zeroing out wavefunction")
        return 0

    correlation_term = np.prod(1 - a / distances)
    return single_particle_term * correlation_term


@jit(nopython=True)
def single_particle_local_energy_one_body_term(pos, alpha, beta, gamma):
    # Precompute a few things
    [x, y, z] = pos
    x2 = x ** 2
    y2 = y ** 2
    z2 = z ** 2

    return alpha * (2 + beta) - 2 * alpha ** 2 * (x2 + y2 + beta ** 2 * z2) + 0.5 * (x2 + y2 + gamma ** 2 * z2)


@jit(nopython=True)
def single_particle_local_energy_correlation_term(pos, alpha, beta, u_prime_terms, vector_terms):
    [x, y, z] = pos
    beta_i = np.array([x, y, beta * z])
    double_sum_correlation_term = 0
    single_sum_correlation_term = np.sum(0.5 * u_prime_terms ** 2)
    for j in range(len(vector_terms)):
        single_sum_correlation_term += 2 * alpha * np.dot(vector_terms[j], beta_i)
        for k in range(len(vector_terms)):
            double_sum_correlation_term += 0.5 * np.dot(vector_terms[j], vector_terms[k])

    return single_sum_correlation_term - double_sum_correlation_term


@jit(nopython=True)
def quantum_force(pos, alpha, beta, vector_terms, interactions):
    [x, y, z] = pos
    single_particle_term = -4 * alpha * np.array([x, y, beta * z])

    if not interactions:  # Ignore correlation terms
        return single_particle_term

    correlation_term = 2 * np.sum(vector_terms, axis=0)

    return single_particle_term + correlation_term


@jit(nopython=True)
def normalised_wf_gradient(pos, beta):
    (x, y, z) = pos
    return - (x ** 2 + y ** 2 + beta * z ** 2)


@jit(nopython=True)
def pick_out_precomputes_for_one_particle(precompute_matrix, particle_index):
    n = len(precompute_matrix) - 1
    output = np.zeros(n)
    for i in range(n):
        if i < particle_index:
            output[i] = precompute_matrix[i, particle_index]
        else:
            output[i] = precompute_matrix[particle_index, i + 1]

    return output


@jit(nopython=True)
def pick_out_vector_precomputes_for_one_particle(vector_matrix, particle_index):
    n = len(vector_matrix) - 1
    output = np.zeros((n, vector_matrix.shape[2]))
    for i in range(n):
        if i < particle_index:
            output[i] = vector_matrix[i, particle_index]
        else:
            output[i] = vector_matrix[particle_index, i + 1]

    return output


@jit(nopython=True)
def get_new_precomputes(new_particle_position, particle_index, old_position_per_particle, a):
    (particles, dims) = old_position_per_particle.shape
    n = particles - 1
    distances = np.zeros(n)
    u_primes = np.zeros(n)
    vector_terms = np.zeros((n, dims))
    for i in range(n):
        index = i if i < particle_index else i + 1
        vec = np.subtract(new_particle_position, old_position_per_particle[index])
        norm = np.linalg.norm(vec)
        distances[i] = norm
        u_primes[i] = (a / (norm - a)) / norm
        vector_terms[i] = u_primes[i] * vec / norm

    return distances, vector_terms, u_primes


def single_run_interaction_mcmc_sampler(alpha: float, number_of_cycles: int, num_particles: int,
                                        importance_sampling: bool, gradient: bool,
                                        interactions: bool, elliptical: bool,
                                        print_to_file: bool, outfile, positions_file):
    # MCMC constants
    D = 0.5
    time_step = 0.25

    # System constants
    dimensions_per_particle = 3
    a = 0.0043
    beta = gamma = 2.8284 if elliptical else 1

    # Instantiate needed arrays
    old_position_per_particle: np.ndarray = np.zeros((num_particles, dimensions_per_particle), np.double)
    if importance_sampling:
        new_qf_per_particle: np.ndarray = np.zeros((num_particles, dimensions_per_particle), np.double)
        old_qf_per_particle: np.ndarray = np.zeros((num_particles, dimensions_per_particle), np.double)
    new_wf_per_particle: np.ndarray = np.zeros(num_particles, np.double)
    old_wf_per_particle: np.ndarray = np.zeros(num_particles, np.double)

    single_particle_one_body_energies: np.ndarray = np.zeros(num_particles, np.double)
    single_particle_correlation_energies: np.ndarray = np.zeros(num_particles, np.double)
    single_particle_gradients: np.ndarray = np.zeros(num_particles, np.double)

    distance_matrix: np.ndarray = np.zeros((num_particles, num_particles), np.double)
    vector_term_matrix: np.ndarray = np.zeros((num_particles, num_particles, dimensions_per_particle), np.double)
    u_prime_matrix: np.ndarray = np.zeros((num_particles, num_particles), np.double)

    # Set random initial position
    for particle_index in range(num_particles):
        for dimension_index in range(dimensions_per_particle):
            old_position_per_particle[particle_index][dimension_index] = normalvariate(0.0, 1.0) * sqrt(time_step)

    # Update matrices with precomputed terms
    for i in range(num_particles):
        for j in range(i + 1, num_particles):
            vec = np.subtract(old_position_per_particle[i], old_position_per_particle[j])
            norm = np.linalg.norm(vec)
            distance_matrix[i][j] = norm
            u_prime_ij = (a / (norm - a)) / norm
            u_prime_matrix[i][j] = u_prime_ij
            vector_term_matrix[i][j] = u_prime_ij * vec / norm

    # Calculate initial wavefunction and quantum force
    for particle_index in range(num_particles):
        distances = pick_out_precomputes_for_one_particle(distance_matrix, particle_index)
        old_wf_per_particle[particle_index] = single_particle_wf_contribution(old_position_per_particle[particle_index],
                                                                              alpha, beta, a, distances, interactions)
        if importance_sampling:
            vector_terms = pick_out_vector_precomputes_for_one_particle(vector_term_matrix, particle_index)
            old_qf_per_particle[particle_index] = quantum_force(old_position_per_particle[particle_index], alpha, beta,
                                                                vector_terms, interactions)

    # Initiate energy estimate over all particles
    for particle_index in range(num_particles):
        single_particle_one_body_energies[particle_index] = single_particle_local_energy_one_body_term(
            old_position_per_particle[particle_index], alpha, beta, gamma)
        if interactions:
            u_prime_terms = pick_out_precomputes_for_one_particle(u_prime_matrix, particle_index)
            vector_terms = pick_out_vector_precomputes_for_one_particle(vector_term_matrix, particle_index)
            single_particle_correlation_energies[particle_index] = single_particle_local_energy_correlation_term(
                old_position_per_particle[particle_index], alpha, beta, u_prime_terms, vector_terms)

        if gradient:
            single_particle_gradients[particle_index] = normalised_wf_gradient(
                old_position_per_particle[particle_index], beta)

    energy_estimate = grad_by_energy_estimate = norm_grad_estimate = 0.0
    count = 0
    progress = 0
    seed()
    # Loop over MCMC cycles
    for cycle in range(number_of_cycles):

        # Trial position by moving a single random particle
        pi = randrange(num_particles)
        new_particle_position: np.ndarray = np.zeros(dimensions_per_particle, np.double)
        for di in range(dimensions_per_particle):
            new_position = old_position_per_particle[pi][di] + normalvariate(0.0, 1.0) * sqrt(time_step)
            if importance_sampling:
                new_position += old_qf_per_particle[pi][di] * time_step * D
            new_particle_position[di] = new_position

        [new_distances, new_vector_terms, new_u_primes] = get_new_precomputes(new_particle_position, pi,
                                                                              old_position_per_particle, a)
        new_wf_per_particle[pi] = single_particle_wf_contribution(new_particle_position,
                                                                  alpha, beta, a, new_distances, interactions)

        # Metropolis acceptance ratio
        acceptance_ratio = (new_wf_per_particle[pi] / old_wf_per_particle[pi]) ** 2

        # Add importance sampling greens function factor
        if importance_sampling:
            new_qf_per_particle[pi] = quantum_force(new_particle_position, alpha, beta, new_vector_terms, interactions)
            greens_function_exponent = 0
            for di in range(dimensions_per_particle):
                greens_function_exponent += 0.5 * (old_qf_per_particle[pi][di] + new_qf_per_particle[pi][di]) \
                                            * (old_position_per_particle[pi][di]
                                               - new_particle_position[di]
                                               + 0.5 * D * time_step * (old_qf_per_particle[pi][di] -
                                                                        new_qf_per_particle[pi][di])
                                               )
            greens_function_factor = exp(greens_function_exponent)
            acceptance_ratio *= greens_function_factor

        if random() < acceptance_ratio:
            single_particle_one_body_energies[pi] = single_particle_local_energy_one_body_term(
                new_particle_position, alpha, beta, gamma)
            if interactions:
                single_particle_correlation_energies[pi] = single_particle_local_energy_correlation_term(
                    new_particle_position, alpha, beta, new_u_primes, new_vector_terms)

            if gradient:
                single_particle_gradients[pi] = normalised_wf_gradient(old_position_per_particle[pi], beta)

            # Update single particle position
            for di in range(dimensions_per_particle):
                old_position_per_particle[pi][di] = new_particle_position[di]
                if importance_sampling:
                    old_qf_per_particle[pi][di] = new_qf_per_particle[pi][di]

            # Update matrices of precomputed values
            for i in range(num_particles - 1):
                if i < pi:
                    distance_matrix[i, pi] = new_distances[i]
                    vector_term_matrix[i, pi] = new_vector_terms[i]
                    u_prime_matrix[i, pi] = new_u_primes[i]
                else:
                    distance_matrix[pi, i + 1] = new_distances[i]
                    vector_term_matrix[pi, i + 1] = new_vector_terms[i]
                    u_prime_matrix[pi, i + 1] = new_u_primes[i]

            old_wf_per_particle[pi] = new_wf_per_particle[pi]
            count += 1

        energy_sample = np.sum(single_particle_one_body_energies) + np.sum(single_particle_correlation_energies)
        gradient_sample = np.sum(single_particle_gradients)

        progress += 1
        if print_to_file:
            if progress % 50000 == 0:
                print("--------------------------------")
                print("Progress: ", 100 * (progress / number_of_cycles), " percent")
                print("Energy estimate:", energy_estimate / progress)
                print("One body energy:", np.sum(single_particle_one_body_energies))
                print("Correlation energy:", np.sum(single_particle_correlation_energies))
            outfile.write('%f\n' % energy_sample)
            for i in range(num_particles):
                positions_file.write('%f %f %f\n' % (
                    old_position_per_particle[i][0], old_position_per_particle[i][1], old_position_per_particle[i][2]))

        energy_estimate += energy_sample
        norm_grad_estimate += gradient_sample
        grad_by_energy_estimate += energy_sample * gradient_sample

    print("Acceptance rate: ", count / number_of_cycles)
    return energy_estimate / number_of_cycles, \
           norm_grad_estimate / number_of_cycles, \
           grad_by_energy_estimate / number_of_cycles


def interaction_gradient_descent(initial_alpha: float,
                                 num_particles: int,
                                 importance_sampling: bool,
                                 interactions: bool,
                                 elliptical: bool):
    learning_rate = 5 * 10 ** (-2)
    if num_particles == 10:
        learning_rate = 3*10 ** (-2)
    if num_particles == 50:
        learning_rate = 3*10 ** (-3)
    if num_particles == 100:
        learning_rate = 3 * 10 ** (-4)
    if num_particles == 500:
        learning_rate = 3*10 ** (-5)

    number_of_cycles = 50000

    alpha = initial_alpha
    energy_estimate = norm_grad_estimate = grad_by_energy_estimate = 0.0
    for n in range(25):
        # Update our variational parameters by using the estimated gradient
        variational_gradient = 2 * (grad_by_energy_estimate - energy_estimate * norm_grad_estimate)
        alpha = alpha - learning_rate * variational_gradient
        print("Step ", n + 1)
        # Run a single round of MCMC
        energy_estimate, norm_grad_estimate, grad_by_energy_estimate = single_run_interaction_mcmc_sampler(
            alpha=alpha,
            number_of_cycles=number_of_cycles,
            num_particles=num_particles,
            importance_sampling=importance_sampling,
            gradient=True,
            interactions=interactions,
            elliptical=elliptical,
            print_to_file=False,
            outfile=False,  # OK because the outfile is never accessed with print set to false
            positions_file=False)  # OK because the outfile is never accessed with print set to false
        print("Alpha: ", alpha,
              " Energy: ", energy_estimate,
              " Gradient: ", 2 * (grad_by_energy_estimate - energy_estimate * norm_grad_estimate))

    return alpha
