import os
import time
import numpy as np
import pandas as pd
from SimulatedSystems.harmonic_oscillator import exact_energy_1d, exact_energy_2d, exact_energy_3d, \
    harmonic_oscillator_gradient_descent, exact_variance_1d, exact_variance_2d, exact_variance_3d, \
    mc_sampler_harmonic_oscillator, single_run_mcmc_sampler
from SimulatedSystems.interacting_case import single_run_interaction_mcmc_sampler, interaction_gradient_descent
from plotting import energy_variance_plot, plot_positions, plot_several_densities
from statistical_analysis import block

# NAME_OF_SYSTEM = "VMCHarmonic"
NAME_OF_SYSTEM = "VMCInteracting"

PROJECT_ROOT_DIR = "Results"
FIGURE_ID = "Results/" + NAME_OF_SYSTEM + "/FigureFiles"
DATA_ID = "Results/" + NAME_OF_SYSTEM + "/DataFiles"
ENERGY_SAMPLES_ID = "Results/" + NAME_OF_SYSTEM + "/EnergySamples"
POSITIONS = "Results/" + NAME_OF_SYSTEM + "/Positions"

if not os.path.exists(PROJECT_ROOT_DIR):
    os.mkdir(PROJECT_ROOT_DIR)

if not os.path.exists(FIGURE_ID):
    os.makedirs(FIGURE_ID)

if not os.path.exists(DATA_ID):
    os.makedirs(DATA_ID)

if not os.path.exists(ENERGY_SAMPLES_ID):
    os.makedirs(ENERGY_SAMPLES_ID)

if not os.path.exists(POSITIONS):
    os.makedirs(POSITIONS)


def image_path(fig_id):
    return os.path.join(FIGURE_ID, fig_id)


def data_path(dat_id):
    return os.path.join(DATA_ID, dat_id)


def raw_samples_path(dat_id):
    return os.path.join(ENERGY_SAMPLES_ID, dat_id)


def positions_path(dat_id):
    return os.path.join(POSITIONS, dat_id)


# Task B & C - Uncomment block and run to reproduce results from report
# Make sure to set NAME_OF_SYSTEM = "VMCHarmonic" in the top of this file
# (Default run takes about 15 minutes at 100k MCMC-cycles with both boolean arguments set to false)
# Argument 'importance_sampling' to the MC sampler toggles the use of importance sampling
# and 'numerical' toggles approximated local energy by numerical differentiation
# ------------------------------------------------
num_variations = 50
alpha_values = np.linspace(0.2, 1.2, num_variations)

particle_numbers = [1, 10, 100, 500]
dimension_numbers = [1, 2, 3]
num_cycles = 200000
for num_dimensions in dimension_numbers:
    for num_particles in particle_numbers:
        outfile = open(data_path(NAME_OF_SYSTEM + f'_{num_dimensions}d_{num_particles}p' + ".dat"), 'w')
        tic = time.perf_counter()
        (energies, variances) = mc_sampler_harmonic_oscillator(alphas=alpha_values,
                                                               num_particles=num_particles,
                                                               dimensions_per_particle=num_dimensions,
                                                               num_cycles=num_cycles,
                                                               numerical=False,
                                                               importance_sampling=False,
                                                               outfile=outfile)

        toc = time.perf_counter()
        print(
            f"Ran simulation for {num_particles} particle(s) in {num_dimensions} dimension(s) in {toc - tic:0.4f} seconds")
        outfile.close()

        exact_energies = num_particles * exact_energy_1d(alpha_values)
        exact_variance = num_particles * exact_variance_1d(alpha_values)
        if num_dimensions == 2:
            exact_energies = num_particles * exact_energy_2d(alpha_values)
            exact_variance = num_particles * exact_variance_2d(alpha_values)
        if num_dimensions == 3:
            exact_energies = num_particles * exact_energy_3d(alpha_values)
            exact_variance = num_particles * exact_variance_3d(alpha_values)

        energy_variance_plot(
            alpha_values=alpha_values,
            energies=energies,
            variances=variances,
            exact_energies=exact_energies,
            exact_variance=exact_variance,
            num_particles=num_particles,
            num_dimensions=num_dimensions,
            save_path=image_path(NAME_OF_SYSTEM))
# ------------------------------------------------

# Task D & E - Uncomment block and run to reproduce results from report
# Make sure to set NAME_OF_SYSTEM = "VMCHarmonic" in the top of this file
# ------------------------------------------------
# num_dimensions = 3
# num_particles = 500
#
# initial_alpha_guess = 1.5
# alpha = harmonic_oscillator_gradient_descent(initial_alpha=initial_alpha_guess,
#                                              num_particles=num_particles,
#                                              dimensions_per_particle=num_dimensions,
#                                              importance_sampling=True)
#
# samples_file_path = raw_samples_path(NAME_OF_SYSTEM + f'_{num_dimensions}d_{num_particles}p' + ".dat")
# outfile = open(samples_file_path, 'w')
#
# number_of_cycles = 2 ** 20
# single_run_mcmc_sampler(alpha=alpha,
#                         number_of_cycles=number_of_cycles,
#                         num_particles=num_particles,
#                         dimensions_per_particle=num_dimensions,
#                         importance_sampling=True,
#                         gradient=False,
#                         print_to_file=True,
#                         outfile=outfile)
# outfile.close()
#
# tic = time.perf_counter()
# x = np.loadtxt(samples_file_path)
# (mean, var) = block(x)
# std = np.sqrt(var)
# toc = time.perf_counter()
# print(f"Ran analysis with the Blocking method for {number_of_cycles} data points in {toc - tic:0.4f} seconds")
#
# results = {'Mean': [mean], 'STDev': [std]}
# frame = pd.DataFrame(results, index=['Values'])
# print(frame)

# ------------------------------------------------

# Task G - Uncomment block and run to reproduce results from report
# Make sure to set NAME_OF_SYSTEM = "VMCInteracting" in the top of this file
# ------------------------------------------------
# num_particles = 10
# elliptical = True
# interacting = True
#
# potential_str = 'elliptical_' if elliptical else 'spherical_'
# interaction_str = 'interacting' if interacting else 'non-interacting'
#
# experiment_string = f'_{3}d_{num_particles}p_' + potential_str + interaction_str + ".dat"
#
# initial_alpha_guess = 1.5
# alpha = interaction_gradient_descent(initial_alpha=initial_alpha_guess,
#                                      num_particles=num_particles,
#                                      importance_sampling=True,
#                                      interactions=interacting,
#                                      elliptical=elliptical)
#
# energy_samples_file_path = raw_samples_path(NAME_OF_SYSTEM + experiment_string)
# positions_file_path = positions_path(NAME_OF_SYSTEM + experiment_string)
# outfile = open(energy_samples_file_path, 'w')
# positions_file = open(positions_file_path, 'w')
#
# print("Running main simulation with alpha = ", alpha)
# number_of_cycles = 2 ** 19
# single_run_interaction_mcmc_sampler(alpha=alpha,
#                                     number_of_cycles=number_of_cycles,
#                                     num_particles=num_particles,
#                                     importance_sampling=True,
#                                     interactions=interacting,
#                                     elliptical=elliptical,
#                                     gradient=False,
#                                     print_to_file=True,
#                                     outfile=outfile,
#                                     positions_file=positions_file)
#
# outfile.close()
#
# tic = time.perf_counter()
# x = np.loadtxt(energy_samples_file_path)
# (mean, var) = block(x)
# std = np.sqrt(var)
# toc = time.perf_counter()
# print(f"Ran analysis with the Blocking method for {number_of_cycles} data points in {toc - tic:0.4f} seconds")
#
# results = {'Mean': [mean], 'STDev': [std]}
# frame = pd.DataFrame(results, index=['Values'])
# print(frame)

# Plotting from files
# el_str = 'elliptical_'
# sp_str = 'spherical_'
# int_str = 'interacting'
# nint_str = 'non-interacting'

# positions_non_int = np.loadtxt(positions_path(NAME_OF_SYSTEM + f'_{3}d_{num_particles}p_' + el_str + nint_str + ".dat"))
# positions_int = np.loadtxt(positions_path(NAME_OF_SYSTEM + f'_{3}d_{num_particles}p_' + el_str + int_str + ".dat"))
#
# plot_several_densities(position_vectors=[positions_non_int, positions_int], resolution=100, particles=num_particles, save_path=image_path("OB-density-nonint"))
