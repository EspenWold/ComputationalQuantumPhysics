import os
import time
import numpy as np
from SimulatedSystems.harmonic_oscillator import exact_energy_1d, exact_energy_2d, exact_energy_3d, \
    harmonic_oscillator_gradient_descent, exact_variance_1d, exact_variance_2d, exact_variance_3d, \
    mc_sampler_harmonic_oscillator
from plotting import energy_variance_plot

NAME_OF_SYSTEM = "VMCHarmonic"

PROJECT_ROOT_DIR = "Results"
FIGURE_ID = "Results/" + NAME_OF_SYSTEM + "/FigureFiles"
DATA_ID = "Results/" + NAME_OF_SYSTEM + "/DataFiles"

if not os.path.exists(PROJECT_ROOT_DIR):
    os.mkdir(PROJECT_ROOT_DIR)

if not os.path.exists(FIGURE_ID):
    os.makedirs(FIGURE_ID)

if not os.path.exists(DATA_ID):
    os.makedirs(DATA_ID)


def image_path(fig_id):
    return os.path.join(FIGURE_ID, fig_id)


def data_path(dat_id):
    return os.path.join(DATA_ID, dat_id)


# Task B & C - Uncomment block and run to reproduce results from report
# (takes about 15 minutes at 100k MCMC-cycles)
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
                                                               importance_sampling=True,
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


