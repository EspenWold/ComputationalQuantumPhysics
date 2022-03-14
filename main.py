import os
import numpy as np
from SimulatedSystems.harmonic_oscillator import harmonic_oscillator_wf, harmonic_oscillator_local_energy, \
    harmonic_oscillator_quantum_force, mc_sampler_harmonic_oscillator_metropolis, exact_energy_1d, exact_energy_2d, \
    exact_energy_3d
from SimulatedSystems.two_particle_quantum_dot import q_dot_wf_squared, q_dot_local_energy, q_dot_q_force, WaveFunction, \
    LocalEnergy, QuantumForce
from VariationalMonteCarlo.MCSampler import mc_sampler_system_agnostic, mc_sampler_separable_wavefunction
from plotting import plot_energy_variance_2dims, create_gif, energy_variance_plot

NAME_OF_SYSTEM = "VMCHarmonic"
# NAME_OF_SYSTEM = "VMCQdotMetropolis"
# NAME_OF_SYSTEM = "MCMC_demo"

PROJECT_ROOT_DIR = "Results"
FIGURE_ID = "Results/FigureFiles"
DATA_ID = "Results/" + NAME_OF_SYSTEM

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


outfile = open(data_path(NAME_OF_SYSTEM + ".dat"), 'w')

# # Task B
num_variations = 50
alpha_values = np.linspace(0.2, 1.5, num_variations)
# beta_values = np.linspace(0., 0.5, num_variations)
# alpha_mesh, beta_mesh = np.meshgrid(alpha_values, beta_values)
# variational_values = np.column_stack((alpha_mesh.flatten()))

num_particles = 1
num_dimensions = 3
(energies, variances) = mc_sampler_harmonic_oscillator_metropolis(alphas=alpha_values,
                                                                  num_particles=num_particles,
                                                                  dimensions_per_particle=num_dimensions,
                                                                  importance_sampling=False,
                                                                  outfile=outfile)

outfile.close()

exact_energies = num_particles*exact_energy_3d(alpha_values)
energy_variance_plot(
    alpha_values=alpha_values,
    energies=energies,
    variances=variances,
    exact_energies=exact_energies,
    exact_variance=False,
    num_particles=num_particles,
    num_dimensions=num_dimensions,
    save_path=data_path(NAME_OF_SYSTEM))


# (energies, variances) = mc_sampler_separable_wavefunction(wavefunction_oracle=harmonic_oscillator_wf,
#                                                           local_energy_oracle=harmonic_oscillator_local_energy,
#                                                           quantum_force_oracle=harmonic_oscillator_quantum_force,
#                                                           dimensions_per_particle=1,
#                                                           num_particles=1,
#                                                           variational_values=variational_values,
#                                                           outfile=outfile)
# reshaped_energies = energies.reshape(num_variations, num_variations).T
# plot_energy_variance_2dims(alpha_values=alpha_values,
#                            beta_values=beta_values,
#                            energies=reshaped_energies,
#                            save_path=image_path(NAME_OF_SYSTEM))
