import os
from math import exp, sqrt
import numpy as np

from SimulatedSystems.two_particle_quantum_dot import q_dot_wf_squared, q_dot_local_energy, q_dot_q_force
from VariationalMonteCarlo.MCSampler import mc_sampler_system_agnostic
from plotting import plot_energy_variance_2dims

# NAME_OF_SYSTEM = "VMCHarmonic"
NAME_OF_SYSTEM = "VMCQdotMetropolis"

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

# Here starts the main program with variable declarations
num_variations = 15
alpha_values = np.linspace(0.5, 2.5, num_variations)
beta_values = np.linspace(0., 0.5, num_variations)
alpha_mesh, beta_mesh = np.meshgrid(alpha_values, beta_values)
variational_values = np.column_stack((alpha_mesh.flatten(), beta_mesh.flatten()))

(energies, variances) = mc_sampler_system_agnostic(probability_oracle=q_dot_wf_squared,
                                                   value_oracle=q_dot_local_energy,
                                                   drift_vector_oracle=q_dot_q_force,
                                                   system_dimensions=4,
                                                   variational_values=variational_values,
                                                   outfile=outfile)

outfile.close()

reshaped_energies = energies.reshape(num_variations, num_variations).T

plot_energy_variance_2dims(alpha_values=alpha_values,
                           beta_values=beta_values,
                           energies=reshaped_energies,
                           save_path=image_path(NAME_OF_SYSTEM))
