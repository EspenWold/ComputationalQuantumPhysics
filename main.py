import os
from math import exp, sqrt
import numpy as np

from VariationalMonteCarlo.MCSampler import mc_sampler_1_dim, mc_sampler_any_dims
from plotting import energy_variance_plot, plot_energy_variance_2dims

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


def harmonic_oscillator_wf(r, alpha):
    return exp(-0.5 * alpha * alpha * r * r)


def harmonic_oscillator_local_energy(r, alpha):
    return 0.5 * r * r * (1 - alpha ** 4) + 0.5 * alpha * alpha


def q_dot_wf(r, variational_params):
    (alpha, beta) = variational_params
    r1 = r[0] ** 2 + r[1] ** 2
    r2 = r[2] ** 2 + r[3] ** 2
    r12 = sqrt((r[0] - r[2]) ** 2 + (r[1] - r[3]) ** 2)
    deno = r12 / (1 + beta * r12)
    return exp(-0.5 * alpha * (r1 + r2) + deno)


def q_dot_local_energy(r, variational_params):
    (alpha, beta) = variational_params
    r1 = (r[0] ** 2 + r[1] ** 2)
    r2 = (r[2] ** 2 + r[3] ** 2)
    r12 = sqrt((r[0] - r[2]) ** 2 + (r[1] - r[3]) ** 2)
    deno = 1.0 / (1 + beta * r12)
    deno2 = deno * deno
    return 0.5 * (1 - alpha * alpha) * (r1 + r2) + 2.0 * alpha + 1.0 / r12 + deno2 * (
            alpha * r12 - deno2 + 2 * beta * deno - 1.0 / r12)


def q_dot_q_force(r, variational_params):
    (alpha, beta) = variational_params
    qforce = np.zeros(r.shape, np.double)
    r12 = sqrt((r[0]-r[2])**2 + (r[1]-r[3])**2)
    deno = 1.0/(1+beta*r12)
    for i in range(r):
        qforce[i] = -2*r[i]*alpha*(r[i]-r[i])*deno*deno/r12
    return qforce


def q_dot_wf_OLD(r, alpha, beta):
    r1 = r[0, 0] ** 2 + r[0, 1] ** 2
    r2 = r[1, 0] ** 2 + r[1, 1] ** 2
    r12 = sqrt((r[0, 0] - r[1, 0]) ** 2 + (r[0, 1] - r[1, 1]) ** 2)
    deno = r12 / (1 + beta * r12)
    return exp(-0.5 * alpha * (r1 + r2) + deno)


def q_dot_local_energy_OLD(r, alpha, beta):
    r1 = (r[0, 0] ** 2 + r[0, 1] ** 2)
    r2 = (r[1, 0] ** 2 + r[1, 1] ** 2)
    r12 = sqrt((r[0, 0] - r[1, 0]) ** 2 + (r[0, 1] - r[1, 1]) ** 2)
    deno = 1.0 / (1 + beta * r12)
    deno2 = deno * deno
    return 0.5 * (1 - alpha * alpha) * (r1 + r2) + 2.0 * alpha + 1.0 / r12 + deno2 * (
            alpha * r12 - deno2 + 2 * beta * deno - 1.0 / r12)


# Here starts the main program with variable declarations
num_variations = 5
alpha_values = np.linspace(0.925, 1.4, num_variations)
beta_values = np.linspace(0.01, 0.2, num_variations)
alpha_mesh, beta_mesh = np.meshgrid(alpha_values, beta_values)
variational_values = np.column_stack((alpha_mesh.flatten(), beta_mesh.flatten()))

(energies, variances) = mc_sampler_any_dims(trial_wave_function=q_dot_wf,
                                            local_energy_function=q_dot_local_energy,
                                            system_dimensions=4,
                                            variational_values=variational_values,
                                            outfile=outfile)

outfile.close()

reshaped_energies = energies.reshape(num_variations, num_variations).T

plot_energy_variance_2dims(alpha_values=alpha_values,
                           beta_values=beta_values,
                           energies=reshaped_energies,
                           save_path=image_path(NAME_OF_SYSTEM))
