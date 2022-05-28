import os
import time
import numpy as np
import pandas as pd
from SimulatedSystems.harmonic_oscillator import exact_energy_1d, exact_energy_2d, exact_energy_3d, \
    harmonic_oscillator_gradient_descent, exact_variance_1d, exact_variance_2d, exact_variance_3d, \
    mc_sampler_harmonic_oscillator, single_run_mcmc_sampler
from SimulatedSystems.interacting_case import single_run_interaction_mcmc_sampler, interaction_gradient_descent
from VariationalMonteCarlo.MCSampler import gradient_descent, BMWaveFunctionModel, single_run_model_mcmc_sampler, \
    log_interaction_factor
from plotting import energy_variance_plot, plot_positions, plot_3d_densities, plot_2d_densities, plot_1d_densities
from statistical_analysis import block

# NAME_OF_SYSTEM = "VMCHarmonic"
# NAME_OF_SYSTEM = "VMCInteracting"
NAME_OF_SYSTEM = "VMC_RBM"

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
# num_variations = 50
# alpha_values = np.linspace(0.2, 1.2, num_variations)
#
# particle_numbers = [1, 10, 100, 500]
# dimension_numbers = [1, 2, 3]
# num_cycles = 200000
# for num_dimensions in dimension_numbers:
#     for num_particles in particle_numbers:
#         outfile = open(data_path(NAME_OF_SYSTEM + f'_{num_dimensions}d_{num_particles}p' + ".dat"), 'w')
#         tic = time.perf_counter()
#         (energies, variances) = mc_sampler_harmonic_oscillator(alphas=alpha_values,
#                                                                num_particles=num_particles,
#                                                                dimensions_per_particle=num_dimensions,
#                                                                num_cycles=num_cycles,
#                                                                numerical=False,
#                                                                importance_sampling=False,
#                                                                outfile=outfile)
#
#         toc = time.perf_counter()
#         print(
#             f"Ran simulation for {num_particles} particle(s) in {num_dimensions} dimension(s) in {toc - tic:0.4f} seconds")
#         outfile.close()
#
#         exact_energies = num_particles * exact_energy_1d(alpha_values)
#         exact_variance = num_particles * exact_variance_1d(alpha_values)
#         if num_dimensions == 2:
#             exact_energies = num_particles * exact_energy_2d(alpha_values)
#             exact_variance = num_particles * exact_variance_2d(alpha_values)
#         if num_dimensions == 3:
#             exact_energies = num_particles * exact_energy_3d(alpha_values)
#             exact_variance = num_particles * exact_variance_3d(alpha_values)
#
#         energy_variance_plot(
#             alpha_values=alpha_values,
#             energies=energies,
#             variances=variances,
#             exact_energies=exact_energies,
#             exact_variance=exact_variance,
#             num_particles=num_particles,
#             num_dimensions=num_dimensions,
#             save_path=image_path(NAME_OF_SYSTEM))
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
# num_particles = 500
# elliptical = True
# interacting = True
#
# potential_str = 'elliptical_' if elliptical else 'spherical_'
# interaction_str = 'interacting' if interacting else 'non-interacting'
#
# experiment_string = f'_{3}d_{num_particles}p_' + potential_str + interaction_str + ".dat"
#
# initial_alpha_guess = 1.0
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
# positions_file.close()
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
#
# # Plotting from files
# el_str = 'elliptical_'
# sp_str = 'spherical_'
# int_str = 'interacting'
# nint_str = 'non-interacting'
#
# positions_non_int = np.loadtxt(positions_path(NAME_OF_SYSTEM + f'_{3}d_{num_particles}p_' + el_str + nint_str + ".dat"))
# positions_int = np.loadtxt(positions_path(NAME_OF_SYSTEM + f'_{3}d_{num_particles}p_' + el_str + int_str + ".dat"))
# #
# plot_several_densities(position_vectors=[positions_non_int, positions_int], resolution=100, particles=num_particles,
#                        save_path=image_path("OB-density-nonint"))

# ------------------------------------------------


# Project 2

# Task B - D
# Make sure to set NAME_OF_SYSTEM = "VMC_RBM" in the top of this file
# ------------------------------------------------
num_particles = 2
dimensions = 2
num_hidden_nodes = 2
interacting = True
importance_sampling = True

interaction_str = 'interacting' if interacting else 'non-interacting'
experiment_string = f'_{dimensions}d_{num_particles}p_' + interaction_str + ".dat"

model = BMWaveFunctionModel(wf_squared=True,
                            sigma_squared=0.5,
                            num_particles=num_particles,
                            dimensions=dimensions,
                            num_hidden_nodes=num_hidden_nodes,
                            std_deviation=0.1)
print(f'Starting model optimisation with model parameters:')
print(model.get_parameters())

gradient_descent(wf_model=model, interactions=interacting, importance_sampling=importance_sampling)

energy_samples_file_path = raw_samples_path(NAME_OF_SYSTEM + experiment_string)
positions_file_path = positions_path(NAME_OF_SYSTEM + experiment_string)
outfile = open(energy_samples_file_path, 'w')
positions_file = open(positions_file_path, 'w')

num_cycles = 2 ** 19
print(f'Running main simulation with {num_cycles} cycles and model parameters:')
print(model.get_parameters())
energy_estimate, variational_gradient = single_run_model_mcmc_sampler(
    wf_model=model,
    number_of_cycles=num_cycles,
    gradient=True,
    interactions=interacting,
    importance_sampling=importance_sampling,
    print_to_file=True,
    outfile=outfile,
    positions_file=positions_file)

outfile.close()
positions_file.close()

tic = time.perf_counter()
x = np.loadtxt(energy_samples_file_path)
(mean, var) = block(x)
std = np.sqrt(var)
toc = time.perf_counter()
print(f"Ran analysis with the Blocking method for {num_cycles} data points in {toc - tic:0.4f} seconds")

results = {'Mean': [mean], 'STDev': [std]}
frame = pd.DataFrame(results, index=['Values'])
print(frame)

positions = np.loadtxt(positions_file_path)
comparison_string = f'_{2}d_{2}p_' + 'non-interacting' + ".dat"
comparison_positions = np.loadtxt(positions_path(NAME_OF_SYSTEM + comparison_string))
print("Plotting", experiment_string, " and ", comparison_string)
if dimensions == 1:
    plot_1d_densities(position_vectors=[positions, comparison_positions], resolution=100, particles=num_particles,
                      save_path=image_path("OB-density-nonint"))
elif dimensions == 2:
    plot_2d_densities(position_vectors=[positions, comparison_positions], resolution=100, particles=num_particles,
                      save_path=image_path("OB-density-nonint"))
elif dimensions == 3:
    plot_3d_densities(position_vectors=[positions, comparison_positions], resolution=100, particles=num_particles,
                      save_path=image_path("OB-density-nonint"))

# # 2-electron VMC code for 2dim quantum dot with importance sampling
# # Using gaussian rng for new positions and Metropolis- Hastings
# # Added restricted boltzmann machine method for dealing with the wavefunction
# # RBM code based heavily off of:
# # https://github.com/CompPhysics/ComputationalPhysics2/tree/gh-pages/doc/Programs/BoltzmannMachines/MLcpp/src/CppCode/ob
# from math import exp, sqrt
# from random import random, seed, normalvariate
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
# import sys
#
#
# # Trial wave function for the 2-electron quantum dot in two dims
# def WaveFunction(r, a, b, w):
#     sigma = 1.0
#     sig2 = sigma ** 2
#     Psi1 = 0.0
#     Psi2 = 1.0
#     Q = Qfac(r, b, w)
#
#     for iq in range(NumberParticles):
#         for ix in range(Dimension):
#             Psi1 += (r[iq, ix] - a[iq, ix]) ** 2
#
#     for ih in range(NumberHidden):
#         Psi2 *= (1.0 + np.exp(Q[ih]))
#
#     Psi1 = np.exp(-Psi1 / (2 * sig2))
#
#     return Psi1 * Psi2
#
#
# # Local energy  for the 2-electron quantum dot in two dims, using analytical local energy
# def LocalEnergy(r, a, b, w):
#     sigma = 1.0
#     sig2 = sigma ** 2
#     locenergy = 0.0
#
#     Q = Qfac(r, b, w)
#
#     for iq in range(NumberParticles):
#         for ix in range(Dimension):
#             sum1 = 0.0
#             sum2 = 0.0
#             for ih in range(NumberHidden):
#                 sum1 += w[iq, ix, ih] / (1 + np.exp(-Q[ih]))
#                 sum2 += w[iq, ix, ih] ** 2 * np.exp(Q[ih]) / (1.0 + np.exp(Q[ih])) ** 2
#
#             dlnpsi1 = -(r[iq, ix] - a[iq, ix]) / sig2 + sum1 / sig2
#             dlnpsi2 = -1 / sig2 + sum2 / sig2 ** 2
#             locenergy += 0.5 * (-dlnpsi1 * dlnpsi1 - dlnpsi2 + r[iq, ix] ** 2)
#
#     if (interaction == True):
#         for iq1 in range(NumberParticles):
#             for iq2 in range(iq1):
#                 distance = 0.0
#                 for ix in range(Dimension):
#                     distance += (r[iq1, ix] - r[iq2, ix]) ** 2
#
#                 locenergy += 1 / sqrt(distance)
#
#     return locenergy
#
#
# # Derivate of wave function ansatz as function of variational parameters
# def DerivativeWFansatz(r, a, b, w):
#     sigma = 1.0
#     sig2 = sigma ** 2
#
#     Q = Qfac(r, b, w)
#
#     WfDer = np.empty((3,), dtype=object)
#     WfDer = [np.copy(a), np.copy(b), np.copy(w)]
#
#     WfDer[0] = (r - a) / sig2
#     WfDer[1] = 1 / (1 + np.exp(-Q))
#
#     for ih in range(NumberHidden):
#         WfDer[2][:, :, ih] = w[:, :, ih] / (sig2 * (1 + np.exp(-Q[ih])))
#
#     return WfDer
#
#
# # Setting up the quantum force for the two-electron quantum dot, recall that it is a vector
# def QuantumForce(r, a, b, w):
#     sigma = 1.0
#     sig2 = sigma ** 2
#
#     qforce = np.zeros((NumberParticles, Dimension), np.double)
#     sum1 = np.zeros((NumberParticles, Dimension), np.double)
#
#     Q = Qfac(r, b, w)
#
#     for ih in range(NumberHidden):
#         sum1 += w[:, :, ih] / (1 + np.exp(-Q[ih]))
#
#     qforce = 2 * (-(r - a) / sig2 + sum1 / sig2)
#
#     return qforce
#
#
# def Qfac(r, b, w):
#     Q = np.zeros((NumberHidden), np.double)
#     temp = np.zeros((NumberHidden), np.double)
#
#     for ih in range(NumberHidden):
#         temp[ih] = (r * w[:, :, ih]).sum()
#
#     Q = b + temp
#
#     return Q
#
#
# # Computing the derivative of the energy and the energy
# def EnergyMinimization(a, b, w):
#     NumberMCcycles = 10000
#     # Parameters in the Fokker-Planck simulation of the quantum force
#     D = 0.5
#     TimeStep = 0.05
#     # positions
#     PositionOld = np.zeros((NumberParticles, Dimension), np.double)
#     PositionNew = np.zeros((NumberParticles, Dimension), np.double)
#     # Quantum force
#     QuantumForceOld = np.zeros((NumberParticles, Dimension), np.double)
#     QuantumForceNew = np.zeros((NumberParticles, Dimension), np.double)
#
#     # seed for rng generator
#     seed()
#     energy = 0.0
#     DeltaE = 0.0
#
#     EnergyDer = np.empty((3,), dtype=object)
#     DeltaPsi = np.empty((3,), dtype=object)
#     DerivativePsiE = np.empty((3,), dtype=object)
#     EnergyDer = [np.copy(a), np.copy(b), np.copy(w)]
#     DeltaPsi = [np.copy(a), np.copy(b), np.copy(w)]
#     DerivativePsiE = [np.copy(a), np.copy(b), np.copy(w)]
#     for i in range(3): EnergyDer[i].fill(0.0)
#     for i in range(3): DeltaPsi[i].fill(0.0)
#     for i in range(3): DerivativePsiE[i].fill(0.0)
#
#     # Initial position
#     for i in range(NumberParticles):
#         for j in range(Dimension):
#             PositionOld[i, j] = normalvariate(0.0, 1.0) * sqrt(TimeStep)
#     wfold = WaveFunction(PositionOld, a, b, w)
#     QuantumForceOld = QuantumForce(PositionOld, a, b, w)
#
#     # Loop over MC MCcycles
#     for MCcycle in range(NumberMCcycles):
#         # Trial position moving one particle at the time
#         for i in range(NumberParticles):
#             for j in range(Dimension):
#                 PositionNew[i, j] = PositionOld[i, j] + normalvariate(0.0, 1.0) * sqrt(TimeStep) + \
#                                     QuantumForceOld[i, j] * TimeStep * D
#             wfnew = WaveFunction(PositionNew, a, b, w)
#             QuantumForceNew = QuantumForce(PositionNew, a, b, w)
#
#             GreensFunction = 0.0
#             for j in range(Dimension):
#                 GreensFunction += 0.5 * (QuantumForceOld[i, j] + QuantumForceNew[i, j]) * \
#                                   (D * TimeStep * 0.5 * (QuantumForceOld[i, j] - QuantumForceNew[i, j]) - \
#                                    PositionNew[i, j] + PositionOld[i, j])
#
#             GreensFunction = exp(GreensFunction)
#             ProbabilityRatio = GreensFunction * wfnew ** 2 / wfold ** 2
#             # Metropolis-Hastings test to see whether we accept the move
#             if random() <= ProbabilityRatio:
#                 for j in range(Dimension):
#                     PositionOld[i, j] = PositionNew[i, j]
#                     QuantumForceOld[i, j] = QuantumForceNew[i, j]
#                 wfold = wfnew
#         # print("wf new:        ", wfnew)
#         # print("force on 1 new:", QuantumForceNew[0,:])
#         # print("pos of 1 new:  ", PositionNew[0,:])
#         # print("force on 2 new:", QuantumForceNew[1,:])
#         # print("pos of 2 new:  ", PositionNew[1,:])
#         DeltaE = LocalEnergy(PositionOld, a, b, w)
#         DerPsi = DerivativeWFansatz(PositionOld, a, b, w)
#
#         DeltaPsi[0] += DerPsi[0]
#         DeltaPsi[1] += DerPsi[1]
#         DeltaPsi[2] += DerPsi[2]
#
#         energy += DeltaE
#
#         DerivativePsiE[0] += DerPsi[0] * DeltaE
#         DerivativePsiE[1] += DerPsi[1] * DeltaE
#         DerivativePsiE[2] += DerPsi[2] * DeltaE
#
#     # We calculate mean values
#     energy /= NumberMCcycles
#     DerivativePsiE[0] /= NumberMCcycles
#     DerivativePsiE[1] /= NumberMCcycles
#     DerivativePsiE[2] /= NumberMCcycles
#     DeltaPsi[0] /= NumberMCcycles
#     DeltaPsi[1] /= NumberMCcycles
#     DeltaPsi[2] /= NumberMCcycles
#     EnergyDer[0] = 2 * (DerivativePsiE[0] - DeltaPsi[0] * energy)
#     EnergyDer[1] = 2 * (DerivativePsiE[1] - DeltaPsi[1] * energy)
#     EnergyDer[2] = 2 * (DerivativePsiE[2] - DeltaPsi[2] * energy)
#     return energy, EnergyDer
#
#
# # Here starts the main program with variable declarations
# NumberParticles = 2
# Dimension = 2
# NumberHidden = 10
#
# interaction = True
#
# # guess for parameters
# a = np.random.normal(loc=0.0, scale=0.01, size=(NumberParticles, Dimension))
# b = np.random.normal(loc=0.0, scale=0.01, size=(NumberHidden))
# w = np.random.normal(loc=0.0, scale=0.01, size=(NumberParticles, Dimension, NumberHidden))
# # Set up iteration using stochastic gradient method
# Energy = 0
# EDerivative = np.empty((3,), dtype=object)
# EDerivative = [np.copy(a), np.copy(b), np.copy(w)]
# # Learning rate eta, max iterations, need to change to adaptive learning rate
# eta = 0.001
# MaxIterations = 100
# iter = 0
# np.seterr(invalid='raise')
# Energies = np.zeros(MaxIterations)
#
# Energy, EDerivative = EnergyMinimization(a, b, w)
#
# while iter < MaxIterations:
#     Energy, EDerivative = EnergyMinimization(a, b, w)
#     agradient = EDerivative[0]
#     bgradient = EDerivative[1]
#     wgradient = EDerivative[2]
#     a -= eta * agradient
#     b -= eta * bgradient
#     w -= eta * wgradient
#     Energies[iter] = Energy
#     print("Energy:", Energy)
#
#     iter += 1
#
# import pandas as pd
#
# data = {
#     'Energy': Energies}
#
# frame = pd.DataFrame(data)
# print(frame)
