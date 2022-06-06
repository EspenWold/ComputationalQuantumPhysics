# ComputationalQuantumPhysics

This repo contains the code for two numerical projects in the course Computational Quantum Physics at the University of Oslo in the spring of 2022. The second project builds on the first, and they are both concerned with using a Markov Chain Monte Carlo (MCMC) scheme to simulate the ground state wave function of a physical system. In short, we define a trial wave function with some number of variational paramters, perform a numerical integration of that wave function by MCMC and thus extract both an estimate for the ground state energy and gradients wrt. the variational parameters. The gradients allow us to minimise the energy by varying the paramters, which should let us find the ground state energy since that is the lowest possible energy of the system under any wave function.

## Project reports

The code in this repo will be much easier to understand by considering the projects for which it was written. Reports describing the projects in detail can be found in the pdf files `Project_1.pdf` and `Project_2.pdf`.

## Code structure

The natural entry point for this repo is the file `main.py` which serves as a control panel of sorts from which the simulations can be configured and run. It is divided in subsequent blocks that can be uncommented and run to perform the various simulations needed for the two projects. Actual code for the numerical integration and variational optimisation is relegated to the folder `SimulatedSystems` for project 1 and to `VariationalMonteCarlo` for project 2. There are multiple parallel implementations for project 1, each adapted to simulating a certain physical system. For project 2 there is only a single implementation because it is decoupled in a way that allows swapping out the physical system or wave function model but using the same code for numerical integration.

There are two auxillary files providing plotting services and statistical analysis of numerical results, `plotting.py` and `statistical_analysis.py`, repectively.

Lastly, there is the `Results` folder with various sub folders for the different simulations. These store figures and data produced by our simulations.
