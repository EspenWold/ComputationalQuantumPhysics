import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def energy_variance_plot(alpha_values, energies, variances, exact_energies, exact_variance, save_path):
    plt.subplot(2, 1, 1)
    plt.plot(alpha_values, energies, 'o-', alpha_values, exact_energies, 'r-')
    plt.title('Energy and variance')
    plt.ylabel('Dimensionless energy')
    plt.subplot(2, 1, 2)
    plt.plot(alpha_values, variances, '.-', alpha_values, exact_variance, 'r-')
    plt.xlabel(r'$\alpha$', fontsize=15)
    plt.ylabel('Variance')
    plt.savefig(save_path + ".png", format='png')
    plt.show()
    data = {'Alpha': alpha_values, 'Energy': energies, 'Exact Energy': exact_energies, 'Variance': variances,
            'Exact Variance': exact_variance, }
    frame = pd.DataFrame(data)
    print(frame)


def plot_energy_variance_2dims(alpha_values, beta_values, energies, save_path):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Plot the surface.
    X, Y = np.meshgrid(alpha_values, beta_values)
    surf = ax.plot_surface(X, Y, energies, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # Customize the z axis.
    zmin = np.matrix(energies).min()
    zmax = np.matrix(energies).max()
    ax.set_zlim(zmin, zmax)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\beta$')
    ax.set_zlabel(r'$\langle E \rangle$')
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig(save_path + ".png", format='png')
    plt.show()
