import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def energy_variance_plot(alpha_values, energies, variances, exact_energies,
                         exact_variance, num_particles, num_dimensions, save_path):
    plt.subplot(2, 1, 1)
    plt.plot(alpha_values, energies, 'o-', alpha_values, exact_energies, 'r-')
    plt.title('Energy and variance - %i particle(s) in %i dimension(s)' % (num_particles, num_dimensions))
    plt.ylabel('Dimensionless energy')
    plt.subplot(2, 1, 2)
    plt.plot(alpha_values, variances, '.-')
    if exact_variance:
        plt.plot(alpha_values, exact_variance, 'r-')
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


def create_gif(data_path):
    d = 10

    def init():
        ax.set_xlim(-2*d, d)
        ax.set_ylim(-d, d)

    def update(pos):
        pos_x.append(pos[0])
        pos_y.append(pos[1])

        ln1.set_data(pos_x[-10:], pos_y[-10:])
        ln2.set_data(pos_x, pos_y)

    data = []
    with open(data_path) as f:
        count = 0
        for line in f:
            coords = line.split(' ')
            data.append([float(coords[0]), float(coords[1]), float(coords[2])])
            count += 1

    fig, ax = plt.subplots()
    pos_x, pos_y = [], []

    ln2, = plt.plot([], [], 'b,')
    ln1, = plt.plot([], [], 'r-o', markersize=1)

    ani = FuncAnimation(fig, update, [pos for pos in data], init_func=init)
    # writer = PillowWriter(fps=25)
    # ani.save("gaussian.gif", writer=writer)
    writer = FFMpegWriter(fps=25)
    ani.save('gaussian.mp4', writer=writer)
