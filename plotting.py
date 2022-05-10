import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def energy_variance_plot(alpha_values, energies, variances, exact_energies,
                         exact_variance, num_particles, num_dimensions, save_path):
    plt.subplot(2, 1, 1)
    plt.plot(alpha_values, energies, 'o', alpha_values, exact_energies, 'r-')
    plt.title('Energy and variance - %i particle(s) in %i dimension(s)' % (num_particles, num_dimensions))
    plt.ylabel('Dimensionless energy')
    plt.subplot(2, 1, 2)
    plt.plot(alpha_values, variances, 'o')
    if len(exact_variance) > 0:
        plt.plot(alpha_values, exact_variance, 'r-')
    plt.xlabel(r'$\alpha$', fontsize=15)
    plt.ylabel('Variance')
    plt.savefig(save_path + f'_{num_dimensions}d_{num_particles}p' + ".png", format='png')
    plt.show()
    # data = {'Alpha': alpha_values, 'Energy': energies, 'Exact Energy': exact_energies, 'Variance': variances,
    #         'Exact Variance': exact_variance, }
    # frame = pd.DataFrame(data)
    # print(frame)


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


def surface_plot(matrix, **kwargs):
    (x, y) = np.meshgrid(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, matrix, **kwargs)
    return fig, ax, surf


def plot_several_densities(position_vectors, resolution, particles, save_path):
    clr = ['b', 'r', 'g', 'm']
    fig = plt.figure(1)
    for i, positions in enumerate(position_vectors):
        xy_dist = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
        xy_hist, xy_bins = np.histogram(xy_dist, bins=np.linspace(0, 2, resolution), density=True)
        z_hist, z_bins = np.histogram(positions[:, 2], bins=np.linspace(-2, 2, resolution), density=True)

        plt.subplot(2, 1, 1)
        plt.title(f"Simulated one-body density for {particles} particles")
        plt.xlabel("xy-distance from origin")
        plt.plot(xy_bins[0:-1], xy_hist, clr[i] + '.')
        plt.subplot(2, 1, 2)
        plt.xlabel("z")
        plt.plot(z_bins[0:-1], z_hist, clr[i] + '.')
    fig.tight_layout()
    plt.savefig(save_path + ".png", format='png')
    plt.show()


def plot_2d_densities(position_vectors, resolution, particles, save_path):
    clr = ['b', 'r', 'g', 'm']
    fig = plt.figure(1)
    for i, positions in enumerate(position_vectors):
        x_hist, x_bins = np.histogram(positions[:, 0], bins=np.linspace(-3, 3, resolution), density=True)
        y_hist, y_bins = np.histogram(positions[:, 1], bins=np.linspace(-3, 3, resolution), density=True)

        plt.subplot(2, 1, 1)
        plt.title(f"Simulated one-body density for {particles} particles")
        plt.xlabel("x")
        plt.plot(x_bins[0:-1], x_hist, clr[i] + '.')
        plt.subplot(2, 1, 2)
        plt.xlabel("y")
        plt.plot(y_bins[0:-1], y_hist, clr[i] + '.')
    fig.tight_layout()
    plt.savefig(save_path + ".png", format='png')
    plt.show()


def plot_positions(positions, resolution):
    max_value = np.max(positions) + 10 ** (-12)
    min_value = np.min(positions)
    length_unit = (max_value - min_value) / resolution
    # 1D bins
    x_bins = np.zeros(resolution, int)
    y_bins = np.zeros(resolution, int)
    z_bins = np.zeros(resolution, int)
    # 2D projection bins
    zx_bins = np.zeros((resolution, resolution), int)
    xy_bins = np.zeros((resolution, resolution), int)
    yz_bins = np.zeros((resolution, resolution), int)
    for p in positions:
        x_bin = int(np.floor((p[0] - min_value) / length_unit))
        y_bin = int(np.floor((p[1] - min_value) / length_unit))
        z_bin = int(np.floor((p[2] - min_value) / length_unit))
        x_bins[x_bin] += 1
        y_bins[y_bin] += 1
        z_bins[z_bin] += 1
        zx_bins[z_bin][x_bin] += 1
        xy_bins[x_bin][y_bin] += 1
        yz_bins[y_bin][z_bin] += 1

    plt.figure(1)
    plt.subplot(3, 1, 1)
    plt.plot(np.linspace(min_value, max_value, resolution), x_bins, 'b-')
    plt.subplot(3, 1, 2)
    plt.plot(np.linspace(min_value, max_value, resolution), y_bins, 'b-')
    plt.subplot(3, 1, 3)
    plt.plot(np.linspace(min_value, max_value, resolution), z_bins, 'b-')

    plt.figure(2)
    (x, y) = np.meshgrid(np.linspace(min_value, max_value, resolution), np.linspace(min_value, max_value, resolution))
    plt.subplot(131)
    plt.contour(x, y, zx_bins)
    plt.subplot(132)
    plt.contour(x, y, xy_bins)
    plt.subplot(133)
    plt.contour(x, y, yz_bins)
    # ax1 = fig.add_subplot(121, projection='3d')
    # ax1.plot_surface(x, y,  bins)

    plt.show()