import numpy as np
import matplotlib.pyplot as plt

def initialize_population(num_particles, dim, bounds):
    """Utility function to initialize a population of particles."""
    population = []
    for _ in range(num_particles):
        position = np.array([np.random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)])
        population.append(position)
    return np.array(population)

def plot_contour(func, bounds, resolution=100):
    """Plot a contour map of a 2D function."""
    x = np.linspace(bounds[0][0], bounds[0][1], resolution)
    y = np.linspace(bounds[1][0], bounds[1][1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))
    plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar()
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Contour Plot of the Objective Function')
    plt.show()
