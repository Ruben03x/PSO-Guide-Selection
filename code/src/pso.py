import numpy as np  # For efficient numerical operations on arrays
import random  # For generating random numbers
from src import benchmark_functions, guide_strategies, utils  # Import benchmark functions and guide strategies for PSO

# Define a Particle class, representing a candidate solution in the search space.
class Particle:
    def __init__(self, dim, bounds):
        """
        Initializes a Particle instance.
        
        Parameters:
        - dim (int): Dimensionality of the search space.
        - bounds (list of tuples): Bounds for each dimension as (min, max).
        """
        # Initialize particle's position with random values within the given bounds
        self.position = np.array([random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)])
        # Initialize velocity as a zero vector for each dimension
        self.velocity = np.zeros(dim)
        # Store the best position this particle has visited (personal best or pBest)
        self.best_position = np.copy(self.position)
        # Initialize the best value (fitness) to infinity since we are minimizing
        self.best_value = float('inf')

# Define the PSO (Particle Swarm Optimization) class which handles the swarm optimization process.
class PSO:
    def __init__(self, func, bounds, dim, num_particles=30, max_iter=100, 
                 w=0.7, c1=1.4, c2=1.4, guide_strategy='elitist', cooling_factor=0.99):
        """
        Initializes the PSO optimizer.
        
        Parameters:
        - func (callable): The objective function to minimize.
        - bounds (list of tuples): Search space bounds for each dimension.
        - dim (int): Number of dimensions.
        - num_particles (int): Number of particles in the swarm.
        - max_iter (int): Maximum number of iterations.
        - w (float): Inertia weight to control the influence of previous velocity.
        - c1 (float): Cognitive constant, weighing the personal best contribution.
        - c2 (float): Social constant, weighing the guide's contribution.
        - guide_strategy (str): Strategy to select the guiding position (e.g., elitist, simulated_annealing, roulette).
        - cooling_factor (float): Factor for exponential cooling (recommended range 0.95–0.99 per Engelbrecht, 2007).
        """
        self.func = func
        self.bounds = bounds
        self.dim = dim
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive coefficient
        self.c2 = c2  # Social coefficient
        self.guide_strategy = guide_strategy  # Strategy to guide velocity update

        # Initialize the swarm with Particle instances
        self.swarm = [Particle(dim, bounds) for _ in range(num_particles)]
        # Initialize global best position found so far; start with a zero vector.
        self.global_best_position = np.zeros(dim)
        # Initialize global best value (fitness) to infinity
        self.global_best_value = float('inf')
        # Temperature parameter for simulated annealing based strategies (if used)
        self.temperature = 1.0
        # Cooling factor for simulated annealing. According to Engelbrecht (2007), a value between 0.95 and 0.99 is effective.
        self.cooling_factor = cooling_factor

    def update_global_best(self):
        """
        Updates each particle's personal best (pBest) and the overall global best (gBest) based on the objective function.
        """
        for particle in self.swarm:
            value = self.func(particle.position)
            if value < particle.best_value:
                particle.best_value = value
                particle.best_position = np.copy(particle.position)
        best_particle = min(self.swarm, key=lambda p: p.best_value)
        self.global_best_position = np.copy(best_particle.best_position)
        self.global_best_value = best_particle.best_value

    def select_guide(self, particle_index):
        """
        Selects a guide (target) position for a given particle based on the specified guide strategy.
        
        Parameters:
        - particle_index (int): Index of the current particle.
        
        Returns:
        - A numpy array representing the guiding position.
        """
        pbests = [p.best_position for p in self.swarm]
        fitnesses = [self.func(p.best_position) for p in self.swarm]

        if self.guide_strategy == 'elitist':
            best_index = np.argmin(fitnesses)
            return pbests[best_index]
        elif self.guide_strategy == 'simulated_annealing':
            candidate = self.global_best_position
            candidate_fit = self.func(candidate)
            current_fit = self.func(self.swarm[particle_index].best_position)
            if guide_strategies.simulated_annealing_acceptance(current_fit, candidate_fit, self.temperature):
                return candidate
            else:
                return self.swarm[particle_index].best_position
        elif self.guide_strategy == 'roulette':
            return guide_strategies.roulette_wheel_selection(pbests, fitnesses)
        elif self.guide_strategy == 'tournament':
            return guide_strategies.tournament_selection(pbests, fitnesses, tournament_size=3)
        elif self.guide_strategy == 'rank':
            return guide_strategies.rank_based_selection(pbests, fitnesses)
        else:
            best_index = np.argmin(fitnesses)
            return pbests[best_index]

    def optimize(self, track_diversity=False):
        diversity_history = []

        for t in range(self.max_iter):
            self.update_global_best()
            if track_diversity:
                mean_position = np.mean([p.position for p in self.swarm], axis=0)
                diversity = np.mean([np.linalg.norm(p.position - mean_position) for p in self.swarm])
                diversity_history.append(diversity)
            for i, particle in enumerate(self.swarm):
                guide = self.select_guide(i)
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                particle.velocity = (self.w * particle.velocity +
                                     self.c1 * r1 * (particle.best_position - particle.position) +
                                     self.c2 * r2 * (guide - particle.position))
                particle.position = np.clip(particle.position + particle.velocity,
                                            [b[0] for b in self.bounds],
                                            [b[1] for b in self.bounds])
            # Update temperature using the exponential cooling schedule.
            self.temperature *= self.cooling_factor
        return self.global_best_position, self.global_best_value, diversity_history

if __name__ == "__main__":
    # Example usage: Optimize the Spherical function in a 2-dimensional space.
    dim = 2
    bounds = [(-100, 100)] * dim
    pso = PSO(func=benchmark_functions.spherical, bounds=bounds, dim=dim,
              num_particles=30, max_iter=100, guide_strategy='elitist', cooling_factor=0.99)
    best_pos, best_val = pso.optimize()
    print("Best position:", best_pos)
    print("Best value:", best_val)
