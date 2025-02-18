import numpy as np
import random
from src import benchmark_functions, guide_strategies, utils

class Particle:
    def __init__(self, dim, bounds):
        self.position = np.array([random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)])
        self.velocity = np.zeros(dim)
        self.best_position = np.copy(self.position)
        self.best_value = float('inf')

class PSO:
    def __init__(self, func, bounds, dim, num_particles=30, max_iter=100, 
                 w=0.7, c1=1.4, c2=1.4, guide_strategy='elitist'):
        self.func = func
        self.bounds = bounds
        self.dim = dim
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.guide_strategy = guide_strategy
        
        self.swarm = [Particle(dim, bounds) for _ in range(num_particles)]
        self.global_best_position = np.zeros(dim)
        self.global_best_value = float('inf')
        self.temperature = 1.0  # for simulated annealing if needed

    def update_global_best(self):
        for particle in self.swarm:
            value = self.func(particle.position)
            if value < particle.best_value:
                particle.best_value = value
                particle.best_position = np.copy(particle.position)
            if value < self.global_best_value:
                self.global_best_value = value
                self.global_best_position = np.copy(particle.position)

    def select_guide(self, particle_index):
        # Collect all pbests and their fitnesses
        pbests = [p.best_position for p in self.swarm]
        fitnesses = [self.func(p.best_position) for p in self.swarm]
        
        if self.guide_strategy == 'elitist':
            # Simply return the best among pbests
            best_index = np.argmin(fitnesses)
            return pbests[best_index]
        elif self.guide_strategy == 'simulated_annealing':
            # Use simulated annealing to decide if global best should be updated
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
            # Default to elitist
            best_index = np.argmin(fitnesses)
            return pbests[best_index]

    def optimize(self):
        for t in range(self.max_iter):
            self.update_global_best()
            for i, particle in enumerate(self.swarm):
                guide = self.select_guide(i)
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                # Velocity update
                particle.velocity = (self.w * particle.velocity +
                                     self.c1 * r1 * (particle.best_position - particle.position) +
                                     self.c2 * r2 * (guide - particle.position))
                # Position update with boundary checking
                particle.position = particle.position + particle.velocity
                for d in range(self.dim):
                    particle.position[d] = np.clip(particle.position[d],
                                                   self.bounds[d][0],
                                                   self.bounds[d][1])
            # Optional: update temperature (if using simulated annealing)
            self.temperature *= 0.99  # simple cooling schedule
        return self.global_best_position, self.global_best_value

if __name__ == "__main__":
    # Example: Optimize the Spherical function in 2D
    dim = 2
    bounds = [(-100, 100)] * dim
    pso = PSO(func=benchmark_functions.spherical, bounds=bounds, dim=dim,
              num_particles=30, max_iter=100, guide_strategy='elitist')
    best_pos, best_val = pso.optimize()
    print("Best position:", best_pos)
    print("Best value:", best_val)
