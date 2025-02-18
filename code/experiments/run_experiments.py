import numpy as np
from src import pso, benchmark_functions, utils
import matplotlib.pyplot as plt

def run_experiment(func, func_name, bounds, dim, guide_strategy, num_runs=10, max_iter=100):
    best_values = []
    best_positions = []
    
    for run in range(num_runs):
        pso_instance = pso.PSO(func=func, bounds=bounds, dim=dim, 
                               num_particles=30, max_iter=max_iter, guide_strategy=guide_strategy)
        best_pos, best_val = pso_instance.optimize()
        best_values.append(best_val)
        best_positions.append(best_pos)
        print(f"Run {run+1}/{num_runs} - Best Value: {best_val}")
    
    avg_val = np.mean(best_values)
    std_val = np.std(best_values)
    print(f"\nResults for {func_name} using {guide_strategy} strategy:")
    print(f"Average Best Value: {avg_val:.4f}, Std Dev: {std_val:.4f}")
    return best_values, best_positions

if __name__ == "__main__":
    # Example experiment: Optimize the Ackley function in 2D with the 'roulette' guide strategy.
    dim = 2
    bounds = [(-32, 32)] * dim
    func = benchmark_functions.ackley
    func_name = "Ackley"
    
    # Choose one of the guide strategies: 'elitist', 'simulated_annealing', 'roulette', 'tournament', or 'rank'
    guide_strategy = 'rank'
    
    best_values, best_positions = run_experiment(func, func_name, bounds, dim, guide_strategy, num_runs=5, max_iter=100)
    
    # Optionally, plot the contour of the function and the best positions from the last run
    utils.plot_contour(func, bounds)
