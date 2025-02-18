import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src import pso, benchmark_functions, utils

def run_experiment(func, func_name, bounds, dim, guide_strategy, num_runs=10, max_iter=100):
    """
    Run a PSO experiment on a given benchmark function with a specific guide strategy.
    
    Parameters:
    - func: The benchmark function to optimize.
    - func_name: Name of the function.
    - bounds: The search space boundaries.
    - dim: Number of dimensions.
    - guide_strategy: Guide selection strategy.
    - num_runs: Number of independent runs.
    - max_iter: Maximum number of iterations.
    
    Returns:
    - DataFrame with results for this strategy.
    - List of diversity histories (one per run).
    """
    results = []
    all_diversity = []

    for run in range(num_runs):
        pso_instance = pso.PSO(func=func, bounds=bounds, dim=dim, 
                               num_particles=30, max_iter=max_iter, 
                               guide_strategy=guide_strategy)

        best_pos, best_val, diversity = pso_instance.optimize(track_diversity=True)
        results.append([run + 1, guide_strategy, best_val])
        all_diversity.append(diversity)

        print(f"Run {run+1}/{num_runs} - Best Value: {best_val:.6f}")

    # Create DataFrame with columns: Run, Strategy, Best Value
    df_results = pd.DataFrame(results, columns=["Run", "Strategy", "Best Value"])
    return df_results, np.mean(df_results["Best Value"]), np.std(df_results["Best Value"]), all_diversity

if __name__ == "__main__":
    benchmark_tests = [
        ("Spherical", benchmark_functions.spherical, [(-100, 100)] * 2),
        ("Booth", benchmark_functions.booth, [(-10, 10)] * 2),
        ("Rosenbrock", benchmark_functions.rosenbrock, [(-30, 30)] * 2),
        ("Ackley", benchmark_functions.ackley, [(-32, 32)] * 2),
        ("Michalewicz", benchmark_functions.michalewicz, [(0, np.pi)] * 2)
    ]

    strategies = ['elitist', 'simulated_annealing', 'roulette', 'tournament', 'rank']

    # For each benchmark function, aggregate results from all strategies into one CSV.
    for func_name, func, bounds in benchmark_tests:
        aggregate_results = []  # list to collect all results for this benchmark
        all_diversity_histories = {}
        strategy_summary = []

        for strategy in strategies:
            print(f"\nRunning experiment for {func_name} using {strategy} strategy...\n")
            df_results, avg_val, std_val, diversity_history = run_experiment(
                func, func_name, bounds, dim=2, guide_strategy=strategy, num_runs=25, max_iter=100
            )
            # Append the results with an extra column for benchmark function name.
            df_results["Benchmark"] = func_name
            aggregate_results.append(df_results)
            strategy_summary.append([strategy, avg_val, std_val])
            all_diversity_histories[strategy] = np.mean(diversity_history, axis=0)

        # Concatenate results from all strategies into one DataFrame.
        df_aggregate = pd.concat(aggregate_results, axis=0)
        # Save the aggregate results to one CSV per benchmark function.
        csv_filename = f"results/{func_name.lower()}_aggregate_results.csv"
        df_aggregate.to_csv(csv_filename, index=False)
        print(f"Aggregate results saved to {csv_filename}")

        # Create a summary DataFrame for bar chart plotting.
        df_summary = pd.DataFrame(strategy_summary, columns=["Strategy", "Avg Best Value", "Std Dev"])

        # Plot 1: Performance Comparison Bar Chart
        plt.figure(figsize=(8, 5))
        plt.bar(
            df_summary["Strategy"],
            df_summary["Avg Best Value"],
            yerr=df_summary["Std Dev"],
            color=plt.cm.viridis(np.linspace(0.2, 0.8, len(df_summary))),
            capsize=5,
            error_kw=dict(ecolor="black", elinewidth=1, capsize=5, capthick=1)
        )
        plt.xlabel("Guide Strategy")
        plt.ylabel("Best Function Value")
        plt.title(f"Optimization Performance on {func_name}")
        plt.xticks(rotation=45)
        plt.yscale("linear")
        plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"results/{func_name.lower()}_performance.png")
        plt.close()  # Closes the figure to avoid blocking further code execution

        # Plot 2: Swarm Diversity Over Time
        plt.figure(figsize=(8, 5))
        for strategy, diversity_values in all_diversity_histories.items():
            plt.plot(diversity_values, label=strategy)
        plt.xlabel("Iterations")
        plt.ylabel("Swarm Diversity")
        plt.title(f"Swarm Diversity Over Time - {func_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/{func_name.lower()}_diversity.png")

        # Plot 3: Convergence Plot
        plt.figure(figsize=(8, 5))
        for strategy in strategies:
            df_temp = df_aggregate[df_aggregate["Strategy"] == strategy]
            plt.plot(df_temp["Run"], df_temp["Best Value"], marker='o', label=strategy)
        plt.xlabel("Run")
        plt.ylabel("Best Function Value")
        plt.title(f"Convergence of Strategies - {func_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/{func_name.lower()}_convergence.png")
