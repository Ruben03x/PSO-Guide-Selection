import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src import pso, benchmark_functions, utils

def run_experiment(func, func_name, bounds, dim, guide_strategy, num_runs=30):
    """
    Run a PSO experiment on a given benchmark function with a specific guide strategy.
    
    Parameters:
      - func: The benchmark function.
      - func_name: Name of the function.
      - bounds: Search space boundaries.
      - dim: Number of dimensions.
      - guide_strategy: Guide selection strategy.
      - num_runs: Number of independent runs.
    
    Returns:
      - df_results: DataFrame with results for this strategy.
      - avg_val: Average best value.
      - std_val: Standard deviation of best values.
      - all_diversity: List of diversity histories (one per run).
    """
    # Dynamically set max_iter based on dim
    max_iter = 20 * dim  
    results = []
    all_diversity = []

    for run in range(num_runs):
        pso_instance = pso.PSO(func=func, bounds=bounds, dim=dim, 
                               num_particles=30, max_iter=max_iter, 
                               guide_strategy=guide_strategy, cooling_factor=0.99)
        best_pos, best_val, diversity = pso_instance.optimize(track_diversity=True)
        results.append([run + 1, guide_strategy, best_val])
        all_diversity.append(diversity)
        print(f"Run {run+1}/{num_runs} - Best Value: {best_val:.6f}")

    df_results = pd.DataFrame(results, columns=["Run", "Strategy", "Best Value"])
    avg_val = np.mean(df_results["Best Value"])
    std_val = np.std(df_results["Best Value"])
    return df_results, avg_val, std_val, all_diversity

if __name__ == "__main__":
    benchmark_tests = [
        ("Spherical", benchmark_functions.spherical, [(-100, 100)] * 2),
        ("Booth", benchmark_functions.booth, [(-10, 10)] * 2),
        ("Rosenbrock", benchmark_functions.rosenbrock, [(-30, 30)] * 2),
        ("Ackley", benchmark_functions.ackley, [(-32, 32)] * 2),
        ("Michalewicz", benchmark_functions.michalewicz, [(0, np.pi)] * 2)
    ]

    strategies = ['elitist', 'simulated_annealing', 'roulette', 'tournament', 'rank']

    for func_name, func, bounds in benchmark_tests:
        aggregate_results = []
        diversity_summary = {}
        strategy_summary = []

        for strategy in strategies:
            print(f"\nRunning experiment for {func_name} using {strategy} strategy...\n")
            df_results, avg_val, std_val, diversity_history = run_experiment(
                func, func_name, bounds, dim=2, guide_strategy=strategy, num_runs=30
            )
            df_results["Benchmark"] = func_name
            aggregate_results.append(df_results)
            strategy_summary.append([strategy, avg_val, std_val])
            diversity_summary[strategy] = np.mean(diversity_history, axis=0)

        df_aggregate = pd.concat(aggregate_results, axis=0)
        csv_filename = f"results/{func_name.lower()}_aggregate_results.csv"
        df_aggregate.to_csv(csv_filename, index=False)
        print(f"Aggregate results saved to {csv_filename}")

        df_summary = pd.DataFrame(strategy_summary, columns=["Strategy", "Avg Best Value", "Std Dev"])
        summary_csv = f"results/{func_name.lower()}_summary.csv"
        df_summary.to_csv(summary_csv, index=False)
        print(f"Summary table saved to {summary_csv}")
        print(f"\nSummary for {func_name}:\n", df_summary)

        # --- Plot: Performance Comparison Bar Chart ---
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
        plt.close()

        # --- Plot: Swarm Diversity Over Time ---
        plt.figure(figsize=(8, 5))
        for strategy, diversity_values in diversity_summary.items():
            plt.plot(diversity_values, label=strategy)
        plt.xlabel("Iterations")
        plt.ylabel("Swarm Diversity")
        plt.title(f"Swarm Diversity Over Time - {func_name}")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"results/{func_name.lower()}_diversity.png")
        plt.close()

        # --- Plot: Convergence Plot ---
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
        plt.close()
