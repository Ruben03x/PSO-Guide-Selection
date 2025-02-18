import numpy as np
import math
import random

def standard_elitist(current_value, candidate_value):
    """
    Standard elitist approach: update only if candidate is better.
    """
    return candidate_value < current_value

def simulated_annealing_acceptance(current_value, candidate_value, temperature):
    """
    Simulated annealing acceptance probability.
    If candidate is better, always accept; otherwise, accept with probability.
    """
    if candidate_value < current_value:
        return True
    else:
        # Calculate probability (using a simple Boltzmann factor)
        prob = math.exp(-(candidate_value - current_value) / temperature)
        return random.random() < prob

def roulette_wheel_selection(candidates, fitnesses):
    """
    Probabilistic selection (roulette wheel).
    Lower fitness is better.
    Invert fitness (or subtract from max) to calculate selection probabilities.
    """
    max_fit = max(fitnesses)
    # Invert fitness so that lower fitness gives higher weight
    weights = [max_fit - f + 1e-6 for f in fitnesses]
    total = sum(weights)
    probs = [w/total for w in weights]
    r = random.random()
    cum_prob = 0.0
    for candidate, p in zip(candidates, probs):
        cum_prob += p
        if r < cum_prob:
            return candidate
    return candidates[-1]

def tournament_selection(candidates, fitnesses, tournament_size=3):
    """
    Tournament selection: randomly choose 'tournament_size' candidates,
    and return the one with the best (lowest) fitness.
    """
    selected_indices = np.random.choice(len(candidates), tournament_size, replace=False)
    best = None
    best_fit = float('inf')
    for i in selected_indices:
        if fitnesses[i] < best_fit:
            best_fit = fitnesses[i]
            best = candidates[i]
    return best

def rank_based_selection(candidates, fitnesses):
    """
    Rank-based selection: sort candidates by fitness and select based on rank.
    Lower fitness gets higher probability.
    """
    ranked = sorted(zip(candidates, fitnesses), key=lambda x: x[1])
    # Assign probability proportional to rank (using linear ranking)
    ranks = np.arange(1, len(candidates)+1)
    total = sum(ranks)
    probs = [rank/total for rank in ranks]
    r = random.random()
    cum_prob = 0.0
    for (candidate, _), p in zip(ranked, probs):
        cum_prob += p
        if r < cum_prob:
            return candidate
    return ranked[-1][0]
