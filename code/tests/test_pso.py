import unittest
import math
import random
import numpy as np
from src.guide_strategies import (
    standard_elitist, 
    simulated_annealing_acceptance, 
    roulette_wheel_selection, 
    tournament_selection, 
    rank_based_selection
)

class TestPSO(unittest.TestCase):
    
    def test_standard_elitist(self):
        # When candidate is better, should return True.
        self.assertTrue(standard_elitist(10, 5), "Failed: candidate is better but not accepted")
        # When candidate is worse, should return False.
        self.assertFalse(standard_elitist(10, 15), "Failed: candidate is worse but accepted")

    def test_simulated_annealing_acceptance(self):
        # If candidate is better, it should always be accepted.
        self.assertTrue(simulated_annealing_acceptance(10, 5, 1.0), "Failed: better candidate not accepted")
        # When candidate is worse, run multiple times and check the acceptance rate.
        # Theoretical probability = exp(-(10-5)/1.0) = exp(-5) ~ 0.0067.
        trials = 1000
        acceptances = sum(simulated_annealing_acceptance(5, 10, 1.0) for _ in range(trials))
        # With 1000 trials, expect roughly 7 acceptances (allow some tolerance)
        self.assertLess(acceptances, 20, "Failed: acceptance probability seems too high for a worse candidate")

    def test_roulette_wheel_selection(self):
        # Create a candidate set where the best candidate (lowest fitness) is 'c'
        candidates = ['a', 'b', 'c']
        fitnesses = [10, 5, 1]  # 'c' is the best
        selections = [roulette_wheel_selection(candidates, fitnesses) for _ in range(1000)]
        freq_c = selections.count('c')
        # Given the weights computed in the function, 'c' is expected to be chosen roughly 64% of the time.
        # That is about 640 out of 1000; we set a threshold of 600.
        self.assertGreater(freq_c, 600, "Failed: best candidate not selected frequently enough")

    def test_tournament_selection(self):
        candidates = ['a', 'b', 'c', 'd']
        fitnesses = [4, 2, 3, 1]  # 'd' is the best
        selections = [tournament_selection(candidates, fitnesses, tournament_size=2) for _ in range(1000)]
        freq_d = selections.count('d')
        # With a tournament size of 2, we expect 'd' to be selected often; threshold is set to 400.
        self.assertGreater(freq_d, 400, "Failed: tournament selection is not favoring the best candidate")

    def test_rank_based_selection(self):
        candidates = ['a', 'b', 'c', 'd']
        fitnesses = [4, 2, 3, 1]  # 'd' is the best
        selections = [rank_based_selection(candidates, fitnesses) for _ in range(1000)]
        freq_d = selections.count('d')
        # For rank-based selection, the best candidate 'd' receives probability proportional to its rank.
        # With four candidates, if ranks 1, 2, 3, 4 are assigned in increasing order, the best candidate gets a probability of 1/10 = 10%.
        # That is approximately 100 selections out of 1000; we set a threshold of 80.
        self.assertGreater(freq_d, 80, "Failed: rank-based selection does not favor the best candidate sufficiently")

if __name__ == "__main__":
    unittest.main()
