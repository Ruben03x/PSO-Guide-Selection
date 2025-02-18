import unittest
import numpy as np
from src import pso, benchmark_functions

class TestPSO(unittest.TestCase):
    def test_spherical_convergence(self):
        # Test that PSO converges on the simple Spherical function in 2D.
        dim = 2
        bounds = [(-100, 100)] * dim
        pso_instance = pso.PSO(func=benchmark_functions.spherical, bounds=bounds, dim=dim,
                               num_particles=30, max_iter=100, guide_strategy='elitist')
        best_pos, best_val = pso_instance.optimize()
        # Expect best value to be close to 0 for the Spherical function.
        self.assertAlmostEqual(best_val, 0.0, delta=1e-2)
        
    def test_booth_convergence(self):
        # Test PSO on the Booth function in 2D.
        dim = 2
        bounds = [(-10, 10)] * dim
        pso_instance = pso.PSO(func=benchmark_functions.booth, bounds=bounds, dim=dim,
                               num_particles=30, max_iter=100, guide_strategy='elitist')
        best_pos, best_val = pso_instance.optimize()
        # Expect best value to be close to 0.
        self.assertAlmostEqual(best_val, 0.0, delta=1e-2)

if __name__ == '__main__':
    unittest.main()
