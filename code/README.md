R. Eberhart and Y. Shi, “Comparing inertia weights and constriction
factors in particle swarm optimization,” in Proceedings of the 2000
Congress on Evolutionary Computation. CEC00 (Cat. No.00TH8512),
vol. 1, 2000, pp. 84–88 vol.1.

For the experiments, I used the following parameters:
w= 0.7
c1= 1.4
c2= 1.4
num_particles= 30
max_iter= 100

---

CEC 2005/2013 Benchmarking Guidelines - IEEE Congress on Evolutionary Computation (CEC).

For the CEC 2005 experiments, I used the following parameters:
runs = 30

---

The comment notes that Engelbrecht (2007) recommends a cooling factor between 0.95 and 0.99 for effective cooling in hybrid PSO-simulated annealing algorithms.
A slower cooling rate (cooling factor closer to 1) helps maintain exploration for a longer period.
A faster cooling rate (lower cooling factor) can lead to premature convergence.
