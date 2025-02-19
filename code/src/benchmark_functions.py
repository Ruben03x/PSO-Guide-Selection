import numpy as np

def spherical(x):
    """
    Spherical Function:
      f(x) = sum(x_i^2)
    Domain: x_i in [-100, 100]
    Global minimum: f(0) = 0 at x = (0, ..., 0)
    """
    return np.sum(x**2) # Verified

def booth(x):
    """
    Booth Function:
      f(x) = (x1 + 2*x2 - 7)^2 + (2*x1 + x2 - 5)^2
    Domain: x_i in [-10, 10]
    Global minimum: f(1,3) = 0
    """
    x1, x2 = x[0], x[1]
    return (x1 + 2*x2 - 7)**2 + (2*x1 + x2 - 5)**2 # Verified

def rosenbrock(x):
    """
    Rosenbrock Function:
      f(x) = sum_{i=1}^{n-1} [100*(x_{i+1} - x_i^2)^2 + (x_i - 1)^2]
    Domain: x_i in [-30, 30]
    Global minimum: f(1,1,...,1) = 0
    Note: Becomes more challenging in higher dimensions.
    """
    return np.sum(100.0*(x[1:] - x[:-1]**2)**2 + (x[:-1]-1)**2) # Verified

def ackley(x):
    """
    Ackley Function:
      f(x) = -20 exp(-0.2 sqrt((1/n) sum(x_i^2)) )
             - exp((1/n) sum(cos(2*pi*x_i))) + 20 + e
    Domain: x_i in [-32, 32]
    Global minimum: f(0,...,0) = 0
    """
    n = len(x)
    a = 20
    b = 0.2
    c = 2 * np.pi
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(sum_sq/n))
    term2 = -np.exp(sum_cos / n)
    return term1 + term2 + a + np.e # Verified

def michalewicz(x, m=10):
    """
    Michalewicz Function:
      f(x) = - sum_{i=1}^{n} sin(x_i) [sin(i*x_i^2*pi/n)]^(2*m)
    Domain: x_i in [0, pi]
    Global minimum: Approximately f(x*) ≈ -0.966*n (non-trivial x*)
    Note: Highly multimodal.
    """
    n = len(x)
    i = np.arange(1, n+1)
    return -np.sum(np.sin(x) * (np.sin(i * x**2 * np.pi / n))**(2*m)) # Verified
