import numpy as np

# import matplotlib.pyplot as plt

"""
    Here we adapt example 4.1 pp. 52-60 of [1] as test problem.
"""


class OptimalControlProblem:
    
    def __init__(self, n_max=1001, t_0=0.0, t_final=10, dimension=1,
                 x_zero=0.0, lambda_final=0):
        self.t = np.linspace(t_0, t_final, n_max)
        self.h = self.t[1] - self.t[0]
        # Parameters for the test example
        self.a = 1.0
        self.b = 1.0
        self.c = 1.0
        self.x_zero = x_zero
        self.x[0] = x_zero
        self.x = np.zeros([dimension, n_max])
        self.lambda_ = np.zeros([dimension, n_max])
        self.lambda_final = lambda_final
        self.lambda_[-1] = lambda_final
        self.u = np.zeros([dimension, n_max])
    
    def set_parameters(self, a, b, c, x_zero, lambda_final):
        self.a = a
        self.b = b
        self.c = c
        self.x_zero = x_zero
        self.lambda_final = lambda_final
    
    def g(self, x_k, u_k):
        c = self.c
        g_xu = -0.5 * x_k ** 2 + c * u_k
        return g_xu
    
    def lambda_f(self, x_k, u_k, lambda_k):
        a = self.a
        lambda_xu = -a + x_k * lambda_k
        return lambda_xu
    
    def optimality_condition(self, x_k, u_k, lambda_k):
        c = self.c
        b = self.b
        u_aster_kp1 = c / (2.0 * b) * lambda_k
        return u_aster_kp1
