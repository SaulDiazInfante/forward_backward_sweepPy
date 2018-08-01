import numpy as np

# import matplotlib.pyplot as plt

"""
    Here we adapt the example of Lab2: Mold and Fungicide
    pp. 63-66 of [1] as test problem.
    
    [1] S. Lenhart and J.T. Workman.Optimal Control Applied 
        to Biological Models.  Chap-man & Hall/CRC Mathematical
        and Computational Biology. CRC Press, 2007. ISBN9781420011418. 
    The optimal control problem reads:
    
    \begin{align}
        \min_{u}
            &
            \int_{0}^T
                a x(t)^2 + u(t)^2 dt
            \\
        x'(t)&=
            r(m - x(t)) - u(t)x(t), \qquad x(0) = 0
    \end{align}    
"""


class OptimalControlProblem:
    
    def __init__(self, n_max=1001, t_0=0.0, t_final=5.0, dimension=1,
                 x_zero=1.0, lambda_final=0):
        self.t = np.linspace(t_0, t_final, n_max)
        self.h = self.t[1] - self.t[0]
        # Parameters for the test example
        self.r = .30
        self.m = 10.0
        self.a = 1.0
        self.x_zero = x_zero
        self.x[0] = x_zero
        self.x = np.zeros([dimension, n_max])
        self.lambda_ = np.zeros([dimension, n_max])
        self.lambda_final = lambda_final
        self.lambda_[-1] = lambda_final
        self.u = np.zeros([dimension, n_max])
    
    def set_parameters(self, r, m, a, x_zero):
        self.r = r
        self.m = m
        self.a = a
        self.x_zero = x_zero
    
    def g(self, x_k, u_k):
        m = self.m
        r = self.r
        g_xu = r * (m - x_k) - u_k * x_k
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
