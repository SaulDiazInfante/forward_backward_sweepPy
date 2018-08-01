import numpy as np

"""
    Here we adapt the example of Lab2: Mold and Fungicide
    pp. 67-70 of [1] as test problem.
    
    [1] S. Lenhart and J.T. Workman.Optimal Control Applied 
        to Biological Models.  Chap-man & Hall/CRC Mathematical
        and Computational Biology. CRC Press, 2007. ISBN9781420011418. 
    The optimal control problem reads:
    
    \begin{align}
        \min_{u}
            &
            c x(1) -
            \int_{0}^T
                 u(t)^2
                 dt
            \\
        x'(t)&=
            rx(t) + ax(t) - b u(t)^x(t) \exp(-x(t)), \qquad x(0) = x_0
    \end{align}    
"""


class OptimalControlProblem:
    
    def __init__(self, n_max=5000, t_0=0.0, t_final=1.0, dimension=1,
                 x_zero=1.0, lambda_final=0):
        self.t = np.linspace(t_0, t_final, n_max)
        self.h = self.t[1] - self.t[0]
        # Parameters for the test example
        self.r = .30
        self.a = 1.0
        self.b = 12.0
        self.c = 1.0
        self.x_zero = x_zero
        self.x[0] = x_zero
        self.x = np.zeros([dimension, n_max])
        self.lambda_ = np.zeros([dimension, n_max])
        self.lambda_final = lambda_final
        self.lambda_[-1] = lambda_final
        self.u = np.zeros([dimension, n_max])
    
    def set_parameters(self, r, a, b, c, x_zero):
        self.r = r
        self.a = a
        self.b = b
        self.c = c
        self.x_zero = x_zero
    
    def g(self, x_k, u_k):
        a = self.a
        b = self.b
        r = self.r
        g_xu = r * x_k + a * u_k * x_k - b * (u_k ** 2) * np.exp(-x_k)
        return g_xu
    
    def lambda_f(self, x_k, u_k, lambda_k):
        a = self.a
        b = self.b
        r = self.r
        lambda_xu = - lambda_k * (r + (a + b * u_k * np.exp(-x_k)) * u_k)
        return lambda_xu
    
    def optimality_condition(self, x_k, u_k, lambda_k):
        a = self.a
        b = self.b
        u_aster_kp1 = (lambda_k * a * x_k) / \
                      (2 * (lambda_k * b * np.exp(-x_k) + 1))
        return u_aster_kp1
