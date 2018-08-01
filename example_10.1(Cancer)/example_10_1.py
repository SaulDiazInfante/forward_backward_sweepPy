import numpy as np

"""
    Here we adapt the example of Lab5: Cancer
    pp. 89-92 of [1] as test problem.
    
    [1] S. Lenhart and J.T. Workman.Optimal Control Applied 
        to Biological Models.  Chap-man & Hall/CRC Mathematical
        and Computational Biology. CRC Press, 2007. ISBN9781420011418. 
    The optimal control problem reads:
    
    \begin{align}
        \min_{u}
            &
            \int_{0}^T
                a n(t)  + u(t)^2
                 dt
            \\
        x'(t)&=
            r x(t) 
            \log
            \left(
                \frac{1}{x(t)}
            \right)
             - \delta u(t)x(t) , \qquad 
            x(0) = x_0 \\
        u(t) \geq 0
    \end{align}    
"""


class OptimalControlProblem(object):
    
    def __init__(self, t_0=0.0, t_f=20.0, dimension=1,
                 x_zero=.975, lambda_final=0.0):
        # Parameters for the test example
        self.t_0 = t_0
        self.t_f = t_f
        self.r = 0.3
        self.a = 3.0
        self.delta = 0.45
        self.c = 4.0
        self.m_1 = -1.0
        self.m_2 = 2.0
        self.dimension = dimension
        self.x_zero = x_zero
        self.lambda_final = lambda_final
    
    def set_parameters(self, r, a, delta, x_zero, t_f):
        self.r = r
        self.a = a
        self.delta = delta
        self.x_zero = x_zero
        self.t_f = t_f
    
    def g(self, x_k, u_k):
        r = self.r
        delta = self.delta
        g_xu = r * x_k * np.log(1.0 / x_k) - delta * u_k * x_k
        return g_xu
    
    def lambda_f(self, x_k, u_k, lambda_k):
        a = self.a
        lambda_xu = -a + x_k * lambda_k
        return lambda_xu
    
    def optimality_condition(self, x_k, u_k, lambda_k):
        c = self.c
        m_1 = self.m_1
        m_2 = self.m_2
        n_max = self.n_max
        aux_1 = np.max([m_1 * np.ones(n_max), c * 0.5 * lambda_k], axis=0)
        u_aster_kp1 = np.min([m_2 * np.ones(n_max), aux_1], axis=0)
        return u_aster_kp1
