import numpy as np

"""
    Here we adapt the example of Lab2: Mold and Fungicide
    pp. 85-87 of [1] as test problem.
    
    [1] S. Lenhart and J.T. Workman.Optimal Control Applied 
        to Biological Models.  Chap-man & Hall/CRC Mathematical
        and Computational Biology. CRC Press, 2007. ISBN9781420011418. 
    The optimal control problem reads:
    
    \begin{align}
        \max_{u}
            &
            \int_{0}^T
                a x(t) -u(t)^2
                 dt
            \\
        x'(t)&=
            -\frac{1}{2} x(t) ^ 2 +  c u(t)^x(t) , \qquad 
            x(0) = x_0 > -2
            \\
        m_1 \leq u(t) \leq m_2
    \end{align}    
"""


class OptimalControlProblem(object):
    
    def __init__(self, t_0=0.0, t_f=1.0, dimension=1,
                 x_zero=1.0, lambda_final=0.0):
        # Parameters for the test example
        self.t_0 = t_0
        self.t_f = t_f
        self.a = 1.0
        self.c = 4.0
        self.m_1 = -1.0
        self.m_2 = 2.0
        self.x_zero = x_zero
        self.lambda_final = lambda_final
    
    def set_parameters(self, a, c, x_zero, m_1, m_2):
        self.a = a
        self.c = c
        self.x_zero = x_zero
        self.m_1 = m_1
        self.m_2 = m_2
    
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
        m_1 = self.m_1
        m_2 = self.m_2
        n_max = self.n_max
        aux_1 = np.max([m_1 * np.ones(n_max), c * 0.5 * lambda_k], axis=0)
        u_aster_kp1 = np.min([m_2 * np.ones(n_max), aux_1], axis=0)
        return u_aster_kp1
