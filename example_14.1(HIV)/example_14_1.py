import numpy as np

"""
    Here we adapt the example of Lab8: HIV Treatment
    pp. 117-122 of [1].
    
    [1] S. Lenhart and J.T. Workman.Optimal Control Applied 
        to Biological Models.  Chap-man & Hall/CRC Mathematical
        and Computational Biology. CRC Press, 2007. ISBN9781420011418. 
    The optimal control problem reads:
    \begin{equation}
        \begin{aligned}
            \max_{u} & \int_{0}^{t_{final}}
                A  T(t) - (1-u(t)) ^ 2 dt
            \\
            \text{s.t. }
                T'(t) &=
                    \frac{s}{1 + V(t)}
                    - m_1 T(t) 
                    + r T(t)
                    \left[
                        1 - \frac{T(t)+ T_{I}(T)}{T_max}
                    \right] 
                    - u(t) k V(t) T(t),
                \\
                T_{I}(t) &=
                    u(t) k V(t) T(t) - m_2 T_{I}(t),
                \\
                V'(t) &= N m_2 T_{I}(t) - m_3 V(t),
                \\
        \end{aligned}
    \end{equation}
"""


class OptimalControlProblem(object):
    def __init__(self, t_0=0.0, t_f=100.0, dimension=3,
                 t_cell_zero=806.4, t_cell_infected_zero=0.04,
                 virus_particle_zero=1.5,
                 ):
        # Parameters for the test example
        self.t_0 = t_0
        self.t_f = t_f
        self.s = 10.0
        self.m_1 = 0.02
        self.m_2 = 0.5
        self.m_3 = 4.4
        self.r = 0.03
        self.t_cell_max = 1500.0
        self.t_cell_zero = t_cell_zero
        self.t_cell_infected_zero = t_cell_infected_zero
        self.lambda_final = np.zeros([1, 3])
        self.virus_particle_zero = virus_particle_zero
        self.a = 0.2  # 2.0/B from the article reference
        self.b = 2 * self.a
        self.k = 0.000024
        self.n_weight = 300
    
    def set_parameters(self, s, m_1, m_2, m_3, r, k, n_weight, a,
                       t_cell_max, t_cell_zero, t_cell_infected_zero,
                       virus_particle_zero):
        self.s = s
        self.m_1 = m_1
        self.m_2 = m_2
        self.m_3 = m_3
        self.r = r
        self.t_cell_max = t_cell_max
        self.k = k
        self.n_weight = n_weight
        self.a = a
        self.t_cell_zero = t_cell_zero
        self.t_cell_infected_zero = t_cell_infected_zero
        self.virus_particle_zero = virus_particle_zero
    
    def g(self, x_k, u_k):
        s = self.s
        m_1 = self.m_1
        m_2 = self.m_2
        m_3 = self.m_3
        r = self.r
        k = self.k
        t_cell_max = self.t_cell_max
        n_weight = self.n_weight
        
        t_cell = x_k[0, 0]
        t_cell_infected = x_k[0, 1]
        v = x_k[0, 2]
        
        term0_t_cell = s / (1.0 + v)
        term1_t_cell = m_1 * t_cell
        num = t_cell + t_cell_infected
        term2_t_cell = r * t_cell * (1.0 - num / t_cell_max)
        term3_t_cell = u_k * k * v * t_cell
        g_t_cell = term0_t_cell - term1_t_cell + term2_t_cell - term3_t_cell
        
        term0_t_cell_infected = u_k * k * v * t_cell
        term1_t_cell_infected = m_2 * t_cell_infected
        g_t_cell_infected = term0_t_cell_infected - term1_t_cell_infected
        
        term0_v = n_weight * m_2 * t_cell_infected
        term1_v = m_3 * v
        g_v = term0_v - term1_v
        
        g_xu = np.array([g_t_cell, g_t_cell_infected, g_v])
        g_xu = g_xu.reshape([1, 3])
        return g_xu
    
    def lambda_f(self, x_k, u_k, lambda_k):
        s = self.s
        m_1 = self.m_1
        m_2 = self.m_2
        m_3 = self.m_3
        r = self.r
        k = self.k
        a = self.a
        t_cell_max = self.t_cell_max
        n_weight = self.n_weight
        
        t_cell = x_k[0, 0]
        t_cell_infected = x_k[0, 1]
        v = x_k[0, 2]
        
        lambda_k0 = lambda_k[0, 0]
        lambda_k1 = lambda_k[0, 1]
        lambda_k2 = lambda_k[0, 2]
        
        term0_lambda_t_cell = m_1 - r * (1.0 - t_cell_infected / t_cell_max)
        
        term1_lambda_t_cell = lambda_k1 * u_k * k * v
        lambda_t_cell = -a + lambda_k0 * term0_lambda_t_cell \
                        - term1_lambda_t_cell
        
        term0_lambda_t_cell_infected = (lambda_k0 * r * t_cell) / t_cell_max
        term1_lambda_t_cell_infected = lambda_k1 * m_2
        term2_lambda_t_cell_infected = lambda_k2 * m_2 * n_weight
        lambda_t_cell_infected = term0_lambda_t_cell_infected \
                                 + term1_lambda_t_cell_infected \
                                 - term2_lambda_t_cell_infected
        term0_lambda_v = lambda_k0 * (s / (1.0 + v) + u_k * k * t_cell)
        term1_lambda_v = lambda_k1 * u_k * k * t_cell
        term2_lambda_v = lambda_k2 * m_3
        lambda_v = term0_lambda_v - term1_lambda_v + term2_lambda_v
        
        l_kxu = np.array([lambda_t_cell, lambda_t_cell_infected, lambda_v])
        l_kxu = l_kxu.reshape([1, 3])
        return l_kxu
    
    def optimality_condition(self, x_k, u_k, lambda_k, n_max):
        k = self.k
        lambda_t_cell = lambda_k[:, 0]
        lambda_t_cell_infected = lambda_k[:, 1]
        t_cell = x_k[:, 0]
        v = x_k[:, 2]
        
        u_aux = 1.0 + 0.5 * k * v * t_cell \
                * (lambda_t_cell_infected - lambda_t_cell)
        positive_part = u_aux
        sign = np.sign(u_aux)
        index = np.where(sign == -1)
        positive_part[index] = 0.0
        u_aster_kp1 = np.min([positive_part, np.ones(n_max)], axis=0)
        return u_aster_kp1
