import numpy as np

"""
    Here we adapt the example of Lab9: Bear Populations
    pp. 129-134 of [1].
    
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
    def __init__(self, t_0=0.0, t_f=25.0, dimension=3,
                 p_zero=0.4, f_zero=0.2, o_zero=0.0
                 ):
        # Parameters for the test example
        self.t_0 = t_0
        self.t_f = t_f
        self.r = 0.1
        self.k = 0.75
        self.m_p = 0.5
        self.m_f = 0.5
        self.c_p = 10000.0
        self.c_f = 10.0
        # initial conditions
        self.p_zero = p_zero
        self.f_zero = f_zero
        self.o_zero = o_zero
        self.lambda_final = np.zeros([1, 3])
    
    def set_parameters(self, r, k, m_p, m_f, c_p, c_f, p_zero, f_zero, o_zero):
        self.r = r
        self.k = k
        self.m_p = m_p
        self.m_f = m_f
        self.c_p = c_p
        self.c_f = c_f
        # initial conditions
        self.p_zero = p_zero
        self.f_zero = f_zero
        self.o_zero = o_zero
    
    def g(self, x_k, u_k):
        r = self.r
        k = self.k
        m_p = self.m_p
        m_f = self.m_f
        p = x_k[0, 0]
        f = x_k[0, 1]
        u_p = u_k[0, 0]
        u_f = u_k[0, 1]
        
        p_log_term = (m_f / k) * r * (1.0 - p / k) * (f ** 2)
        rhs_p = p * (r - (r / k) * p - u_p) + p_log_term
        
        f_log_term = (m_p / k) * r * (1.0 - f / k) * (p ** 2)
        rhs_f = f * (r - (r / k) * f - u_f) + f_log_term
        
        o_term_p = (1.0 - m_p) * (r / k) * (p ** 2) \
                   + (m_f * r / (k ** 2)) * p * (f ** 2)
        o_term_f = (1.0 - m_f) * (r / k) * (f ** 2) \
                   + (m_p * r / (k ** 2)) * f * (p ** 2)
        rhs_o = o_term_p + o_term_f
        
        g_xu = np.array([rhs_p, rhs_f, rhs_o])
        g_xu = g_xu.reshape([1, 3])
        return g_xu
    
    def lambda_function(self, x_k, u_k, lambda_k):
        r = self.r
        k = self.k
        m_p = self.m_p
        m_f = self.m_f
        p = x_k[0, 0]
        f = x_k[0, 1]
        u_p = u_k[0, 0]
        u_f = u_k[0, 1]
        lambda_p = lambda_k[0, 0]
        lambda_f = lambda_k[0, 1]
        lambda_o = lambda_k[0, 2]
        #
        #
        l_p_term_1 = r - 2.0 * (r / k) * p - (m_f * r / (k ** 2)) * (f ** 2) \
                     - u_p
        l_p_term_2 = 2.0 * m_p * (r / k) * (1.0 - f / k) * p
        l_p_term_3 = 2.0 * (r / k) * (1.0 - m_p) * p \
                     + m_f * (r / (k ** 2)) * f ** 2 \
                     + 2.0 * m_p * (r / (k ** 2)) * p * f
        
        rhs_l_p = lambda_p * l_p_term_1 + lambda_f * l_p_term_2 \
                  + lambda_o * l_p_term_3
        rhs_l_p = -1.0 * rhs_l_p
        
        l_f_term_1 = 2.0 * m_f * (r / k) * (1.0 - p / k) * f
        l_f_term_2 = r - 2.0 * (r / k) * f - (m_p * r / (k ** 2)) * (p ** 2) \
                     - u_f
        #
        #
        l_f_term_3 = 2.0 * (r / k) * (1.0 - m_f) * f \
                     + m_p * (r / k ** 2) * p ** 2 \
                     + 2.0 * m_f * (r / (k ** 2)) * p * f
        
        rhs_l_f = lambda_p * l_f_term_1 + lambda_f * l_f_term_2 \
                  + lambda_o * l_f_term_3
        rhs_l_f = -1.0 * rhs_l_f
        rhs_l_o = -1.0
        l_kxu = np.array([rhs_l_p, rhs_l_f, rhs_l_o])
        l_kxu = l_kxu.reshape([1, 3])
        return l_kxu
    
    def optimality_condition(self, x_k, u_k, lambda_k, n_max):
        c_p = self.c_p
        c_f = self.c_f
        p = x_k[:, 0]
        f = x_k[:, 1]
        lambda_p = lambda_k[:, 0]
        lambda_f = lambda_k[:, 1]
        aux_p = 1.0 / (2.0 * c_p) * lambda_p * p
        aux_f = 1.0 / (2.0 * c_f) * lambda_f * f
        positive_part_p = np.max([np.zeros(n_max), aux_p], axis=0)
        positive_part_f = np.max([np.zeros(n_max), aux_f], axis=0)
        u_aster_p = np.min([np.ones(n_max), positive_part_p], axis=0)
        u_aster_f = np.min([np.ones(n_max), positive_part_f], axis=0)
        u_aster = np.zeros([n_max, 2])
        u_aster[:, 0] = u_aster_p
        u_aster[:, 1] = u_aster_f
        return u_aster
