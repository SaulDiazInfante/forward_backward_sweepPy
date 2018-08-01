import numpy as np

"""
    Here we adapt the example of Lab7: Epidemic Model
    pp. 123-127 of [1].
    
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
    
    def __init__(self, t_0=0.0, t_f=20.0, dimension=4,
                 s_zero=1000, e_zero=100, i_zero=50.0,
                 r_zero=15, n_zero=1165):
        # Parameters for the test example
        self.t_0 = t_0
        self.t_f = t_f
        self.b = 0.525
        self.d = 0.5
        self.c = 0.0001
        self.e = 0.5
        self.g = 0.1
        self.a = 0.2
        self.a_w = 2.0
        # Initial conditions
        self.s_zero = s_zero
        self.e_zero = e_zero
        self.i_zero = i_zero
        self.r_zero = r_zero
        self.n_zero = n_zero
        #
        self.dimension = dimension
    
    def set_parameters(self, t_0, t_f, b, d, c, e, g, a, a_w, s_zero, e_zero,
                       i_zero, r_zero, n_zero):
        self.t_0 = t_0
        self.t_f = t_f
        self.b = b
        self.d = d
        self.c = c
        self.e = e
        self.g = g
        self.a = a
        self.a_w = a_w
        # Initial conditions
        self.s_zero = s_zero
        self.e_zero = e_zero
        self.i_zero = i_zero
        self.r_zero = r_zero
        self.n_zero = n_zero
    
    def g_ode(self, x_k, u_k):
        s = x_k[0, 0]
        e = x_k[0, 1]
        i = x_k[0, 2]
        n = x_k[0, 3]
        
        b = self.b
        d = self.d
        c = self.c
        e_par = self.e
        g = self.g
        a = self.a
        
        rhs_s = b * n - d * s - c * s * i - u_k * s
        rhs_e = c * s * i - (e_par + d) * e
        rhs_i = e_par * e - (g + a + d) * i
        rhs_n = (b - d) * n - a * i
        
        g_rhs = np.array([rhs_s, rhs_e, rhs_i, rhs_n])
        g_rhs = g_rhs.reshape([1, 4])
        return g_rhs
    
    def lambda_rhs(self, x_k, u_k, lambda_k):
        s = x_k[0, 0]
        e = x_k[0, 1]
        i = x_k[0, 2]
        n = x_k[0, 3]
        
        lambda_s = lambda_k[0, 0]
        lambda_e = lambda_k[0, 1]
        lambda_i = lambda_k[0, 2]
        lambda_n = lambda_k[0, 3]
        
        b = self.b
        d = self.d
        c = self.c
        e_par = self.e
        g = self.g
        a = self.a
        a_w = self.a_w
        
        rhs_ls = lambda_s * (d + c * i + u_k) - c * lambda_e * i
        rhs_le = lambda_e * (e_par + d) - lambda_i * e_par
        rhs_li = -a_w + (lambda_i - lambda_e) * c * s \
                 + lambda_i * (g + a + d) + lambda_n * a
        rhs_ln = - lambda_s * b - lambda_n * (b - d)
        
        rhs_l = np.array([rhs_ls, rhs_le, rhs_li, rhs_ln])
        rhs_l = rhs_l.reshape([1, 4])
        return rhs_l
    
    def optimality_condition(self, x_k, u_k, lambda_k, n_max):
        s = x_k[:, 0]
        lambda_s = lambda_k[:, 0]
        aux = 0.5 * s * lambda_s
        aux_pos = np.max([np.zeros(n_max), aux], axis=0)
        u_aster = np.min([0.9 * np.ones(n_max), aux_pos], axis=0)
        return u_aster
