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
    def __init__(self, t_0=0.0, t_f=5.0, dynamics_dim=7, control_dim=2,
                 n_whole=30000, s_zero=76.0 / 120.0,
                 l_zero=36.0 / 120.0, i_zero=4 / 120,
                 l_r_zero=2.0 / 120, i_r_zero=1.0 / 120,
                 r_zero=1.0 / 120
                 ):
        # Parameters for the test example
        self.t_0 = t_0
        self.t_f = t_f
        self.dynamics_dim = dynamics_dim
        self.control_dim = control_dim
        
        #
        self.beta_1 = 13.0
        self.beta_2 = 13.0
        self.beta_3 = 0.0131
        #
        self.mu = 0.0143
        self.d_1 = 0.0
        self.d_2 = 0.0
        self.k_1 = 0.5
        self.k_2 = 1.0
        self.r_1 = 2.0
        self.r_2 = 1.0
        #
        self.p = 0.4
        self.q = 0.1
        self.n_whole = n_whole
        self.lambda_recruitment = self.mu * self.n_whole
        # initial conditions
        self.s_zero = s_zero * n_whole
        self.l_zero = l_zero * n_whole
        self.i_zero = i_zero * n_whole
        self.l_r_zero = l_r_zero * n_whole
        self.i_r_zero = i_r_zero * n_whole
        self.r_zero = r_zero * n_whole
        self.lambda_final = np.zeros([1, dynamics_dim - 1])
        #
        # Functional Cost
        #
        self.b_1 = 50.0
        self.b_2 = 500.0
    
    def set_parameters(self, beta_1, beta_2, beta_3,
                       mu, d_1, d_2, k_1, k_2, r_1, r_2, p, q,
                       n_whole, lambda_recruitment, b_1, b_2,
                       s_zero, l_zero, i_zero,
                       l_r_zero, i_r_zero, r_zero):
        #
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.beta_3 = beta_3
        #
        self.mu = mu
        self.d_1 = d_1
        self.d_2 = d_2
        self.k_1 = k_1
        self.k_2 = k_2
        self.r_1 = r_1
        self.r_2 = r_2
        #
        self.p = p
        self.q = q
        self.lambda_recruitment = lambda_recruitment
        self.b_1 = b_1
        self.b_2 = b_2
        # initial conditions
        self.n_whole = n_whole
        self.s_zero = s_zero * n_whole
        self.l_zero = l_zero * n_whole
        self.i_zero = i_zero * n_whole
        self.l_r_zero = l_r_zero * n_whole
        self.i_r_zero = i_r_zero * n_whole
        self.r_zero = r_zero * n_whole
    
    def g(self, x_k, u_k):
        beta_1 = self.beta_1
        beta_2 = self.beta_2
        beta_3 = self.beta_3
        #
        mu = self.mu
        d_1 = self.d_1
        d_2 = self.d_2
        k_1 = self.k_1
        k_2 = self.k_2
        r_1 = self.r_1
        r_2 = self.r_2
        #
        p = self.p
        q = self.q
        lambda_recruitment = self.lambda_recruitment
        n_whole = self.n_whole
        s = x_k[0, 0]
        l = x_k[0, 1]
        i = x_k[0, 2]
        l_r = x_k[0, 3]
        i_r = x_k[0, 4]
        r = x_k[0, 5]
        u_1 = u_k[0, 0]
        u_2 = u_k[0, 1]
        
        rhs_s = lambda_recruitment - beta_1 * s * (i / n_whole) \
                - beta_3 * s * (i_r / n_whole) - mu * s
        rhs_l = beta_1 * s * (i / n_whole) - (mu + k_1 + u_1 * r_1) * l \
                + (1.0 - u_2) * p * r_2 * i + beta_2 * r * (i / n_whole) \
                - beta_3 * l * (i_r / n_whole)
        rhs_i = k_1 * l - (mu + d_1 + r_2) * i
        
        rhs_l_r = (1.0 - u_2) * q * r_2 * i - (mu + k_2) * l_r \
                  + beta_3 * (s + l + r) * (i_r / n_whole)
        
        rhs_i_r = k_2 * l_r - (mu + d_2) * i_r
        
        term_u2_proportion = (1.0 - (1.0 - u_2) * (p + q)) * r_2 * i
        
        rhs_r = u_1 * r_1 * l + term_u2_proportion \
                - beta_2 * r * (i / n_whole) \
                - beta_3 * r * (i_r / n_whole) \
                - mu * r
        rhs_pop = np.array([rhs_s, rhs_l, rhs_i, rhs_l_r, rhs_i_r, rhs_r])
        rhs_n_whole = rhs_pop.sum()
        rhs = np.array([rhs_s, rhs_l, rhs_i,
                        rhs_l_r, rhs_i_r, rhs_r,
                        rhs_n_whole])
        rhs = rhs.reshape([1, self.dynamics_dim])
        return rhs
    
    def lambda_function(self, x_k, u_k, lambda_k):
        beta_1 = self.beta_1
        beta_2 = self.beta_2
        beta_3 = self.beta_3
        #
        mu = self.mu
        d_1 = self.d_1
        d_2 = self.d_2
        k_1 = self.k_1
        k_2 = self.k_2
        r_1 = self.r_1
        r_2 = self.r_2
        #
        p = self.p
        q = self.q
        n_whole = self.n_whole
        s = x_k[0, 0]
        l = x_k[0, 1]
        i = x_k[0, 2]
        l_r = x_k[0, 3]
        i_r = x_k[0, 4]
        r = x_k[0, 5]
        u_1 = u_k[0, 0]
        u_2 = u_k[0, 1]
        
        lambda_1 = lambda_k[0, 0]
        lambda_2 = lambda_k[0, 1]
        lambda_3 = lambda_k[0, 2]
        lambda_4 = lambda_k[0, 3]
        lambda_5 = lambda_k[0, 4]
        lambda_6 = lambda_k[0, 5]
        
        rhs_l1 = lambda_1 * (beta_1 * (i / n_whole)
                             + beta_3 * (i_r / n_whole) + mu) \
                 - lambda_2 * beta_1 * (i / n_whole) \
                 - lambda_4 * beta_3 * (i_r / n_whole)
        rhs_l2 = lambda_2 * (mu + k_1 + u_1 * r_1 + beta_3 * (i_r / n_whole)) \
                 - lambda_3 * k_1 - lambda_4 * beta_3 * (i_r / n_whole) \
                 - lambda_6 * u_1 * r_1
        rhs_l3 = lambda_1 * beta_1 * (s / n_whole) \
                 - lambda_2 * (beta_1 * (s / n_whole)
                               + (1.0 - u_2) * p * r_2
                               + beta_2 * (r / n_whole)) \
                 + lambda_3 * (mu + d_1 + r_2) \
                 - lambda_4 * (1.0 - u_2) * q * r_2 \
                 - lambda_6 * ((1.0 - (1.0 - u_2) * (p + q)) * r_2
                               - beta_2 * (r / n_whole))
        rhs_l4 = -1.0 + lambda_4 * (mu + k_2) - lambda_5 * k_2
        rhs_l5 = -1.0 + lambda_1 * beta_3 * (s / n_whole) \
                 + lambda_2 * beta_3 * (l / n_whole) \
                 - lambda_4 * beta_3 * (s + l + r) / n_whole \
                 + lambda_5 * (mu + d_2) \
                 + lambda_6 * beta_3 * (r / n_whole)
        rhs_l6 = -lambda_2 * beta_2 * (i / n_whole) \
                 - lambda_4 * beta_3 * (i_r / n_whole) \
                 + lambda_6 * (beta_2 * (i / n_whole)
                               + beta_3 * (i_r / n_whole)
                               + mu)
        rhs_lambda = np.array([rhs_l1, rhs_l2, rhs_l3,
                               rhs_l4, rhs_l5, rhs_l6])
        rhs_lambda = rhs_lambda.reshape([1, self.dynamics_dim - 1])
        return rhs_lambda
    
    def optimality_condition(self, x_k, u_k, lambda_k, n_max):
        b_1 = self.b_1
        b_2 = self.b_2
        p = self.p
        q = self.q
        r_1 = self.r_1
        r_2 = self.r_2
        #
        l = x_k[:, 1]
        i = x_k[:, 2]
        lambda_2 = lambda_k[:, 1]
        lambda_4 = lambda_k[:, 3]
        lambda_6 = lambda_k[:, 5]
        aux_1 = (r_1 / b_1) * (lambda_2 - lambda_6) * l
        aux_2 = (r_2 / b_2) * (p * lambda_2 + q * lambda_4
                               - (p + q) * lambda_6) * i
        positive_part_1 = np.max([0.05 * np.ones(n_max), aux_1], axis=0)
        positive_part_2 = np.max([0.05 * np.ones(n_max), aux_2], axis=0)
        u_aster_1 = np.min([positive_part_1, 0.95 * np.ones(n_max)], axis=0)
        u_aster_2 = np.min([positive_part_2, 0.95 * np.ones(n_max)], axis=0)
        u_aster = np.zeros([n_max, 2])
        u_aster[:, 0] = u_aster_1
        u_aster[:, 1] = u_aster_2
        return u_aster
