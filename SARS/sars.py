import numpy as np

"""
    Here we reproduce the simulation of
    Optimal and sub-optimal quarantine and isolation control in SARS epidemics.
    Xiefei Yan, Yun Zou.
    
    The optimal control problem reads:
    \begin{equation}
        \begin{aligned}
            \dfrac{dS}{dt} &=
                \Lambda 
                -\dfrac{
                    S
                    \left(
                        \beta I 
                        + \mathcal{E}_E  \beta E
                        + \mathcal{E}_Q  \beta Q
                        + \mathcal{E}_J  \beta J
                    \right)
                }{N}
                - \mu S,
            \\
            \dfrac{dE}{dt} &=
                p +
                \dfrac{
                    \beta S
                    \left(
                        \beta I 
                            + \mathcal{E}_E \beta E
                            + \mathcal{E}_Q \beta Q
                            + \mathcal{E}_J \beta J
                    \right)
                }{N}
                -(
                    u_1(t) + k_1 + \mu
                )E,
            \\
            \dfrac{dQ}{dt} &=
                u_1(t) E 
                - (k_2 + \mu) Q
            \\
            \dfrac{dI}{dt} &=
                k_1 E 
                -(u_2(t) + d_1  + \sigma_1 + \mu) I,
            \\
            \dfrac{dJ}{dt} &=
                u_2(t) I 
                + k_2 Q
                - (d_2 + \sigma_2 + \mu) J,
            \\
            \dfrac{dR}{dt} &=
                \sigma_1 I
                +\sigma_2 J
                - \mu R.
        \end{aligned}
    \end{equation}
"""


class OptimalControlProblem(object):
    def __init__(self, t_0=0.0, t_f=365.0, dynamics_dim=6, control_dim=2,
                 s_zero=12000000, e_zero=1565, q_zero=292,
                 i_zero=695, j_zero=326, r_zero=20
                 ):
        # Parameters for the test example
        self.t_0 = t_0
        self.t_f = t_f
        self.dynamics_dim = dynamics_dim
        self.control_dim = control_dim
        self.n_whole = s_zero + e_zero + q_zero + i_zero + j_zero + r_zero
        #
        self.beta = 0.2
        self.e_e = 0.3
        self.e_q = 0.0
        self.e_j = 0.1
        
        #
        self.mu = 0.000034
        self.lambda_recruitment = self.n_whole * self.mu
        
        self.p = 0.0
        self.k_1 = 0.1
        self.k_2 = 0.125
        self.d_1 = 0.0079
        self.d_2 = 0.0068
        #
        self.sigma_1 = 0.0337
        self.sigma_2 = 0.0386
        #
        # initial conditions
        self.s_zero = s_zero
        self.e_zero = e_zero
        self.q_zero = q_zero
        self.i_zero = i_zero
        self.j_zero = j_zero
        self.r_zero = r_zero
        self.lambda_final = np.zeros([1, dynamics_dim])
        #
        # Functional Cost
        #
        self.b_1 = 1.0
        self.b_2 = 1.0
        self.b_3 = 1.0
        self.b_4 = 1.0
        self.c_1 = 300.0
        self.c_2 = 600.0
        self.u_1_lower = 0.05
        self.u_1_upper = 0.5
        self.u_2_lower = 0.05
        self.u_2_upper = 0.5
    
    def set_parameters(self, beta, e_e, e_q, e_j,
                       mu, p, k_1, k_2, d_1, d_2, sigma_1, sigma_2,
                       n_whole, b_1, b_2, b_3, b_4, c_1, c_2,
                       s_zero, e_zero, q_zero,
                       i_zero, j_zero, r_zero):
        #
        self.beta = beta
        self.e_e = e_e
        self.e_q = e_q
        self.e_j = e_j
        
        #
        self.mu = mu
        self.lambda_recruitment = self.n_whole * self.mu
        
        self.p = p
        self.k_1 = k_1
        self.k_2 = k_2
        self.d_1 = d_1
        self.d_2 = d_2
        #
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        #
        # initial conditions
        self.s_zero = s_zero
        self.e_zero = e_zero
        self.q_zero = q_zero
        self.i_zero = i_zero
        self.j_zero = j_zero
        self.r_zero = r_zero
        self.n_whole = n_whole
        #
        # Functional Cost
        #
        self.b_1 = b_1
        self.b_2 = b_2
        self.b_3 = b_3
        self.b_4 = b_4
        self.c_1 = c_1
        self.c_2 = c_2
    
    def g(self, x_k, u_k):
        beta = self.beta
        e_e = self.e_e
        e_q = self.e_q
        e_j = self.e_j
        #
        mu = self.mu
        k_1 = self.k_1
        k_2 = self.k_2
        d_1 = self.d_1
        d_2 = self.d_2
        
        sigma_1 = self.sigma_1
        sigma_2 = self.sigma_2
        #
        p = self.p
        lambda_recruitment = self.lambda_recruitment
        n_whole = self.n_whole
        s = x_k[0, 0]
        e = x_k[0, 1]
        q = x_k[0, 2]
        i = x_k[0, 3]
        j = x_k[0, 4]
        r = x_k[0, 5]
        u_1 = u_k[0, 0]
        u_2 = u_k[0, 1]
        
        rhs_s = lambda_recruitment - (s / n_whole) \
                * (beta * i
                   + e_e * beta * e
                   + e_q * beta * q
                   + e_j * beta * j) - mu * s
        
        rhs_e = p + (s / n_whole) \
                * (beta * i
                   + e_e * beta * e
                   + e_q * beta * q
                   + e_j * beta * j) \
                - (u_1 + k_1 + mu) * e
        
        rhs_q = u_1 * e - (k_2 + mu) * q
        
        rhs_i = k_1 * e - (u_2 + d_1 + sigma_1 + mu) * i
        
        rhs_j = u_2 * i + k_2 * q - (d_2 + sigma_2 + mu) * j
        
        rhs_r = sigma_1 * i + sigma_2 * j - mu * r
        
        rhs_pop = np.array([rhs_s, rhs_e, rhs_q, rhs_i, rhs_j, rhs_r])
        rhs_n_whole = rhs_pop.sum()
        # self.n_whole = rhs_n_whole
        rhs_pop = rhs_pop.reshape([1, self.dynamics_dim])
        return rhs_pop
    
    def lambda_function(self, x_k, u_k, lambda_k):
        beta = self.beta
        e_e = self.e_e
        e_q = self.e_q
        e_j = self.e_j
        #
        mu = self.mu
        k_1 = self.k_1
        k_2 = self.k_2
        d_1 = self.d_1
        d_2 = self.d_2
        
        sigma_1 = self.sigma_1
        sigma_2 = self.sigma_2
        #
        p = self.p
        lambda_recruitment = self.lambda_recruitment
        n_whole = self.n_whole
        s = x_k[0, 0]
        e = x_k[0, 1]
        q = x_k[0, 2]
        i = x_k[0, 3]
        j = x_k[0, 4]
        r = x_k[0, 5]
        u_1 = u_k[0, 0]
        u_2 = u_k[0, 1]
        b_1 = self.b_1
        b_2 = self.b_2
        b_3 = self.b_3
        b_4 = self.b_4
        c_1 = self.c_1
        c_2 = self.c_2
        
        lambda_1 = lambda_k[0, 0]
        lambda_2 = lambda_k[0, 1]
        lambda_3 = lambda_k[0, 2]
        lambda_4 = lambda_k[0, 3]
        lambda_5 = lambda_k[0, 4]
        lambda_6 = lambda_k[0, 5]
        
        common_ter = (beta * i
                      + e_e * beta * e
                      + e_q * beta * q
                      + e_j * beta * j)
        rhs_l_1 = (lambda_1 - lambda_2) \
                  * common_ter * (n_whole - s) / (n_whole ** 2)
        
        rhs_l_2 = -b_1 - lambda_1 \
                  * (mu
                     - (e_e * beta * s * n_whole - s * common_ter)
                     / (n_whole ** 2)
                     ) \
                  - lambda_2 * ((e_e * beta * s * n_whole - s * common_ter)
                                / (n_whole ** 2)
                                - (u_1 + k_1 + mu)) \
                  - lambda_3 * u_1 - lambda_4 * k_1
        
        rhs_l_3 = -b_2 - lambda_1 \
                  * (mu
                     - (e_e * beta * s * n_whole - s * common_ter)
                     / (n_whole ** 2)
                     ) \
                  - lambda_2 * ((e_q * beta * s * n_whole - s * common_ter)
                                / (n_whole ** 2)) \
                  + lambda_3 * (k_2 + mu) - lambda_5 * k_2
        
        rhs_l_4 = -b_3 - lambda_1 \
                  * (mu
                     - (beta * s * n_whole - s * common_ter)
                     / (n_whole ** 2)
                     ) \
                  - lambda_2 * ((beta * s * n_whole - s * common_ter)
                                / (n_whole ** 2)) \
                  + lambda_4 * (u_2 + d_1 + sigma_1 + mu) \
                  - lambda_5 * u_2 - lambda_6 * sigma_1
        
        rhs_l_5 = -b_4 - lambda_1 \
                  * (mu
                     - (e_j * beta * s * n_whole - s * common_ter)
                     / (n_whole ** 2)
                     ) \
                  - lambda_2 * ((e_j * beta * s * n_whole - s * common_ter)
                                / (n_whole ** 2)) \
                  + lambda_5 * (d_2 + sigma_2 + mu) \
                  - lambda_6 * sigma_2
        
        rhs_l_6 = lambda_1 * (mu - (s * common_ter) / (n_whole ** 2)) \
                  + lambda_2 * (s * common_ter) / (n_whole ** 2) \
                  + lambda_6 * mu
        
        rhs_l = np.array([rhs_l_1, rhs_l_2, rhs_l_3, rhs_l_4, rhs_l_5, rhs_l_6])
        rhs_l = rhs_l.reshape([1, 6])
        return rhs_l
    
    def optimality_condition(self, x_k, u_k, lambda_k, n_max):
        u_1_lower = self.u_1_lower
        u_2_lower = self.u_2_lower
        u_1_upper = self.u_1_upper
        u_2_upper = self.u_2_upper
        c_1 = self.c_1
        c_2 = self.c_2
        
        #
        e = x_k[:, 1]
        i = x_k[:, 3]
        lambda_2 = lambda_k[:, 1]
        lambda_3 = lambda_k[:, 2]
        lambda_4 = lambda_k[:, 3]
        lambda_5 = lambda_k[:, 4]
        aux_1 = (1.0 / c_1) * (lambda_2 - lambda_3) * e
        aux_2 = (1.0 / c_2) * (lambda_4 - lambda_5) * i
        
        positive_part_1 = np.max([u_1_lower * np.ones(n_max), aux_1], axis=0)
        positive_part_2 = np.max([u_2_lower * np.ones(n_max), aux_2], axis=0)
        
        u_aster_1 = np.min([positive_part_1, u_1_upper * np.ones(n_max)],
                           axis=0)
        u_aster_2 = np.min([positive_part_2, u_2_upper * np.ones(n_max)],
                           axis=0)
        
        u_aster = np.zeros([n_max, 2])
        u_aster[:, 0] = u_aster_1
        u_aster[:, 1] = u_aster_2
        return u_aster
