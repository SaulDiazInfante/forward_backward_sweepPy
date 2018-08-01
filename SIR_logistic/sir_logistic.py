# coding=utf-8
import numpy as np

"""
    Here we reproduce the simulation of
    [1].
    
    The optimal control problem reads:
    \begin{equation}
    \int_{0}^T
        \left[
            B_1 I(t)
            + B_2 \left[\frac{R(t)}{K}\right]^m [u_1(t)]^2 + B_3 [u_2(t)]^2
        \right] dt,
        \qquad  m\geq 1,
    \end{equation}
    subject to
    \begin{equation}
        \begin{aligned}
            \frac{dS}{dt} &=
                \mu N
                - \beta \frac{S I}{N}
                - \mu \frac{N}{K} S - u_1(t) S,
            \\
            \frac{dI}{dt} &=
                \beta \frac{S I}{N}
                - (\gamma  + \mu) I
                - \mu \frac{N}{K} I
                - u_2(t) I,
        \\
        \frac{dR}{dt} &=
            \gamma I
            - \mu \frac{N}{K} R
            + u_1(t) S
            + u_2(t) I,
        \\
        S(0) &= S_0, \quad
        I(0) = I_0, \quad
        R(0) = R_0. \quad
        \end{aligned}
    \end{equation}

    [1] Elsa Schaefer and Holly Gaff. Optimal control applied to vaccination
    and treatment strategies for various epidemiological models. Mathematical
    Biosciences and Engineering, 6(3):469–492, jun 2009. ISSN 1551-0018.
    doi: 10.3934/mbe.2009.6.469.
    URL http://www.aimsciences.org/journals/displayArticles.jsp?paperID=4251.
"""


class OptimalControlProblem(object):
    def __init__(self, t_0=0.0, t_f=25.0, dynamics_dim=3, control_dim=2,
                 s_zero=4500, i_zero=499, r_zero=1
                 ):
        # Parameters for the test example
        self.t_0 = t_0
        self.t_f = t_f
        self.dynamics_dim = dynamics_dim
        self.control_dim = control_dim
        #
        self.k = 5000
        self.mu = 0.00004
        self.delta = 0.0
        self.beta = 0.05
        self.gamma = 0.1
        self.omega = .001
        self.epsilon = 0.1
        self.m = 10
        #
        
        #
        # initial conditions
        self.s_zero = s_zero
        self.i_zero = i_zero
        self.r_zero = r_zero
        self.n_whole = s_zero + i_zero + r_zero
        self.lambda_final = np.zeros([1, dynamics_dim])
        #
        # Functional Cost
        #
        self.b_1 = 1.0
        self.b_2 = 1000
        self.b_3 = 1000
        self.u_1_lower = 0.00
        self.u_1_upper = 0.1
        self.u_2_lower = 0.00
        self.u_2_upper = 0.6
    
    def set_parameters(self, k, mu, delta, beta, gamma, omega, epsilon, m,
                       b_1, b_2, b_3,
                       s_zero, i_zero, r_zero):
        #
        self.k = k
        self.mu = mu
        self.delta = delta
        self.beta = beta
        self.gamma = gamma
        self.omega = omega
        self.epsilon = epsilon
        self.m = m
        self.b_1 = b_1
        self.b_2 = b_2
        self.b_3 = b_3
        self.s_zero = s_zero
        self.i_zero = i_zero
        self.r_zero = r_zero
    
    def g(self, x_k, u_k):
        k = self.k
        mu = self.mu
        delta = self.delta
        beta = self.beta
        gamma = self.gamma
        omega = self.omega
        epsilon = self.epsilon
        b_1 = self.b_1
        b_2 = self.b_2
        b_3 = self.b_3
        
        s = x_k[0, 0]
        i = x_k[0, 1]
        r = x_k[0, 2]
        n_whole = s + i + r
        u_1 = u_k[0, 0]
        u_2 = u_k[0, 1]
        
        rhs_s = mu * n_whole - beta * s * i / n_whole - u_1 * s \
                - mu * n_whole * s / k
        rhs_i = beta * s * i / n_whole - (gamma + u_2 + delta) * i \
                - mu * n_whole * i / k
        
        rhs_r = (gamma + u_2) * i + u_1 * s - - mu * n_whole * r / k
        
        rhs_pop = np.array([rhs_s, rhs_i, rhs_r])
        self.n_whole = n_whole
        rhs_pop = rhs_pop.reshape([1, self.dynamics_dim])
        return rhs_pop
    
    def lambda_function(self, x_k, u_k, lambda_k):
        k = self.k
        mu = self.mu
        delta = self.delta
        beta = self.beta
        gamma = self.gamma
        omega = self.omega
        epsilon = self.epsilon
        b_1 = self.b_1
        b_2 = self.b_2
        b_3 = self.b_3
        
        s = x_k[0, 0]
        i = x_k[0, 1]
        r = x_k[0, 2]
        n_whole = s + i + r
        u_1 = u_k[0, 0]
        u_2 = u_k[0, 1]
        
        lambda_1 = lambda_k[0, 0]
        lambda_2 = lambda_k[0, 1]
        lambda_3 = lambda_k[0, 2]
        m = self.m
        
        rhs_l_1 = mu / k * (lambda_1 * (s + n_whole - k) + lambda_2 * i
                            + lambda_3 * r) \
                  + beta * (i * (n_whole - s) / (n_whole ** 2)) \
                  * (lambda_1 - lambda_2) + u_1 * (lambda_1 - lambda_3)
        
        rhs_l_2 = -b_1 + mu / k * (lambda_1 * (s - k)
                                   + lambda_2 * (n_whole + i) + lambda_3 * r) \
                  + beta * (s * (n_whole - i) / (n_whole ** 2)) \
                  * (lambda_1 - lambda_2) + delta * lambda_2 \
                  + (gamma + u_2) * (lambda_2 - lambda_3)
        
        rhs_l_3 = m * b_2 * (u_1 ** 2) * (r ** (m - 1)) / (k ** m) \
                  + mu / k * ((s - k) * lambda_1 + i * lambda_2
                              + (n_whole + r) * lambda_3) \
                  + beta * s * i / (n_whole ** 2) * (lambda_2 - lambda_1)
        #
        #
        #
        rhs_l = np.array([rhs_l_1, rhs_l_2, rhs_l_3])
        rhs_l = rhs_l.reshape([1, 3])
        return rhs_l
    
    def optimality_condition(self, x_k, u_k, lambda_k, n_max):
        u_1_lower = self.u_1_lower
        u_2_lower = self.u_2_lower
        u_1_upper = self.u_1_upper
        u_2_upper = self.u_2_upper
        m = self.m
        k = self.k
        b_2 = self.b_2
        b_3 = self.b_3
        
        #
        s = x_k[:, 0]
        i = x_k[:, 1]
        r = x_k[:, 2]
        lambda_1 = lambda_k[:, 0]
        lambda_2 = lambda_k[:, 1]
        lambda_3 = lambda_k[:, 2]
        
        aux_1 = s * (lambda_1 - lambda_3) / (2 * b_2 * ((r / k) ** m))
        aux_2 = i * (lambda_2 - lambda_3) / (2 * b_3)
        
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
