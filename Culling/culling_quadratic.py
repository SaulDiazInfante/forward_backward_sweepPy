import numpy as np

"""
    Here we present the model reported in \cite*{Bolzoni2014} to describe a
    outbreak of bovine tuberculosis. The regarding uncontrolled model reads
    \begin{equation}
        \begin{aligned}
            \min_{u(t)\in \mathcal{U}}
            &
            \int_0^T
                I(t) + P [u(t)]^{\theta}, \quad \theta \in \{1,2\},
                \quad P = B/A
            \\
            \textrm{subject to:} &
            \\
            &\dfrac{dS}{dt} =
            r S
            \left (
                1 - \dfrac{S+I}{K}
            \right)
            - \beta SI - u(t) S
            \\
            &\dfrac{dI}{dt} =
            \beta SI - (\alpha + \mu + u(t)) I.
        \end{aligned}
    \end{equation}
"""


class QuadraticOptimalControlProblem(object):
    def __init__(self, t_0=0.0, t_f=15.0, dynamics_dim=2, control_dim=1,
                 s_zero=84, i_zero=1.0):
        # Parameters for the test example
        self.t_0 = t_0
        self.t_f = t_f
        self.dynamics_dim = dynamics_dim
        self.control_dim = control_dim
        self.n_whole = s_zero + i_zero
        #
        self.nu = 0.6
        self.mu = 0.4
        self.k = 84
        self.alpha = 0.05
        self.theta = 2
        self.r_zero = 6.0
        self.beta = self.r_zero * (self.alpha + self.mu) / self.k
        
        #
        # initial conditions
        self.s_zero = self.k
        self.i_zero = i_zero
        self.lambda_final = np.zeros([1, dynamics_dim])
        #
        # Functional Cost
        #
        self.p = 40.0
        self.u_1_lower = 0.00
        self.u_1_upper = 0.1
    
    def set_parameters(self, nu, mu, k, alpha, theta, r_zero, i_zero, p):
        #
        self.nu = nu
        self.mu = mu
        self.k = k
        self.alpha = alpha
        self.theta = theta
        self.r_zero = r_zero
        self.beta = self.r_zero * (alpha + mu) / self.k
        #
        # initial conditions
        self.s_zero = self.k
        self.i_zero = i_zero
        self.p = p
    
    def g(self, x_k, u_k):
        k = self.k
        alpha = self.alpha
        beta = self.beta
        mu = self.mu
        r = self.nu - self.mu
        s = x_k[0, 0]
        i = x_k[0, 1]
        u = u_k[0]
        #
        rhs_s = r * s * (1.0 - (s + i) / k) - beta * s * i - u * s
        #
        rhs_i = beta * s * i - (alpha + mu + u) * i
        rhs_pop = np.array([rhs_s, rhs_i])
        rhs_pop = rhs_pop.reshape([1, self.dynamics_dim])
        return rhs_pop
    
    def lambda_function(self, x_k, u_k, lambda_k):
        k = self.k
        alpha = self.alpha
        beta = self.beta
        mu = self.mu
        r = self.nu - mu
        s = x_k[0, 0]
        i = x_k[0, 1]
        u = u_k[0]
        lambda_1 = lambda_k[0, 0]
        lambda_2 = lambda_k[0, 1]
        
        rhs_l_1 = lambda_1 * (r * ((2 * s + i) / k - 1.0) + beta * i + u) \
                  - lambda_2 * beta * i
        rhs_l_2 = lambda_1 * s * (beta + r / k) \
                  + lambda_2 * (alpha + mu + u - beta * s) - 1.0
        rhs_l = np.array([rhs_l_1, rhs_l_2])
        rhs_l = rhs_l.reshape([1, 2])
        return rhs_l
    
    def optimality_condition(self, x_k, u_k, lambda_k, n_max):
        u_lower = self.u_1_lower
        u_upper = self.u_1_upper
        p = self.p
        #
        s = x_k[:, 0]
        i = x_k[:, 1]
        lambda_1 = lambda_k[:, 0]
        lambda_2 = lambda_k[:, 1]
        aux = (1.0 / (2.0 * p)) * (lambda_1 * s + lambda_2 * i)
        right_bound = np.min([u_upper * np.ones(n_max), aux], axis=0)
        #
        u_aster = np.max([right_bound, u_lower * np.ones(n_max)], axis=0)
        return u_aster.reshape([n_max, 1])
