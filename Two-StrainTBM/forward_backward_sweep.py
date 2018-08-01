import numpy as np
from two_strain_tbm import OptimalControlProblem

"""
Forward-Backward Sweep Method for the problem
    \begin{align}
        \max_{u} &
            \int_{t_0}^{t_1}
                f(t, x(t), u(t)) dt
        \\
        \text{s.t. }
            x'(t) & = g (t, x(t), u(t),x(t_0)), \qquad x(t_0) = a.
    \end{align}

    Check the Lenhart's book Optimal Control Applied to Biological Models [1]
    as main reference.
    Modify the methods: set_parameters, g, and f  in the problem class to 
    adapt the class ForwardBackwardMethod via inheritance.
"""


class ForwardBackwardSweep(OptimalControlProblem):
    
    def __init__(self, eps=.0001, n_max=50):
        """

        :type t_0: initial time
        """
        #
        super(ForwardBackwardSweep, self).__init__()
        self.n_max = n_max
        self.eps = eps
        self.t = np.linspace(self.t_0, self.t_f, n_max)
        self.h = self.t[1] - self.t[0]
        dyn_dim = self.dynamics_dim
        con_dim = self.control_dim
        self.x = np.zeros([n_max, dyn_dim])
        self.u = np.zeros([n_max, con_dim])
        # self.u[:, 1] = np.ones(n_max)
        self.lambda_adjoint = np.zeros([n_max, dyn_dim - 1])
    
    def runge_kutta_forward(self, u):
        x_0 = np.array([self.s_zero, self.l_zero, self.i_zero,
                        self.l_r_zero, self.i_r_zero, self.r_zero,
                        self.n_whole])
        h = self.h
        n_max = self.n_max
        dyn_dim = self.dynamics_dim
        con_dim = self.control_dim
        sol = np.zeros([n_max, dyn_dim])
        sol[0] = x_0
        #
        for j in np.arange(n_max - 1):
            x_j = sol[j].reshape([1, dyn_dim])
            u_j = u[j].reshape([1, con_dim])
            u_jp1 = u[j + 1].reshape([1, con_dim])
            u_mj = 0.5 * (u_j + u_jp1)
            
            k_1 = self.g(x_j, u_j)
            k_2 = self.g(x_j + 0.5 * h * k_1, u_mj)
            k_3 = self.g(x_j + 0.5 * h * k_2, u_mj)
            k_4 = self.g(x_j + h * k_3, u_jp1)
            
            sol[j + 1] = x_j + (h / 6.0) * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
        self.x = sol
        return sol
    
    def runge_kutta_backward(self, x, u):
        lambda_final = self.lambda_final
        h = self.h
        n_max = self.n_max
        dyn_dim_l = self.dynamics_dim - 1
        dyn_dim_x = self.dynamics_dim
        con_dim = self.control_dim
        sol = np.zeros([n_max, dyn_dim_l])
        sol[-1] = lambda_final
        #
        for j in np.arange(n_max - 1, 0, -1):
            lambda_j = sol[j].reshape([1, dyn_dim_l])
            x_j = x[j].reshape([1, dyn_dim_x])
            x_jm1 = x[j - 1].reshape([1, dyn_dim_x])
            x_mj = 0.5 * (x_j + x_jm1)
            u_j = u[j].reshape([1, con_dim])
            u_jm1 = u[j - 1].reshape([1, con_dim])
            u_mj = 0.5 * (u_j + u_jm1)
            #
            k_1 = self.lambda_function(x_j, u_j, lambda_j)
            k_2 = self.lambda_function(x_mj, u_mj, lambda_j - 0.5 * h * k_1)
            k_3 = self.lambda_function(x_mj, u_mj, lambda_j - 0.5 * h * k_2)
            k_4 = self.lambda_function(x_jm1, u_jm1, lambda_j - h * k_3)
            iter = lambda_j - (h / 6.0) * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
            sol[j - 1] = iter  # lambda_j - (h / 6.0) * (k_1 + 2 * k_2 + 2 *
            # k_3 + k_4)
        self.lambda_adjoint = sol
        return sol
    
    def forward_backward_sweep(self):
        flag = True
        cont = 1
        eps = self.eps
        x = self.x
        n_max = self.n_max
        u = self.u + 0.95 * np.ones([n_max, 2])
        lambda_ = self.lambda_adjoint
        #
        while flag:
            u_old = u
            x_old = x
            lambda_old = lambda_
            x = self.runge_kutta_forward(u)
            lambda_ = self.runge_kutta_backward(x, u)
            u_1 = self.optimality_condition(x, u, lambda_, n_max)
            alpha = 0.4
            u = alpha * u_1 + (1.0 - alpha) * u_old
            test_1 = np.linalg.norm(u_old - u, 1) * (
                    np.linalg.norm(u, 1) ** (-1))
            test_2 = np.linalg.norm(x_old - x, 1) * (
                    np.linalg.norm(x, 1) ** (-1))
            test_3 = np.linalg.norm(lambda_old - lambda_, 1) * (
                    np.linalg.norm(lambda_, 1) ** (-1))
            #
            test = np.max([test_1, test_2, test_3])
            flag = (test > eps)
            cont = cont + 1
            print cont, test
        return [x, lambda_, u]
