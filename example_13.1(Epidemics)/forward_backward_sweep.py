import numpy as np
from example_13_1 import OptimalControlProblem

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
    
    def __init__(self, eps=.001, n_max=5000):
        """
        :type t_0: initial time
        """
        #
        super(ForwardBackwardSweep, self).__init__()
        self.n_max = n_max
        self.eps = eps
        self.t = np.linspace(self.t_0, self.t_f, n_max)
        self.h = self.t[1] - self.t[0]
        self.x = np.zeros([n_max, 4])
        self.u = np.zeros(n_max)
        self.lambda_adjoint = np.zeros([n_max, 4])
    
    def runge_kutta_forward(self, u):
        s_zero = self.s_zero
        e_zero = self.e_zero
        i_zero = self.i_zero
        n_zero = self.n_zero
        x_zero = np.array([s_zero, e_zero, i_zero, n_zero])
        x_zero = x_zero.reshape([1, 4])
        
        h = self.h
        n_max = self.n_max
        sol = np.zeros([n_max, 4])
        sol[0] = x_zero
        #
        for j in np.arange(n_max - 1):
            x_j = sol[j].reshape([1, 4])
            u_j = u[j]
            u_jp1 = u[j + 1]
            u_mj = 0.5 * (u_j + u_jp1)
            
            k_1 = self.g_ode(x_j, u_j)
            k_2 = self.g_ode(x_j + 0.5 * h * k_1, u_mj)
            k_3 = self.g_ode(x_j + 0.5 * h * k_2, u_mj)
            k_4 = self.g_ode(x_j + h * k_3, u_jp1)
            
            sol[j + 1] = x_j + (h / 6.0) * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
        self.x = sol
        return sol
    
    def runge_kutta_backward(self, x, u):
        h = self.h
        n_max = self.n_max
        sol = np.zeros([n_max, 4])
        #
        for j in np.arange(n_max - 1, 0, -1):
            lambda_j = sol[j].reshape([1, 4])
            x_j = x[j].reshape([1, 4])
            x_jm1 = x[j - 1].reshape([1, 4])
            x_mj = 0.5 * (x_j + x_jm1)
            u_j = u[j]
            u_jm1 = u[j - 1]
            u_mj = 0.5 * (u_j + u_jm1)
            
            k_1 = self.lambda_rhs(x_j, u_j, lambda_j)
            k_2 = self.lambda_rhs(x_mj, u_mj, lambda_j - 0.5 * h * k_1)
            k_3 = self.lambda_rhs(x_mj, u_mj, lambda_j - 0.5 * h * k_2)
            k_4 = self.lambda_rhs(x_jm1, u_jm1, lambda_j - h * k_3)
            
            sol[j - 1] = lambda_j - (h / 6.0) * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
        self.lambda_adjoint = sol
        return sol
    
    def forward_backward_sweep(self):
        flag = True
        cont = 1
        eps = self.eps
        x = self.x
        u = self.u
        lambda_ = self.lambda_adjoint
        n_max = self.n_max
        #
        while flag:
            u_old = u
            x_old = x
            x = self.runge_kutta_forward(u)
            lambda_old = lambda_
            lambda_ = self.runge_kutta_backward(x, u)
            u_1 = self.optimality_condition(x, u, lambda_, n_max)
            u = 0.5 * (u_1 + u_old)
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
