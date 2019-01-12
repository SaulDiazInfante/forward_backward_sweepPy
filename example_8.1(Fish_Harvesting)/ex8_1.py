# Optimal Control Applied to Biological Models. S. Lenhart, J. Workman.
import matplotlib.pyplot as plt
import numpy as np


def g(x_i, u_i):
    g_xu = np.log(k) + np.log(x_i) + np.log(np.abs(m - x_i))
    g_xu = np.exp(g_xu) - u_i * x_i
    return g_xu


def lambda_function(lambda_i, x_i, u_i):
    l = np.log(p_1)  +  np.log(u_i - 2) + np.log( p_2) +2.0 * np.log(u_i) + np.log(x_i) \
        + np.log(lambda_i) + np.log( (k * m - 2 * k * x_i - u_i)
    l = - l
    return l


def runge_kutta_forward(g, u, x_0, h, n_max):
    sol = np.zeros(n_max)
    sol[0] = x_0
    for j in np.arange(n_max - 1):
        x_j = sol[j]
        u_j = u[j]
        u_jp1 = u[j + 1]
        u_mj = 0.5 * (u_j + u_jp1)
        #
        k_1 = g(x_j, u_j)
        k_2 = g(x_j + 0.5 * h * k_1, u_mj)
        k_3 = g(x_j + 0.5 * h * k_2, u_mj)
        k_4 = g(x_j + h * k_3, u_jp1)
        sol_jp1 = x_j + (h / 6.0) * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
        sol[j + 1] = sol_jp1
    return sol


def runge_kutta_backward(lambda_function, x, u, lambda_final, h, n_max):
    sol = np.zeros(n_max)
    sol[-1] = lambda_final
    #
    for j in np.arange(n_max - 1, 0, -1):
        lambda_j = sol[j]
        x_j = x[j]
        x_jm1 = x[j - 1]
        x_mj = 0.5 * (x_j + x_jm1)
        #
        u_j = u[j]
        u_jm1 = u[j - 1]
        u_mj = 0.5 * (u_j + u_jm1)
        
        k_1 = lambda_function(lambda_j, x_j, u_j)
        k_2 = lambda_function(lambda_j - 0.5 * h * k_1, x_mj, u_mj)
        k_3 = lambda_function(lambda_j - 0.5 * h * k_2, x_mj, u_mj)
        k_4 = lambda_function(lambda_j - h * k_3, x_jm1, u_jm1)
        sol[j - 1] = lambda_j - (h / 6.0) * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
    return sol


def forward_backward_sweep(g, lambda_function, u, x_0, lambda_final, h, n_max):
    flag = True
    cont = 1
    x = np.zeros(n_max)
    lambda_ = np.zeros(n_max)
    #
    while flag:
        u_old = u
        x_old = x
        x = runge_kutta_forward(g, u, x_0, h, n_max)
        lambda_old = lambda_
        lambda_ = runge_kutta_backward(lambda_function,
                                       x, u, lambda_final, h, n_max)
        
        num_u1 = (-1.0) * lambda_ * x + p_1 * x - c * np.ones(n_max)
        den_u1 = 2.0 * p_2 * (x ** 2) + np.finfo(np.float64).eps
        u_1 = num_u1 / den_u1
        alpha = .5
        u = alpha * u_old + (1.0 - alpha) * u_1
        '''
        test_1 = np.linalg.norm(u_old - u, 1) * (np.linalg.norm(u, 1) ** (-1))
        test_2 = np.linalg.norm(x_old - x, 1) * (np.linalg.norm(x, 1) ** (-1))
        test_3 = np.linalg.norm(lambda_old - lambda_, 1) * \
            (np.linalg.norm(lambda_, 1) ** (-1))
        test = np.max([test_1, test_2, test_3])
        '''
        test_1 = np.linalg.norm(u_old - u, 1) / np.linalg.norm(u, 1)
        test_2 = np.linalg.norm(x_old - x, 1) / np.linalg.norm(x, 1)
        test_3 = np.linalg.norm(lambda_old - lambda_, 1) / \
                 np.linalg.norm(lambda_, 1)
        test = np.max([test_1, test_2, test_3])
        flag = (test >= eps)
        cont = cont + 1
        print cont, test, '\t convergence:', not flag
    return [x, lambda_, u]


t_0 = 0.0
t_f = 5.0
n_max = 100000
t = np.linspace(t_0, t_f, n_max)
h = t[2] - t[1]
#
x_0 = 0.5
k = 0.25
m = 10.0
p_1 = 2.0
p_2 = 1.0
c = 1.0
lambda_final = 0.0
u = np.zeros(n_max) + .001
# u = 0.5*np.ones(n_max)
# u = np.random.rand(n_max)

eps = 0.1
#
# x_uncontrol = runge_kutta_forward(g, u, x_0, h, n_max)
#
[x, lambda_, u] = forward_backward_sweep(g, lambda_function, u, x_0,
                                         lambda_final, h, n_max)
plt.plot(t, u, '-', ms=3, lw=1, alpha=0.7, mfc='blue', label='M=1.0')
m = 1.0
[x, lambda_, u] = forward_backward_sweep(g, lambda_function, u, x_0,
                                         lambda_final, h, n_max)
plt.plot(t, u, '-', ms=3, lw=1, alpha=0.7, mfc='blue', label='M=10.0')
plt.plot(t, np.zeros(u.shape[0]), 'k-', lw=1, alpha=0.7,
                     label='M=10.0')
plt.axis([0, 5.0, -.6, .4])
plt.xlabel('Time')
plt.ylabel('Optimal Control (harvesting proportion)')
plt.legend(loc=0)
plt.show()
