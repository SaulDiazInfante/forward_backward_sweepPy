#Forward Backward Sweep Method: Mold and Fungcide
# Optimal Control Applied to Biological Models. S. Lenhart, J. Workman.
# Chap 6 Page 63 - 66

import matplotlib.pyplot as plt
import numpy as np


def f(x, u):
    return r * (m - x) - u * x


def lambda_function(lambda_, x):
    return - 2 * a * x + r * lambda_  + 0.5 * (lambda_  ** 2) * x


def runge_kutta_forward(f, u, x_0, h, n_max):
    sol = np.zeros(n_max)
    sol[0] = x_0

    for j in np.arange(n_max - 1):
        x_j = sol[j]
        u_j = u[j]
        u_jp1 = u[j + 1]
        u_mj = 0.5 * (u_j + u_jp1)

        k_1 = f(x_j , u_j)
        k_2 = f(x_j + 0.5 * h * k_1, u_mj)
        k_3 = f(x_j + 0.5 * h * k_2, u_mj)
        k_4 = f(x_j + h * k_3, u_jp1)

        sol[j + 1] = x_j + (h / 6.0) * (k_1 + 2 * k_2 + 2 * k_3 + k_4)

    return sol


def runge_kutta_backward(lambda_function, x, lambda_final, h, n_max):
    sol = np.zeros(n_max)
    sol[-1] = lambda_final

    for j in np.arange(n_max -1, 0, -1):
        lambda_j = sol[j]
        x_j = x[j]
        x_jm1 = x[j - 1]
        x_mj = 0.5 * (x_j + x_jm1)

        k_1 = lambda_function(lambda_j, x_j)
        k_2 = lambda_function(lambda_j - 0.5 * h * k_1, x_mj)
        k_3 = lambda_function(lambda_j - 0.5 * h * k_2, x_mj)
        k_4 = lambda_function(lambda_j - h * k_3, x_jm1)

        sol[j - 1] = lambda_j - (h / 6.0) * (k_1 + 2 * k_2 + 2 * k_3 + k_4)

    return sol


def forward_backward_sweep(f, lambda_function, u, x_0, lambda_final, h, n_max):
    flag = False
    cont = 1
    x = np.zeros(n_max)
    lambda_ = np.zeros(n_max)

    while not(flag):
        u_old = u
        x_old = x
        x = runge_kutta_forward(f, u, x_0, h, n_max)
        lambda_old = lambda_
        lambda_ = runge_kutta_backward(lambda_function, x, lambda_final, h, n_max)

        u_1 = 0.5 * x * lambda_
        u = 0.5 * (u_1 + u_old)
        test_1 = eps * np.linalg.norm(u, 1) - np.linalg.norm(u_old - u, 1)
        test_2 = eps * np.linalg.norm(x, 1) - np.linalg.norm(x_old - x, 1)
        test_3 = eps * np.linalg.norm(lambda_, 1) - np.linalg.norm(lambda_old - lambda_, 1)

        test = np.min([test_1, test_2, test_3])

        flag = (test >= 0.0)
        cont = cont + 1
        print cont, test
    return [x, lambda_, u]


t_0 = 0
t_f = 5
n_max = 5000

t = np.linspace(t_0, t_f, n_max)
h = t[1] - t[0]
x_0 = 1.0
lambda_final = 0
a = 10.0
r = 0.3
m = 10.0

u = np.zeros(n_max)
eps = 0.001

x_wc = runge_kutta_forward(f, u, x_0, h, n_max)

[x, lambda_, u] = forward_backward_sweep(f, lambda_function, u, x_0, lambda_final, h, n_max)

plt.style.use('ggplot')


plt.subplot(3, 1, 1)
plt.plot(t, x)
plt.ylabel('State')

plt.subplot(3, 1, 2)
plt.plot(t, lambda_)
plt.ylabel('Adjoint')

plt.subplot(3, 1, 3)
plt.plot(t, u)
plt.ylabel('Control')
plt.xlabel('Time')

plt.figure()

plt.plot(t, x_wc, '-',
         ms=3,
         lw=1,
         alpha=0.7,
         color='green'
       )
plt.plot(t, x, '--',
         ms=3,
         lw=1,
         alpha=0.7,
         color='orange',
         label='Controled')

plt.ylabel('Bacteria Concentration')
plt.xlabel(r'Time')
plt.legend(loc=0)
axes = plt.gca()
axes.set_xlim([0, 5])
axes.set_ylim([0, 4.5])
plt.show()
