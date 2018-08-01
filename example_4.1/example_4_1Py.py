import matplotlib.pyplot as plt
import numpy as np
from forward_backward_sweep import ForwardBackwardSweep

a = 1.0
b = 1.0
c = 4.0
x_zero = 1.0
lambda_final = 0.0

fbsm = ForwardBackwardSweep()
fbsm.set_parameters(a, b, c, x_zero, lambda_final)
x_wc = fbsm.runge_kutta_forward(fbsm.u)
[x, lambda_, u] = fbsm.forward_backward_sweep()

t = fbsm.t
plt.plot(t, x_wc, '-', color='orange', label='state without control')
plt.plot(t, x, 'g-', label='controlled state')
plt.ylabel(r'x(t)')
plt.xlabel(r'$t$')
plt.legend(loc=0)
plt.show()
