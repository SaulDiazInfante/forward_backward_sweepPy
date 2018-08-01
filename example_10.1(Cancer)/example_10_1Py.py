from forward_backward_sweep import ForwardBackwardSweep
import matplotlib.pyplot as plt

a = 1.0
c = 4.0
x_zero = 1.0
m_1 = -1.0
m_2 = 2.0

fbsm = ForwardBackwardSweep()
fbsm.eps = 0.001
fbsm.set_parameters(a, c, x_zero, m_1, m_2)
t = fbsm.t

x_wc = fbsm.runge_kutta_forward(fbsm.u)
[x, lambda_, u] = fbsm.forward_backward_sweep()

plt.figure()
plt.plot(t, x_wc, '-',
         ms=3,
         lw=1,
         alpha=0.7,
         color='green',
         label='State without control'
         )
plt.plot(t, x, '--',
         ms=3,
         lw=1,
         alpha=0.7,
         color='orange',
         label='State controlled')

plt.ylabel(r'$x(t)$')
plt.xlabel(r'Time')
plt.legend(loc=0)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t, x)
plt.ylabel('State')

plt.subplot(2, 1, 2)
plt.plot(t, u)
plt.ylabel('Control')
plt.xlabel('Time')

plt.legend(loc=0)
plt.show()
