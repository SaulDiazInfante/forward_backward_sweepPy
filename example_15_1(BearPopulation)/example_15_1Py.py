from forward_backward_sweep import ForwardBackwardSweep
import matplotlib.pyplot as plt
import matplotlib as mpl

r = 0.1
k = 0.75
m_p = 0.1
m_f = 0.1
c_p = 10000.0
c_f = 1.0
# initial conditions
p_zero = 0.7
f_zero = 0.7
o_zero = 0.25

fbsm = ForwardBackwardSweep()
fbsm.set_parameters(r, k, m_p, m_f, c_p, c_f, p_zero, f_zero, o_zero)
t = fbsm.t

x_wc = fbsm.runge_kutta_forward(fbsm.u)
[x, lambda_, u] = fbsm.forward_backward_sweep()
mpl.style.use('ggplot')
# plt.ion()
plt.show()

ax1 = plt.subplot2grid((2, 3), (0, 0))
ax2 = plt.subplot2grid((2, 3), (0, 1))
ax3 = plt.subplot2grid((2, 3), (0, 2))
ax4 = plt.subplot2grid((2, 3), (1, 0))
ax5 = plt.subplot2grid((2, 3), (1, 1))

ax1.plot(t, x_wc[:, 0], '-',
         ms=3,
         lw=2,
         alpha=1,
         color='darkgreen',
         label='Uncontrolled Park Bear'
         )
ax1.plot(t, x[:, 0], '--',
         ms=3,
         lw=2,
         alpha=1.0,
         color='orange',
         label='Cotrolled T cells')
ax1.set_ylabel(r'Park Density')
ax1.set_xlabel(r'Time (years)')
# ax1.legend(loc=0)
#
ax2.plot(t, x_wc[:, 1], '-',
         ms=3,
         lw=2,
         alpha=1,
         color='darkgreen',
         label='Uncontrolled Park Bear'
         )

ax2.plot(t, x[:, 1], '--',
         ms=3,
         lw=2,
         alpha=1.0,
         color='orange',
         label=r'Forest Density')
ax2.set_ylabel(r'Forest Density')
ax2.set_xlabel(r'Time(years)')
#
ax3.plot(t, x_wc[:, 2], '-',
         ms=3,
         lw=2,
         alpha=1,
         color='darkgreen',
         label='Uncontrolled Park Bear'
         )
ax3.plot(t, x[:, 2], '--',
         ms=3,
         lw=2,
         alpha=1.0,
         color='orange',
         label=r'Controlled $Virus')
ax3.set_ylabel(r' $Outside Density$')
ax3.set_xlabel(r'Time(years)')
#
#
ax4.semilogy(t, u[:, 0], '--',
             ms=3,
             lw=2,
             alpha=1.0,
             color='orange',
             label='Control')
ax4.set_ylabel('Park Harvesting  $u_f(t)$')
ax4.set_xlabel('Time (days)')
#
ax5.plot(t, u[:, 1], '--',
         ms=3,
         lw=2,
         alpha=1.0,
         color='orange',
         label='Control')
ax4.set_ylabel('Park Harvesting  $u_f(t)$')
ax4.set_xlabel('Time (days)')
plt.tight_layout()
plt.show()
