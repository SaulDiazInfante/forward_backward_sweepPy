from forward_backward_sweep import ForwardBackwardSweep
from matplotlib import rcParams

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']
params = {
    'figure.titlesize': 10,
    'axes.titlesize':   10,
    'axes.labelsize':   10,
    'font.size':        10,
    'legend.fontsize':  8,
    'xtick.labelsize':  8,
    'ytick.labelsize':  8,
    'text.usetex':      True
}
rcParams.update(params)
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

t_0 = 0.0
t_f = 20.0
b = 0.525
d = 0.5
c = 0.00115
e = 0.5
g = 0.1
a = 0.2
a_w = 2.0
# Initial conditions
s_zero = 1000.0
e_zero = 100.0
i_zero = 50.0
r_zero = 15.0
n_zero = s_zero + e_zero + i_zero + r_zero

fbsm = ForwardBackwardSweep()
fbsm.set_parameters(t_0, t_f, b, d, c, e, g, a, a_w, s_zero, e_zero,
                    i_zero, r_zero, n_zero)

t = fbsm.t
n_max = fbsm.n_max
u = np.zeros(n_max)
x_wc = fbsm.runge_kutta_forward(u)
[x, lambda_, u] = fbsm.forward_backward_sweep()

# plotting
name_file_1 = 'epidemics_lenhart_lab7.eps'
mpl.style.use('ggplot')
# plt.ion()
plt.show()

ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((2, 2), (0, 1))
ax3 = plt.subplot2grid((2, 2), (1, 1))
ax1.plot(t, x_wc[:, 2], '-',
         ms=3,
         lw=2,
         alpha=1,
         color='darkgreen',
         label='Uncontrolled Infection'
         )
ax1.plot(t, x[:, 2], '--',
         ms=3,
         lw=2,
         alpha=1.0,
         color='orange',
         label='Controlled Infection')
ax1.set_ylabel(r'Infected $I(t)$')
ax1.set_xlabel(r'Time (years)')
art = []
lgd = ax1.legend(bbox_to_anchor=(0., -.3, 1., .102), loc=0,
                 ncol=1, mode="expand", borderaxespad=0.)
art.append(lgd)
#
ax2.plot(t, x[:, 3], '-',
         ms=3,
         lw=2,
         alpha=1.0,
         color='orange',
         label='Adjoint')

ax2.set_ylabel(r'Adjoint $\lambda(t)$')
# ax2.set_xlabel(r'Time (years)')

# ax3.subplot(2, 1, 2)
ax3.plot(t, u,
         ms=3,
         lw=2,
         alpha=1.0,
         color='orange',
         label='Control')
ax3.set_ylabel('Vaccination $u(t)$')
ax3.set_xlabel('Time (years)')

# plt.legend(loc=0)
plt.tight_layout()
fig = mpl.pyplot.gcf()
fig.set_size_inches(5.5, 5.5 / 1.618)
fig.savefig(name_file_1)
plt.show()
