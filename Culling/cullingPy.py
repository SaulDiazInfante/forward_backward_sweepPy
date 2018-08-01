from forward_backward_sweep import ForwardBackwardSweepQuadratic
from forward_backward_sweep import ForwardBackwardSweepLinear
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
#
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import numpy as np

#
#
nu = 0.6
mu = 0.4
k = 84.0
alpha = 0.05
#
r_zero = 6.5
p = 70
#
#
theta = 2.0
i_zero = 1.0
name_file_1 = 'figure_1_culling.eps'
name_file_2 = 'figure_2_culling.eps'
name_file_3 = 'figure_3_culling.eps'
#
fbsm_quadratic = ForwardBackwardSweepQuadratic()
fbsm_linear = ForwardBackwardSweepLinear()

fbsm_quadratic_1 = ForwardBackwardSweepQuadratic()
fbsm_linear_1 = ForwardBackwardSweepLinear()
#
fbsm_quadratic.set_parameters(nu, mu, k, alpha, theta, r_zero, i_zero, p)
fbsm_linear.set_parameters(nu, mu, k, alpha, 1.0, r_zero, i_zero, p)
#
fbsm_quadratic_1.set_parameters(nu, mu, k, alpha, theta, 3.5, i_zero, 110)
fbsm_linear_1.set_parameters(nu, mu, k, alpha, 1.0, 3.5, i_zero, 110)

t = fbsm_quadratic.t
x_wc = fbsm_quadratic.runge_kutta_forward(fbsm_quadratic.u)
#
#
[x, lambda_, u] = fbsm_quadratic.forward_backward_sweep()
[x_bb, lambda_bb, u_bb] = fbsm_linear.forward_backward_sweep()
#
mpl.style.use('ggplot')
# plt.ion()
ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((2, 2), (0, 1))
ax3 = plt.subplot2grid((2, 2), (1, 1))
#
infected_wc = x_wc[:, 1]
infected = x[:, 1]
infected_bb = x_bb[:, 1]
ax1.plot(t, infected_wc, '-',
         ms=3,
         lw=2,
         alpha=1,
         color='darkgreen',
         label='Not Controlled'
         )
ax1.plot(t, infected, '--',
         ms=3,
         lw=2,
         alpha=1.0,
         color='orange',
         label='Quadratic')
ax1.plot(t, infected_bb, '.',
         ms=3,
         lw=2,
         alpha=1.0,
         color='blue',
         label='Bang Bang')
ax1.set_ylabel(r'Infected population $I(t)$')
ax1.set_xlabel(r'Time (years)')
ax1.set_title(r'$P=70, R_0=6$', fontsize=10)
art = []
lgd = ax1.legend(bbox_to_anchor=(.27, .7, .72, .302), loc=0,
                 ncol=1, mode="expand", borderaxespad=0.,
                 shadow=False, fancybox=True,
                 frameon=False)
art.append(lgd)
#
#
#
ax2.plot(t, u[:, 0], '-',
         ms=3,
         lw=2,
         alpha=1.0,
         color='orange',
         label=r'$u(t)$')
ax2.set_ylabel(r'Quadratic')
ax2.set_title(r'Culling', fontsize=10)
#
ax3.plot(t, u_bb[:, 0], '-',
         ms=3,
         lw=2,
         alpha=1.0,
         color='darkblue',
         label=r'$u(t)$')
ax3.set_ylabel(r' Bang bang')
ax3.set_xlabel(r'Time(years)')
plt.tight_layout()
fig = mpl.pyplot.gcf()
fig.set_size_inches(5.5, 5.5 / 1.618)
fig.savefig(name_file_1)
#
#
# Figure 2
#
#
plt.figure()
x_1_wc = fbsm_quadratic_1.runge_kutta_forward(fbsm_quadratic_1.u)
#
#
[x_1, lambda_, u_1] = fbsm_quadratic_1.forward_backward_sweep()
[x_1_bb, lambda_1_bb, u_1_bb] = fbsm_linear_1.forward_backward_sweep()
#
ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((2, 2), (0, 1))
ax3 = plt.subplot2grid((2, 2), (1, 1))
#
infected_1_wc = x_1_wc[:, 1]
infected_1 = x_1[:, 1]
infected_1_bb = x_1_bb[:, 1]
ax1.plot(t, infected_1_wc, '-',
         ms=3,
         lw=2,
         alpha=1,
         color='darkgreen',
         label='Not\nControlled'
         )
ax1.plot(t, infected_1, '--',
         ms=3,
         lw=2,
         alpha=1.0,
         color='orange',
         label='Quadratic')
ax1.plot(t, infected_1_bb, '.',
         ms=3,
         lw=2,
         alpha=1.0,
         color='blue',
         label='Bang Bang')
ax1.set_ylabel(r'Infected population $I(t)$')
ax1.set_xlabel(r'Time (years)')
ax1.set_title(r'$P=110, R_0=3.5$', fontsize=10)
art = []
lgd = ax1.legend(bbox_to_anchor=(.415, .7, .72, .302), loc=0,
                 ncol=1, mode="expand", borderaxespad=0.,
                 shadow=False, fancybox=True,
                 frameon=False)
art.append(lgd)
#
# ax1.legend(loc=0)
#
ax2.plot(t, u_1[:, 0], '-',
         ms=3,
         lw=2,
         alpha=1.0,
         color='orange',
         label=r'$u(t)$')
ax2.set_ylabel(r'Quadratic')
ax2.set_title(r'Culling', fontsize=10)
#
#
ax3.plot(t, u_1_bb[:, 0], '-',
         ms=3,
         lw=2,
         alpha=1.0,
         color='darkblue',
         label=r'$u(t)$')
ax3.set_ylabel(r' Bang bang')
ax3.set_xlabel(r'Time(years)')
plt.tight_layout()
fig = mpl.pyplot.gcf()
fig.set_size_inches(5.5, 5.5 / 1.618)
fig.savefig(name_file_2)
#
#
# Figure 3
#
#
quadratic_control_cost = fbsm_quadratic.control_cost(x, u)
linear_control_cost = fbsm_linear.control_cost(x_bb, u_bb)
quadratic_control_cost_1 = fbsm_quadratic.control_cost(x_1, u_1)
linear_control_cost_1 = fbsm_linear.control_cost(x_1_bb, u_1_bb)
#
plt.figure()
ax4 = plt.subplot2grid((2, 1), (0, 0))
ax5 = plt.subplot2grid((2, 1), (1, 0))
#
ax4.plot(t, quadratic_control_cost,
         label=r'$\theta = 2$',
         color='orange'
         )
ax4.plot(t, linear_control_cost,
         label=r'Quadratic',
         color='darkblue'
         )
ax4.set_ylabel('Control Cost')
ax4.set_title(r'$p=70, R_0=6$', fontsize=10)

ax5.plot(t, quadratic_control_cost_1,
         label=r'Quadratic',
         color='orange')
ax5.plot(t, linear_control_cost_1,
         label=r'Bang bang',
         color='darkblue'
         )
ax5.set_ylabel('Control Cost')
ax5.set_xlabel('Time (years)')
ax5.set_title(r'$p=110, R_0=3.5$', fontsize=10)
str_legend = 'Final costs:'
str_culling_quadratic_cost = 'Quadratic: '
str_culling_quadratic_cost += "{0:.2f}".format(quadratic_control_cost_1[-1])
str_culling_linear_cost = 'Bang-bang: '
str_culling_linear_cost += "{0:.2f}".format(linear_control_cost_1[-1])
ax5.text(8.0, 220, str_legend, style='oblique')
ax5.text(8.4, 140, str_culling_quadratic_cost, style='italic')
ax5.text(8.25, 60, str_culling_linear_cost, style='italic')
art = []

lgd = plt.legend(bbox_to_anchor=(0.45, 0.6, 1.0, 1.0),
                 loc='center right',
                 # mode='expand',
                 ncol=1, borderaxespad=0.)
art.append(lgd)

plt.tight_layout()
fig = mpl.pyplot.gcf()
fig.set_size_inches(3.5, 5.5 / 1.618)
fig.savefig(name_file_3,
            additional_artists=art,
            bbox_inches="tight")
