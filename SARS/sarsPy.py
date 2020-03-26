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
import matplotlib.patches as mpatches
import matplotlib as mpl
import numpy as np

#
#
beta = 0.2
e_e = 0.3
e_q = 0.0
e_j = 0.1
#
#
mu = 0.000034
#
p = 0.0
k_1 = 0.1
k_2 = 0.125
d_1 = 0.0079
d_2 = 0.0068
#
#
sigma_1 = 0.0337
sigma_2 = 0.0386
#
# initial conditions
s_zero = 12e6
e_zero = 1565.0
q_zero = 292.0
i_zero = 695
j_zero = 326
r_zero = 20
n_whole = s_zero + e_zero + q_zero + i_zero + j_zero + r_zero
# functional cost
b_1 = 1.0
b_2 = 1.0
b_3 = 1.0
b_4 = 1.0
c_1 = 300.0
c_2 = 600.0

name_file_1 = 'figure_1_sars.eps'
name_file_2 = 'figure_2_sars.eps'
name_file_3 = 'figure_3_sars.eps'
#
fbsm = ForwardBackwardSweep()
fbsm.set_parameters(beta, e_e, e_q, e_j,
                    mu, p, k_1, k_2, d_1, d_2, sigma_1, sigma_2,
                    n_whole, b_1, b_2, b_3, b_4, c_1, c_2,
                    s_zero, e_zero, q_zero,
                    i_zero, j_zero, r_zero)
#
t = fbsm.t
x_wcc = fbsm.runge_kutta_forward(fbsm.u)
constant_cost = fbsm.control_cost(x_wcc, fbsm.u)
#
[x, lambda_, u] = fbsm.forward_backward_sweep()
optimal_cost = fbsm.control_cost(x, u)
#
mpl.style.use('ggplot')
# plt.ion()
n_whole = fbsm.n_whole
ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((2, 2), (0, 1))
ax3 = plt.subplot2grid((2, 2), (1, 1))
#
infected_wcc = x_wcc[:, 1] + x_wcc[:, 2] + x_wcc[:, 3] + x_wcc[:, 4]
infected = x[:, 1] + x[:, 2] + x[:, 3] + x[:, 4]
ax1.plot(t, infected_wcc, '-',
         ms=3,
         lw=2,
         alpha=1,
         color='darkgreen',
         label='Infected population with' +
               ' constant control policy'
         )
ax1.plot(t, infected, '--',
         ms=3,
         lw=2,
         alpha=1.0,
         color='orange',
         label='Infected population with\\ optimal control policy')
ax1.set_ylabel(r'$E + Q + I + J$')
ax1.set_xlabel(r'Time (days)')
ax1.set_ylim([0.0, 5000])
art = []
lgd = ax1.legend(bbox_to_anchor=(-0.2, -.3, 1.5, .102), loc=0,
                 ncol=1, mode="expand", borderaxespad=0.)
art.append(lgd)

#
#
#
text = 'time, Quarantine, Isolation'
np.savetxt('sars_optimal_control.dat', np.transpose([t, u[:, 0], u[:, 1]]), header=text, delimiter=',')
ax2.plot(t, u[:, 0], '-',
         ms=3,
         lw=2,
         alpha=1.0,
         color='orange',
         label=r'$u_1(t)$')
ax2.set_ylabel(r'Quarantine')
ax2.set_xlabel(r'Time(days)')
ax3.plot(t, u[:, 1], '-',
         ms=3,
         lw=2,
         alpha=1.0,
         color='orange',
         label=r'$u_2(t)$')
ax3.set_ylabel('Isolation')
ax3.set_xlabel('Time (days)')
#
#
#

plt.tight_layout()
fig = mpl.pyplot.gcf()
fig.set_size_inches(5.5, 5.5 / 1.618)
fig.savefig(name_file_1,
            additional_artists=art,
            bbox_inches="tight")
#
#
# For figure 2 of the article Optimal control of treatments in a two
# strain Tuberculosis Model
#
#
fbsm_lc = ForwardBackwardSweep()
n_max = fbsm_lc.n_max
fbsm_lc.u[:, 0] = fbsm_lc.u_1_lower * np.ones([n_max])
fbsm_lc.u[:, 1] = fbsm_lc.u_2_lower * np.ones([n_max])
x_wlc = fbsm_lc.runge_kutta_forward(fbsm_lc.u)
lc_control = fbsm_lc.control_cost(x_wlc, fbsm_lc.u)

fig = plt.figure()
axes = fig.add_subplot(2, 3, 1)
axes.plot(t, x[:, 0], '-',
          ms=3,
          lw=1,
          alpha=.60,
          color='orange',
          label='Under Optimal Control'
          )
axes.set_ylabel(r'Suceptible')
#
#
#
#
axes = fig.add_subplot(2, 3, 2)
axes.plot(t, x_wcc[:, 0], '-',
          ms=3,
          lw=1,
          alpha=1.0,
          color='darkgreen',
          label='Under Constant Control'
          )
#
#
#
axes = fig.add_subplot(2, 3, 3)
axes.plot(t, x_wlc[:, 0], '-',
          ms=3,
          lw=1,
          alpha=1.0,
          color='darkblue',
          label='Under Lower Bound Control'
          )

#
#
#
axes = fig.add_subplot(2, 3, 4)
l4 = axes.plot(t, x[:, 5], '-',
               ms=3,
               lw=1,
               alpha=.60,
               color='orange',
               label='Under Optimal Control'
               )
axes.set_ylabel(r'Recovered')
axes.set_xlabel(r'Time(days)')
#
axes = fig.add_subplot(2, 3, 5)
l5 = axes.plot(t, x_wcc[:, 5], '-',
               ms=3,
               lw=1,
               alpha=1.0,
               color='darkgreen',
               label='Under Constant Control'
               )
axes.set_xlabel('Time (days)')
#
#
axes = fig.add_subplot(2, 3, 6)
l6 = axes.plot(t, x_wlc[:, 5],
               '-',
               ms=3,
               lw=1,
               alpha=1.0,
               color='darkblue',
               label='Under Lower Bound Control'
               )
axes.set_xlabel('Time (days)')

lines = [l4, l5, l6]
labels = ['Under Optimal Control',
          'Under Constant Control',
          'Under Lower Bound Control']

l4_patch = mpatches.Patch(color='orange',
                          lw=1,
                          label='Optimal Control')
l5_patch = mpatches.Patch(color='darkgreen',
                          lw=1,
                          label='Constant Control')
l6_patch = mpatches.Patch(color='darkblue',
                          lw=1,
                          label='Lower Bound Control')

colors = ['orange', 'darkgreen', 'darkblue']
texts = ['Optimal Control', 'Constant Control', 'Lower Bound Control']
patches = [plt.plot([], [], ls="-", lw=1, color=colors[i],
                    label="{:s}".format(texts[i]))[0]
           for i in range(len(texts))]

plt.tight_layout()
fig = mpl.pyplot.gcf()
art = []
lgd = plt.legend(handles=patches,
                 loc='upper left',
                 bbox_to_anchor=(-2.7, 1.7, 1, 1),
                 ncol=3
                 )
art.append(lgd)
fig.set_size_inches(5.5, 5.5 / 1.618)
fig.savefig(name_file_2,
            additional_artists=art,
            bbox_inches="tight")

# plt.show()
#
#
# Figure 3 Control Cost
#
#
fig = plt.figure()
axes = fig.add_subplot(1, 2, 1)
axes.plot(t, optimal_cost,
          color='orange',
          label='Optimal Control')
axes.plot(t, constant_cost,
          color='darkgreen',
          label='Constant Control')
axes.set_xlabel(r'Time (days)')
axes.set_ylabel(r'Cost of controls')

axes = fig.add_subplot(1, 2, 2)
axes.plot(t, lc_control,
          color='darkblue',
          label='Lower Bound Control')
axes.set_xlabel(r'Time (days)')

colors = ['orange', 'darkgreen', 'darkblue']
texts = ['Optimal Control', 'Constant Control', 'Lower Bound Control']
patches = [plt.plot([], [], ls="-", lw=1, color=colors[i],
                    label="{:s}".format(texts[i]))[0]
           for i in range(len(texts))]

plt.tight_layout()
fig = mpl.pyplot.gcf()
art = []
lgd = plt.legend(handles=patches,
                 loc='upper left',
                 bbox_to_anchor=(-1.17, 0.22, 2.2, 1),
                 mode='expand',
                 ncol=2
                 )
art.append(lgd)
fig.set_size_inches(5.5, 5.5 / 1.618)
fig.savefig(name_file_3,
            additional_artists=art,
            bbox_inches="tight")
