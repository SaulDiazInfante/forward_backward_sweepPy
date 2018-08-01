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

#
#
beta_1 = 13.0
beta_2 = 13.0
beta_3 = 0.0436
#
mu = 0.0143
d_1 = 0.0
d_2 = 0.0
k_1 = 0.5
k_2 = 1.0
r_1 = 2.0
r_2 = 1.0
#
p = 0.4
q = 0.1
n_whole = 30000.0
b_1 = 50.0
b_2 = 500.0
s_zero = 76.0 / 120.0
l_zero = 36.0 / 120.0
i_zero = 4.0 / 120.0
l_r_zero = 2.0 / 120.0
i_r_zero = 1.0 / 120.0
r_zero = 1.0 / 120.0
lambda_recruitment = n_whole * mu
name_file_1 = 'figure_1_two_strain_tbm.eps'
name_file_2 = 'figure_2_two_strain_tbm.eps'
name_file_3 = 'figure_3_two_strain_tbm.eps'

#
fbsm = ForwardBackwardSweep()
fbsm.set_parameters(beta_1, beta_2, beta_3,
                    mu, d_1, d_2, k_1, k_2, r_1, r_2, p, q,
                    n_whole, lambda_recruitment, b_1, b_2,
                    s_zero, l_zero, i_zero,
                    l_r_zero, i_r_zero, r_zero)

t = fbsm.t
x_wc = fbsm.runge_kutta_forward(fbsm.u)
[x, lambda_, u] = fbsm.forward_backward_sweep()
mpl.style.use('ggplot')
# plt.ion()
n_whole = fbsm.n_whole
ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((2, 2), (0, 1))
ax3 = plt.subplot2grid((2, 2), (1, 1))
#
ax1.plot(t, (x_wc[:, 3] + x_wc[:, 4]) / n_whole, '-',
         ms=3,
         lw=2,
         alpha=1,
         color='darkgreen',
         label='Without control MR-TB'
         )
ax1.plot(t, (x[:, 3] + x[:, 4]) / n_whole, '--',
         ms=3,
         lw=2,
         alpha=1.0,
         color='orange',
         label='Controlled MR-TB')
ax1.set_ylabel(r'$(L_2 + I_2)/N$')
ax1.set_xlabel(r'Time (years)')
ax1.set_ylim([0.02, 0.16])
art = []
lgd = ax1.legend(bbox_to_anchor=(0., -.3, 1., .102), loc=0,
                 ncol=1, mode="expand", borderaxespad=0.)
art.append(lgd)
#
# ax1.legend(loc=0)
#
ax2.plot(t, u[:, 0], '-',
         ms=3,
         lw=2,
         alpha=1.0,
         color='orange',
         label=r'$u_1(t)$')
ax2.set_ylabel(r'Case Finding $u_1(t)$')
ax2.set_xlabel(r'Time(years)')
ax3.plot(t, u[:, 1], '-',
         ms=3,
         lw=2,
         alpha=1.0,
         color='orange',
         label=r'$u_2(t)$')
ax3.set_ylabel('Case Holding $u_2(t)$')
ax3.set_xlabel('Time (years)')
ax3.set_ylim([0.0, 1.0])
#
#
plt.tight_layout()
fig = mpl.pyplot.gcf()
fig.set_size_inches(5.5, 5.5 / 1.618)
fig.savefig(name_file_1, additional_artists=art,
            bbox_inches="tight")
#
# For figure 2 of the article Optimal control of treatments in a two
# strain Tuberculosis Model
#
plt.figure()
ax1 = plt.subplot2grid((2, 1), (0, 0))
ax2 = plt.subplot2grid((2, 1), (1, 0))
beta_3_values = np.array([0.0131, 0.0217, 0.0290, 0.0436])
label_beta = str(r'$\beta_3=$')
fbsm_2 = ForwardBackwardSweep()
for beta_3 in beta_3_values:
    fbsm_2.set_parameters(beta_1, beta_2, beta_3,
                          mu, d_1, d_2, k_1, k_2, r_1, r_2, p, q,
                          n_whole, lambda_recruitment, b_1, b_2,
                          s_zero, l_zero, i_zero,
                          l_r_zero, i_r_zero, r_zero)
    [x, lambda_, u] = fbsm_2.forward_backward_sweep()
    ax1.plot(t, u[:, 0], '-',
             ms=3,
             lw=1,
             alpha=.60,
             # color='orange',
             label=label_beta + str(beta_3))
    ax1.set_ylabel(r'$u_1(t)$, $u_2(t)$')
    ax1.set_xlabel(r'Time(years)')
    ax1.plot(t, u[:, 1], '-',
             ms=3,
             lw=1,
             alpha=1.0
             # color='orange',
             # label=r'$u_2(t)$'
             )
    #
    ax2.plot(t, u[:, 1], '-',
             ms=3,
             lw=1,
             alpha=1.0,
             # color='orange',
             label=label_beta + str(beta_3)
             )
    ax2.set_ylabel('Case Holding')
    ax2.set_xlabel('Time (years)')
    ax2.set_xlim([4.0, 4.6])
    ax2.set_ylim([0.8, 1.0])
art = []
lgd = plt.legend(bbox_to_anchor=(-0.1, -.65, 1.1, .102), loc=3,
                 ncol=4, mode="expand", borderaxespad=0.)
art.append(lgd)
plt.tight_layout()
plt.tight_layout()
fig = mpl.pyplot.gcf()
fig.set_size_inches(5.5, 5.5 / 1.618)
fig.savefig(name_file_2, additional_artists=art,
            bbox_inches="tight")
# plt.show()
#
# For figure 3 of the article Optimal control of treatments in a two
# strain Tuberculosis Model
#

mpl.style.use('fast')
n_whole_values = np.array([6000, 12000, 30000])
# n_whole_values = np.array([12000])
beta_3 = 0.029
ax1 = plt.subplot2grid((2, 1), (0, 0))
ax2 = plt.subplot2grid((2, 1), (1, 0))
fbsm_3 = ForwardBackwardSweep()
label_n = 'N = '
for n_whole in n_whole_values:
    fbsm_3.set_parameters(beta_1, beta_2, beta_3,
                          mu, d_1, d_2, k_1, k_2, r_1, r_2, p, q,
                          n_whole, lambda_recruitment, b_1, b_2,
                          s_zero, l_zero, i_zero,
                          l_r_zero, i_r_zero, r_zero)
    [x, lambda_, u] = fbsm_3.forward_backward_sweep()
    ax1.plot(t, u[:, 0], '-',
             ms=3,
             lw=1,
             alpha=1.0,
             # color='orange',
             label=label_n + str(n_whole)
             )
    ax1.set_ylabel(r'Case Finding')
    ax1.set_xlabel(r'Time(years)')
    ax2.plot(t, u[:, 1], '-',
             ms=3,
             lw=1,
             alpha=1.0,
             # color='orange',
             label=label_n + str(n_whole))
    ax2.set_ylabel('Case Holding')
    ax2.set_xlabel('Time (years)')
    ax2.set_ylim([0.0, 1.0])

art = []
lgd = plt.legend(bbox_to_anchor=(0., -.73, 1., .102), loc=3,
                 ncol=3, mode="expand", borderaxespad=0.)
art.append(lgd)

plt.tight_layout()
fig = mpl.pyplot.gcf()
fig.set_size_inches(5.5, 5.5 / 1.618)
fig.savefig(name_file_3, additional_artists=art,
            bbox_inches="tight")
