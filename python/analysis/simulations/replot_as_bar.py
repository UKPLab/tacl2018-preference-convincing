'''
Generate a bar chart with error bars -- might be easier to read than a line graph with error bars.
'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_result(idx, filename, xlabel, ylabel, linelabel, fig=None):

    if fig is None:
        fig = plt.figure(figsize=(5, 4))
    else:
        plt.figure(fig.number)

    # ax1.semilogx(inverse_scales
    # plt.plot(mean_results[:, 0], mean_results[:, idx],
    #          marker=markers[lineidx], label=linelabel, linewidth=2, markersize=8, linestyle=linestyles[lineidx])
    # plt.errorbar(mean_results[:, 0], mean_results[:, idx], std_results[:, idx],
    #             marker=markers[lineidx], label=linelabel, linewidth=2, markersize=8, linestyle=linestyles[lineidx])

    plt.bar(mean_results[:, 0]+(50*idx), mean_results[:, idx+1], yerr=std_results[:, idx+1], label=linelabel, width=50)

    plt.ylabel(ylabel)
    # plt.xlabel('inverse function scale, s')
    plt.xlabel(xlabel)
    plt.grid('on', axis='y')

    plt.legend(loc='best')

    plt.tight_layout()
    plt.savefig(figure_save_path + '/%s.pdf' % filename)

    return fig

if __name__ == '__main__':

    figure_save_path = './results/synth2/'
    filename = 'r_pairs_bar.pdf'

    fig = None

    cs = [1,3,10,20]

    markers = ['o', 'x', '+', '>', '<', '*']
    linestyles = [':', '-.', '--', '-']

    for idx, c in enumerate(cs):

        mean_results = np.genfromtxt('./results/synth_latent_mean_results_%i.csv' % c)
        std_results = np.genfromtxt('./results/synth_latent_std_results_%i.csv' % c)

        fig = plot_result(idx, filename, 'noise rate in pairwise training labels', '$\\tau$ (on test set)', 'C=%i' % c,
                          fig)