import numpy as np
import matplotlib.pyplot as plt
from os import makedirs, path
from helper_functions.automatic_plot_helper import detect_all_isings
from helper_functions.automatic_plot_helper import load_isings_from_list
from matplotlib.lines import Line2D


def main(sim_name, isings_list, attr_tuple, colors=['darkorange', 'royalblue', 'maroon'],
         attr_c ='isolated_population', name_extension=''):

    attr_x, attr_y = attr_tuple
    attr_tuple = (attr_x, attr_y, attr_c)
    iter_list = detect_all_isings(sim_name)
    folder = 'save/' + sim_name
    savefolder = folder + '/figs/' + attr_x + '_vs_' + attr_y + '/' # + '_line/'

    savefilename = savefolder + '{}_vs_{}_color_{}_gen{}-{}-_{}.png'.format(
        attr_x, attr_y, attr_c, str(iter_list[0]), str(iter_list[-1]), name_extension)


    if not path.exists(savefolder):
        makedirs(savefolder)

    # All individuals = generations * num_individuals:
    num_inds_in_run = len(isings_list) * len(isings_list[0])
    # x_value, y_value, colour_value
    all_inds_plot_arr = np.zeros((num_inds_in_run, 3))

    tot_ind_num = 0
    for isings in isings_list:
        for I in isings:
            for i, attr in enumerate(attr_tuple):
                exec('all_inds_plot_arr[tot_ind_num, i] = I.{}'.format(attr))
            tot_ind_num += 1


    plt.figure(figsize=(19, 10))
    legend_elements = []


    plt.figure(figsize=(19, 10))

    plot_colors = list(map(generate_colors_from_var_c, all_inds_plot_arr[:, 2]))
    plt.scatter(all_inds_plot_arr[:, 0], all_inds_plot_arr[:, 1], c=plot_colors, s=0.8, alpha=0.15)

    # all_plotting_elemnts = [generation


    legend_elements = []
    for i in range(2):
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label='population {}'.format(i),
                                      markerfacecolor=colors[i], markersize=5, alpha=0.75))

    # plt.legend(loc="lower right", bbox_to_anchor=(0.95, 0.05), handles=legend_elements)
    if attr_y == 'Beta':
        plt.yscale('log')
    elif attr_y == 'norm_food_and_ts_avg_energy':
        plt.ylim(0, 0.0002)
    plt.legend(handles=legend_elements)
    plt.xlabel(attr_x)
    plt.ylabel(attr_y)
    plt.savefig(savefilename, dpi=300, bbox_inches='tight')
    print('Saving figure: {}'.format(savefilename))
    plt.show()

def generate_colors_from_var_c(var_c):
    colors = ['darkorange', 'royalblue', 'maroon']
    return(colors[int(var_c)])

if __name__ == '__main__':
    sim_name = 'sim-20200714-210215-g_6000_-rand_ts_-iso_-ref_500_-rec_c_250_-a_100_250_500_1000_-no_trace_-n_different_betas_from_scratch_isolated' #'sim-20200714-190003-g_100_-t_5_-iso_-n_test'
    # isings_list = load_isings(sim_name, wait_for_memory=False)
    isings_list = load_isings_from_list(sim_name, np.arange(10))
    attr_triple = ('generation', 'Beta')
    main(sim_name, isings_list, attr_triple)