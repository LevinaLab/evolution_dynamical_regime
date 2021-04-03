import numpy as np
import matplotlib.pyplot as plt
from os import makedirs, path
from helper_functions.automatic_plot_helper import detect_all_isings
from helper_functions.automatic_plot_helper import load_isings
from helper_functions.isolated_population_helper import fittest_in_isolated_populations
from helper_functions.isolated_population_helper import seperate_isolated_populations
from helper_functions.automatic_plot_helper import load_settings
from matplotlib.lines import Line2D


def main(sim_name, isings_list_dict, attr, colors=['darkorange', 'royalblue', 'maroon'], name_extension=''):

    iter_list = detect_all_isings(sim_name)
    folder = 'save/' + sim_name
    savefolder = folder + '/figs/' + attr + '/' # + '_line/'

    savefilename = savefolder + '{}_gen{}-{}-_{}.png'.format(
        attr, str(iter_list[0]), str(iter_list[-1]), name_extension)


    if not path.exists(savefolder):
        makedirs(savefolder)

    #colors = ['red', 'blue', 'green']
    plt.figure(figsize=(19, 10))
    legend_elements = []
    for i, iso_pop_name in enumerate(isings_list_dict):

        y_axis = []
        for isings in isings_list_dict[iso_pop_name]:
            attr_values_onegen = []
            for I in isings:
                exec('attr_values_onegen.append(I.{})'.format(attr))
            gen_avg = np.mean(attr_values_onegen)
            y_axis.append(gen_avg)

        x_axis = np.arange(len(y_axis))


        legend_elements.append(Line2D([0], [0], marker='o', color='w', label='population {}'.format(iso_pop_name), markerfacecolor=colors[i],
                                      markersize=5, alpha=0.75))
        plt.scatter(x_axis, y_axis, c=colors[i], alpha=0.15)

    if attr == 'norm_food_and_ts_avg_energy':
        plt.ylim(0, 0.0002)
    # plt.legend(loc="lower right", bbox_to_anchor=(0.95, 0.05), handles=legend_elements)
    plt.legend(handles=legend_elements)
    plt.xlabel('Generation')
    plt.ylabel(attr)
    plt.savefig(savefilename, dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    sim_name = 'sim-20201005-115242-g_4000_-t_2000_-rand_seas_-rec_c_1000_-c_props_100_50_-2_2_100_40_-iso_-ref_1000_-c_4_-a_1000_1001_10002_2000_3998_3999_-no_trace_-n_different_betas_rand_seas2_TEST_COPY_2_DYNAMIC_RANGE' #'sim-20200714-190003-g_100_-t_5_-iso_-n_test'
    isings_list = load_isings(sim_name, wait_for_memory=False)
    #isings_list = load_isings_from_list(sim_name, np.arange(100))
    isings_list_dict = seperate_isolated_populations(isings_list)
    isings_list_dict = fittest_in_isolated_populations(isings_list_dict)
    settings = load_settings(sim_name)

    if True:
        for isings in isings_list:
            for I in isings:
                I.norm_avg_energy = I.avg_energy / I.time_steps

        if settings['random_food_seasons']:
            for isings in isings_list:
                for I in isings:
                    I.norm_food_and_ts_avg_energy = I.norm_avg_energy / I.food_in_env



    main(sim_name, isings_list_dict, 'norm_avg_energy')