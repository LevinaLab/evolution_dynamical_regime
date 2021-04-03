import os
import numpy as np
from automatic_plot_helper import load_isings_specific_path
from automatic_plot_helper import attribute_from_isings
from automatic_plot_helper import all_folders_in_dir_with
from automatic_plot_helper import load_settings
import copy
import pandas as pd
import glob
# import _pickle as pickle
import pickle
from run_combi import RunCombi
import matplotlib.pylab as plt
from matplotlib.lines import Line2D
import seaborn as sns
import re
from isolated_population_helper import seperate_isolated_populations
from automatic_plot_helper import all_sim_names_in_parallel_folder
from automatic_plot_helper import choose_copied_isings
from automatic_plot_helper import calc_normalized_fitness
from automatic_plot_helper import load_isings_specific_path_decompress
import time

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


class ResponseCurveSimData:
    def __init__(self, sim_name, folder_name, key, folder_num_in_key,  attrs_list_each_food_num, food_num_list,
                 dynamic_range_folder_includes, dynamic_range_folder_includes_index):
        self.sim_name = sim_name
        self.sim_num = sim_name[sim_name.rfind('Run_')+4:]

        self.folder_name = folder_name
        self.folder_num_in_key = folder_num_in_key
        # Key defines dynamical regime (critical, sub-critical,...)
        self.key = key
        self.attrs_list_each_food_num = attrs_list_each_food_num
        self.food_num_list = food_num_list
        # calculate averages
        self.avg_attr_list = [np.mean(attrs) for attrs in attrs_list_each_food_num]
        self.dynamic_range_folder_includes = dynamic_range_folder_includes
        self.dynamic_range_folder_includes_index = dynamic_range_folder_includes_index

        self.highlight_this_sim = False
        self.label = self.sim_num

        self.legend_label = '{}_{}_{}'.format(self.key, self.folder_name, self.dynamic_range_folder_includes)

    def highlight_certain_sims(self, plot_settings):
        '''
        This function changes attributes, such that the simulation specified in plot_settings['label_highlighted_sims']
        are highlighted and relabeled
        '''
        self.highlight_this_sim = False
        self.label = None

        folder_name_label_dict = plot_settings['label_highlighted_sims']
        for folder_name in folder_name_label_dict:
            if folder_name == self.folder_name:
                include_name_label_dict = folder_name_label_dict[folder_name]
                for include_name in include_name_label_dict:
                    if include_name == self.dynamic_range_folder_includes:
                        sim_num_label_dict = include_name_label_dict[include_name]
                        for sim_num in sim_num_label_dict:
                            if type(sim_num) is int:
                                sim_num_compare = str(sim_num)
                            else:
                                sim_num_compare = sim_num

                            if sim_num_compare == self.sim_num:
                                self.label = sim_num_label_dict[sim_num]
                                self.highlight_this_sim = True

    def make_old_class_compatible_with_current_version(self):
        '''
        This function makes previously saved plotting data compatible with the current version of this script
        Can be left away in future...
        '''
        self.sim_num = self.sim_name[self.sim_name.rfind('Run_')+4:]
        self.highlight_this_sim = False
        self.label = self.sim_num

    def create_custom_legend_labels(self, plot_settings):
        custom_legend_labels = plot_settings['custom_legend_labels']
        for folder_name in custom_legend_labels:
            if folder_name == self.folder_name:
                include_name_label_dict = custom_legend_labels[folder_name]
                for include_name in include_name_label_dict:
                    if include_name == self.dynamic_range_folder_includes:
                        self.legend_label = include_name_label_dict[include_name]


def dynamic_range_main(folder_name_dict, plot_settings):

    if not plot_settings['only_plot']:
        plot_settings['savefolder_name'] = 'response_plot_{}_{}' \
            .format(time.strftime("%Y%m%d-%H%M%S"), plot_settings['varying_parameter'])
        os.makedirs('save/{}'.format(plot_settings['savefolder_name']))
        sim_data_list_each_folder = prepare_data(folder_name_dict, plot_settings)
        save_plot_data(sim_data_list_each_folder, plot_settings)
    else:
        sim_data_list_each_folder = load_plot_data(plot_settings['only_plot_folder_name'])
        plot_settings['savefolder_name'] = plot_settings['only_plot_folder_name']

    settings_folder = 'save/{}/settings/'.format(plot_settings['savefolder_name'])
    save_settings(settings_folder, plot_settings)
    plot_axis(sim_data_list_each_folder, plot_settings)


def prepare_data(folder_name_dict, plot_settings):

    sim_data_list_each_folder = []
    # All folder list dicts (sub critical or critical?)
    for key in folder_name_dict:
        folder_name_includes_dict = folder_name_dict[key]
        # Iteration through all folder names
        for folder_num_in_key, folder_name in enumerate(folder_name_includes_dict):
            dynamic_range_folder_includes_list = folder_name_includes_dict[folder_name]
            # Iterationg through all "dynamic_range_folder_includes", so basically through each specified run of the dynamic_range_pipeline
            for dynamic_range_folder_includes_index, dynamic_range_folder_includes in enumerate(dynamic_range_folder_includes_list):
                sim_names = all_sim_names_in_parallel_folder(folder_name)
                attrs_food_num_lists_each_sim = []
                # Iterating through each simulation
                for sim_name in sim_names:
                    attrs_list_each_food_num_all, food_num_list = load_data(sim_name, folder_name,
                                                                            dynamic_range_folder_includes, plot_settings)
                    sim_data = ResponseCurveSimData(sim_name, folder_name, key, folder_num_in_key,
                                                    attrs_list_each_food_num_all, food_num_list,
                                                    dynamic_range_folder_includes, dynamic_range_folder_includes_index)

                    attrs_food_num_lists_each_sim.append(sim_data)
                sim_data_list_each_folder.append(attrs_food_num_lists_each_sim)

    return sim_data_list_each_folder


def save_settings(folder, settings):
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(folder + 'plot_settings.csv', 'w') as f:
        for key in settings.keys():
            f.write("%s,%s\n" % (key, settings[key]))
    pickle_out = open('{}plot_settings.pickle'.format(folder), 'wb')
    pickle.dump(settings, pickle_out)
    pickle_out.close()


def save_plot_data(plot_data, plot_settings):
    save_dir = 'save/{}/plot_data/'.format(plot_settings['savefolder_name'])
    save_name = 'plot_data.pickle'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    pickle_out = open(save_dir + save_name, 'wb')
    pickle.dump(plot_data, pickle_out)
    pickle_out.close()


def load_plot_data(folder_name):
    save_dir = 'save/{}/plot_data/'.format(folder_name)
    save_name = 'plot_data.pickle'
    print('Load plot data from: {}{}'.format(save_dir, save_name))
    file = open(save_dir+save_name, 'rb')
    plot_data = pickle.load(file)
    file.close()
    return plot_data


def plot_axis(sim_data_list_each_folder, plot_settings):
    font = {'family': 'serif', 'size': 14, 'serif': ['computer modern roman']}
    plt.rc('font', **font)
    plt.rc('legend', **{'fontsize': 10})

    # plt.rcParams.update({'font.size': 22})
    plt.rc('text', usetex=True)

    plt.figure(figsize=(10, 7))
    ax_main = plt.subplot(111)

    # Make main plot
    plt.axvline(2000, linestyle='dashed', color='firebrick', alpha=0.8, linewidth=1)
    plt.text(2200, 250, 'Trained on \n 2000 time steps', color='firebrick', alpha=0.8)

    plot_data(sim_data_list_each_folder, plot_settings, label_each_sim=True)
    plt.ylim(-10, 1300)
    plt.legend()
    plt.ylabel(r'$\langle \langle \langle E_\mathrm{org} \rangle \rangle \rangle$')
    # plt.xlabel('Percentage of food that population was originally trained on')
    if plot_settings['varying_parameter'] == 'time_steps':
        plt.xlabel('Number of time steps')
    elif plot_settings['varying_parameter'] == 'food':
        plt.xlabel('Number of foods')



    # PLot zoomed-in inset


    ax_zoom1 = inset_axes(ax_main, 3.4, 3.4, loc='upper left')
    # plt.axvline(2000, linestyle='dashed', color='firebrick', alpha=0.3, linewidth=1)
    plt.vlines(2000, 42, 70, linestyles='dashed', colors='firebrick', alpha=0.8, linewidth=1)
    plt.vlines(2000, 0, 4, linestyles='dashed', colors='firebrick', alpha=0.8, linewidth=1)

    plot_data(sim_data_list_each_folder, plot_settings, label_each_sim=False)

    ax_zoom1.set_xlim(1500, 20000)
    ax_zoom1.set_ylim(0, 70)

    # ax_zoom1.set_ylim(1.94, 1.98)
    # ax_zoom1.set_xlim(0, 1)

    plt.yticks(visible=False)
    plt.xticks(visible=False)
    mark_inset(ax_main, ax_zoom1, loc1=3, loc2=4, fc='none', ec='0.5')



    save_name = 'response_plot.png'
    save_folder = 'save/{}/figs/'.format(plot_settings['savefolder_name'])
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    plt.savefig(save_folder+save_name, bbox_inches='tight', dpi=300)


def plot_data(sim_data_list_each_folder, plot_settings, label_each_sim=True):
    # Iterating through each folder
    for sim_data_list in sim_data_list_each_folder:
        list_of_avg_attr_list = []
        list_of_food_num_list = []
        for sim_data in sim_data_list:
            list_of_avg_attr_list.append(sim_data.avg_attr_list)
            list_of_food_num_list.append(sim_data.food_num_list)

            sim_data.make_old_class_compatible_with_current_version()
            if plot_settings['highlight_certain_sims']:
                sim_data.highlight_certain_sims(plot_settings)
            if plot_settings['customize_legend_labels']:
                sim_data.create_custom_legend_labels(plot_settings)

        # for food_num_list in list_of_food_num_list:
        #     if not food_num_list == list_of_food_num_list[0]:
        #         raise Exception('There seem to be files for different food numbers within the simulations of folder {}'
        #                         .format(sim_data.folder_name))



        # food_num_list is not ordered yet, order both lists acc to food_num list for line plotting
        list_of_food_num_list, list_of_avg_attr_list = sort_lists_of_lists(list_of_food_num_list, list_of_avg_attr_list)

        avg_of_avg_attr_list = []
        # This goes through all lists and takes averages of the inner nesting, such that instead of a list of lists
        # we have one list with average value of each entriy of the previous lists,
        # in future do this with np. array and define axis to take average over
        for i in range(len(list_of_avg_attr_list[0])):
            avg_of_avg_attr_list.append(np.mean([list_of_avg_attr_list[j][i] for j in range(len(list_of_avg_attr_list))]))

        marker = plot_settings['marker'][sim_data.folder_num_in_key]
        colors = plot_settings['color'][sim_data.key]
        try:
            color = colors[sim_data.dynamic_range_folder_includes_index]
        except IndexError:
            raise IndexError('Color list is out of bounds check whether dynamic_range_folder_includes_list is longer'
                             ' than color lists in color dict')
        # Plot each simulation
        plt.scatter(list_of_food_num_list, list_of_avg_attr_list, marker=marker, c=color, s=3, alpha=0.2)
        # Connect each simulation datapoint with lines
        for food_num_list, avg_attr_list, sim_data in zip(list_of_food_num_list, list_of_avg_attr_list, sim_data_list):
            if sim_data.highlight_this_sim:
                plt.plot(food_num_list, avg_attr_list, c=color, alpha=0.5, linewidth=1)
            else:
                plt.plot(food_num_list, avg_attr_list, c=color, alpha=0.2, linewidth=0.3)

        # Plot averages of each folder
        if plot_settings['plot_means']:
            plt.scatter(list_of_food_num_list[0], avg_of_avg_attr_list, marker=marker, c=color, s=10, alpha=1,
                        label=sim_data_list[0].legend_label)
        else:
            # If switched off just plot empty list for the legend labels
            plt.scatter([], [], marker=marker, c=color, s=10, alpha=1,
                        label=sim_data_list[0].legend_label)

        # Label each simulation:
        if label_each_sim:
            for sim_data, food_num_list, avg_attr_list in zip(sim_data_list, list_of_food_num_list, list_of_avg_attr_list):

                x_offset = 200
                y_offset = 0
                coordinates = (food_num_list[-1]+x_offset, avg_attr_list[-1]+y_offset)

                label = sim_data.label #sim_data.sim_name[sim_data.sim_name.rfind('Run_')+4:]  # TODO check whether this is run number!
                if label is not None:
                    plt.text(coordinates[0], coordinates[1], label, fontsize=3, c=color)


def sort_lists_of_lists(listof_lists_that_defines_order, second_listof_lists):
    '''
    Input is a list of lists. The inner lists of the list of lists is sorted
    '''
    ordered_order_list = []
    ordered_second_list = []
    for order_list, second_list in zip(listof_lists_that_defines_order, second_listof_lists):
        order_list = np.array(order_list)
        second_list = np.array(second_list)
        order = np.argsort(order_list)
        ordered_order_list.append(list(order_list[order]))
        ordered_second_list.append(list(second_list[order]))
    return ordered_order_list, ordered_second_list


def load_data(sim_name, folder_name, dynamic_range_folder_includes, plot_settings):
    sim_dir = 'save/{}'.format(sim_name)

    attrs_list_each_food_num_all = []
    attrs_list_each_food_num_critical = []
    attrs_list_each_food_num_sub_critical = []
    food_num_list = []
    dir_list = all_folders_in_dir_with('{}/repeated_generations'.format(sim_dir), dynamic_range_folder_includes)
    for dir in dir_list:
        if plot_settings['compress_save_isings']:
            isings_list = load_isings_specific_path_decompress(dir)
        else:
            isings_list = load_isings_specific_path(dir)
        if plot_settings['only_copied']:
            isings_list = [choose_copied_isings(isings) for isings in isings_list]
        if plot_settings['attr'] == 'norm_avg_energy' or plot_settings['attr'] == 'norm_food_and_ts_avg_energy':
            settings = load_settings(sim_name)
            calc_normalized_fitness(isings_list, plot_settings, settings)
        # MERGING INDIVIDUALS OF ALL REPEATS WITH SIMILAR SETTINGS INTO ONE LIST:
        isings = make_2d_list_1d(isings_list)
        # isings_populations_seperated = seperate_isolated_populations([isings])
        # isings_critical = isings_populations_seperated[0][0]
        # isings_sub_critical = isings_populations_seperated[1][0]
        attrs_list_each_food_num_all.append(attribute_from_isings(isings, plot_settings['attr']))
        # attrs_list_each_food_num_critical.append(attribute_from_isings(isings_critical, attr))
        # attrs_list_each_food_num_sub_critical.append(attribute_from_isings(isings_sub_critical, attr))
        food_num_list.append(get_int_end_of_str(dir))
    return attrs_list_each_food_num_all, food_num_list


def get_int_end_of_str(s):
    m = re.search(r'\d+$', s)
    return int(m.group()) if m else None


def make_2d_list_1d(in_list):
    out_list = []
    for sub_list in in_list:
        for en in sub_list:
            out_list.append(en)
    return out_list

def find_number_after_char_in_str(str, char):

    match = re.search('uniprotkb:P(\d+)', str)
    if match:
        return match.group(1)

if __name__ == '__main__':
    # In these dicts all folders, with parallel runs, that shall be loaded must be specified as keys.
    # The entry of each key is a list of all "dynamic_range_folder_includes", which is a string for each run of the
    # dynamic_range_parallel_pipline. This string is a characteristic substring of the folder name of the runs that
    # shall be loaded in the dynamic range folder of each simulation
    #
    # folder_name_dict has the form
    # {-simulation_name1-:[-included_substr1-, -included_substr2-, ...], -simulation_name1-:[-included_substr1-, -included_substr2-, ...]}
    # critical_folder_name_dict = {'sim-20201022-190553_parallel_b1_normal_seas_g4000_t2000':
    #                                  ['gen100_100foods_energies_saved_compressed_try_2', 'gen1000_100foods_energies_saved_compressed_try_2']}
    # sub_critical_folder_name_dict = {'sim-20201022-190615_parallel_b10_normal_seas_g4000_t2000':
    #                                      ['gen1000_100foods_energies_saved_compressed_try_2']}
    # critical_folder_name_dict = {'sim-20201119-190135_parallel_b1_normal_run_g4000_t2000_27_sims': ['ds_res_10_try_2_gen_100d', 'gen4000_100foods_res_10_try_2dy']}
    # sub_critical_folder_name_dict = {'sim-20201119-190204_parallel_b10_normal_run_g4000_t2000_54_sims': ['gen4000_100foods_res_10_try_2dy']}
    # critical_folder_name_dict = {'sim-20201116-182731_parallel_b10_1000ts_fixed_compressed': ['period_overfitting_compressed']}
    critical_folder_name_dict = {'sim-20201119-190135_parallel_b1_normal_run_g4000_t2000_27_sims': ['food_experiment_gen_100_res_10', 'food_experiment_res_10']}
    sub_critical_folder_name_dict = {'sim-20201119-190204_parallel_b10_normal_run_g4000_t2000_54_sims': ['food_experiment_res_10']}

    # critical_folder_name_dict = {'sim-20201022-184145_parallel_TEST_repeated': ['res_10_try_2']}
    # sub_critical_folder_name_dict = {'sim-20201022-184145_parallel_TEST_repeated': ['gen50_100foods_COMPRESSdynamic']}
    plot_settings = {}
    plot_settings['varying_parameter'] = 'time_steps'  # 'time_steps' or 'food'
    plot_settings['only_plot'] = False

    plot_settings['only_plot_folder_name'] = 'response_plot_20201123-225136_time_steps_2000ts_fixed_CritGen100_3999_SubCritGen3999_HUGE_RUN'
    plot_settings['add_save_name'] = ''
    plot_settings['only_copied'] = True
    plot_settings['attr'] = 'avg_energy'
    # Colors for each dynamical regime. The color lists of each dynamical regime are chosen by the index of the
    # currently plotted entry of dynamic_range_folder_includes_list
    plot_settings['color'] = {'critical': ['darkorange', 'olive', 'turquoise'],
                              'sub_critical': ['royalblue', 'pink', 'magenta'],
                              'super_critical': ['maroon', 'red', 'steelblue']}
    # This setting defines the markers, which are used in the order that the folder names are listed
    plot_settings['marker'] = ['.', 'x', '+']
    # This feature looks for compressed ising-files and decompresses them
    plot_settings['compress_save_isings'] = True
    # This plots the means of all simulations in one folder for one value of the y-axis
    plot_settings['plot_means'] = False
    plot_settings['critical_folder_name_dict'] = critical_folder_name_dict
    plot_settings['sub_critical_folder_name_dict'] = sub_critical_folder_name_dict

    #  This feature highlights certain simulation runs and relabels them. Those simulations, that shall be highlighted
    #  and relabeled have to be specified in plot_settings['label_highlighted_sims']. All other simulations are not
    #  labeled
    #  plot_settings['label_highlighted_sims'] is a dict of dicts of dicts with the following shape:
    #  {folder_name_1: {include_name_1: {simulation_number: new_label_1}, ...}, ...}
    #  The include name ("dynamic_range_folder_includes") has to be equal to the one used in the folder_name_dict s.
    plot_settings['highlight_certain_sims'] = True

    # plot_settings['label_highlighted_sims'] = {'sim-20201119-190135_parallel_b1_normal_run_g4000_t2000_27_sims': {'ds_res_10_try_2_gen_100d': {1: '1'}, 'gen4000_100foods_res_10_try_2dy': {21: '21'}},
    #                                            'sim-20201119-190204_parallel_b10_normal_run_g4000_t2000_54_sims': {'gen4000_100foods_res_10_try_2dy': {28: '28', 19: '19', 53: '53', 7: '7', 30: '30', 39: '39'}}}

    plot_settings['label_highlighted_sims'] = {'sim-20201119-190135_parallel_b1_normal_run_g4000_t2000_27_sims': {'_intermediate_run_res_40_gen_100d': {1: '1'}, 'gen4000_100foods_intermediate_run_res_40d': {21: '21'}},
                                               'sim-20201119-190204_parallel_b10_normal_run_g4000_t2000_54_sims': {'gen4000_100foods_intermediate_run_res_40d': {28: '28', 19: '19', 53: '53', 7: '7', 30: '30', 39: '39'}}}

    plot_settings['customize_legend_labels'] = True
    # plot_settings['custom_legend_labels'] = {'sim-20201119-190135_parallel_b1_normal_run_g4000_t2000_27_sims': {'ds_res_10_try_2_gen_100d': 'Critical Generation 100', 'gen4000_100foods_res_10_try_2dy': 'Critical Generation 4000'},
    #                                            'sim-20201119-190204_parallel_b10_normal_run_g4000_t2000_54_sims': {'gen4000_100foods_res_10_try_2dy': 'Sub Critical Generation 4000'}}
    plot_settings['custom_legend_labels'] = {'sim-20201119-190135_parallel_b1_normal_run_g4000_t2000_27_sims': {'food_experiment_gen_100_res_10': 'Critical Generation 100', 'food_experiment_res_10': 'Critical Generation 4000'},
                                             'sim-20201119-190204_parallel_b10_normal_run_g4000_t2000_54_sims': {'food_experiment_res_10': 'Sub Critical Generation 4000'}}

    folder_name_dict = {'critical': critical_folder_name_dict, 'sub_critical': sub_critical_folder_name_dict}

    t1 = time.time()

    if plot_settings['only_plot']:
        print('Loading plot_data instead of ising files')
    else:
        print('Loading ising files')
    dynamic_range_main(folder_name_dict, plot_settings)

    t2 = time.time()
    print('total time:', t2-t1)
