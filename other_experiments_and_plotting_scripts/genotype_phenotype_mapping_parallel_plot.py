import os
import numpy as np
from helper_functions.automatic_plot_helper import load_isings_specific_path
from helper_functions.automatic_plot_helper import attribute_from_isings
from helper_functions.automatic_plot_helper import all_folders_in_dir_with
from helper_functions.automatic_plot_helper import load_settings
from helper_functions.automatic_plot_helper import mscatter
# import _pickle as pickle
import pickle
import matplotlib.pylab as plt
from matplotlib.lines import Line2D
import re
from helper_functions.automatic_plot_helper import all_sim_names_in_parallel_folder
from helper_functions.automatic_plot_helper import choose_copied_isings
from helper_functions.automatic_plot_helper import calc_normalized_fitness
from helper_functions.automatic_plot_helper import load_isings_specific_path_decompress
import time

import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
from helper_functions.heat_capacity_parameter import calc_heat_cap_param_main
from helper_functions.automatic_plot_helper import custom_color_map
import matplotlib.cm as cm




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
        plot_settings['savefolder_name'] = 'genotype_phenotype_plot_{}_{}' \
            .format(time.strftime("%Y%m%d-%H%M%S"), plot_settings['varying_parameter'])
        os.makedirs('save/{}'.format(plot_settings['savefolder_name']))
        sim_data_list_each_folder = prepare_data(folder_name_dict, plot_settings)
        save_plot_data(sim_data_list_each_folder, plot_settings)
        # Save settings:
        settings_folder = 'save/{}/settings/'.format(plot_settings['savefolder_name'])
        save_settings(settings_folder, plot_settings)
    else:
        sim_data_list_each_folder = load_plot_data(plot_settings['only_plot_folder_name'])
        plot_settings['savefolder_name'] = plot_settings['only_plot_folder_name']
    # if plot_settings['load_dynamic_range_parameter'] or not plot_settings['only_plot']:
    #     sim_data_list_each_folder = load_dynamic_range_parameter(sim_data_list_each_folder, plot_settings)
    #     save_plot_data(sim_data_list_each_folder, plot_settings)

    plot_axis(sim_data_list_each_folder, plot_settings)
    font = {'family': 'serif', 'size': 32, 'serif': ['computer modern roman']}
    plt.rc('font', **font)
    # scatter_plot(sim_data_list_each_folder, plot_settings)
    # dynamic_range_parameter_plot(sim_data_list_each_folder, plot_settings)


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
    font = {'family': 'serif', 'size': 16, 'serif': ['computer modern roman']}
    plt.rc('font', **font)
    plt.rc('legend', **{'fontsize': 10})

    # plt.rcParams.update({'font.size': 22})
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{xcolor}')

    # matplotlib.use('ps')


    plt.figure(figsize=(14, 10))
    ax_main = plt.subplot(111)

    # Make main plot
    # plt.axvline(2000, linestyle='dashed', color='firebrick', alpha=0.8, linewidth=1)
    # plt.text(2200, 250, 'Trained on \n 2000 time steps', color='firebrick', alpha=0.8)

    plot_data(sim_data_list_each_folder, plot_settings, label_each_sim=True, zoomed_axis=False)
    # plt.ylim(-10, 1800)

    # ax_main.set_yticks(ax_main.get_yticks()[:-5])
    # Hide some tick labels:
    # for i in range(1,6):
    #     ax_main.yaxis.get_major_ticks()[-i].draw = lambda *args:None
    # plt.ylabel(r'$\langle \langle \langle E_\mathrm{org} \rangle_\mathrm{life time} \rangle_\mathrm{simulation} \rangle_\mathrm{repeats}$')
    # plt.ylabel(r'$\langle E_\mathrm{org} \rangle$')
    # plt.xlabel('Percentage of food that population was originally trained on')
    # if plot_settings['varying_parameter'] == 'time_steps':
    #     plt.xlabel('Number of time steps')
    # elif plot_settings['varying_parameter'] == 'food':
    #     plt.xlabel('Number of foods')
    #
    # pad = +30
    # color = 'dimgray'
    # ax_main.annotate('Duration of $2$D-simulation', xy=(0, 1.5), xytext=(158, -ax_main.xaxis.labelpad - pad),
    #             xycoords=ax_main.xaxis.label, textcoords='offset points',
    #             size=15, ha='right', va='center', rotation=0, color=color)



    # PLot zoomed-in inset


    # ax_zoom1 = inset_axes(ax_main, 4.3, 4.9, loc='upper right')
    # plt.axvline(2000, linestyle='dashed', color='firebrick', alpha=0.3, linewidth=1)
    # plt.vlines(2000, 42, 70, linestyles='dashed', colors='firebrick', alpha=0.8, linewidth=1)
    # plt.vlines(2000, 0, 4, linestyles='dashed', colors='firebrick', alpha=0.8, linewidth=1)
    #
    # plot_data(sim_data_list_each_folder, plot_settings, label_each_sim=False, y_upper_cut_off_label_sim=70
    #           , x_offset_boo=False, y_offset_boo=False, zoomed_axis=True)
    #
    # ax_zoom1.set_xlim(10000, 52000)
    # ax_zoom1.set_ylim(0, 70)

    # ax_zoom1.set_ylim(1.94, 1.98)
    # ax_zoom1.set_xlim(0, 1)


    # ax_zoom1.xaxis.get_major_ticks()[-1].draw = lambda *args:None

    # plt.yticks(visible=False)
    # plt.xticks(visible=False)
    # Still has to be tested:
    # ax_zoom1.set_xticks([])
    # ax_zoom1.set_yticks([])
    # mark_inset(ax_main, ax_zoom1, loc1=3, loc2=4, fc='none', ec='0.5')


    # ax_zoom2 = inset_axes(ax_main, 3.31, 4.9, loc='upper left')
    # plt.axvline(2000, linestyle='dashed', color='firebrick', alpha=0.3, linewidth=1)
    # plt.vlines(2000, 42, 70, linestyles='dashed', colors='firebrick', alpha=0.8, linewidth=1)
    # plt.vlines(2000, 0, 4, linestyles='dashed', colors='firebrick', alpha=0.8, linewidth=1)
    #
    # plot_data(sim_data_list_each_folder, plot_settings, label_each_sim=False, zoomed_axis=True)
    #
    # ax_zoom2.set_xlim(1500, 10000)
    # ax_zoom2.set_ylim(0, 70)

    # ax_zoom1.set_ylim(1.94, 1.98)
    # ax_zoom1.set_xlim(0, 1)

    # plt.yticks(visible=False)
    # plt.xticks(visible=False)

    # Still has to be tested:
    # ax_zoom1.set_xticks([])
    # ax_zoom1.set_yticks([])
    # ax_zoom2.set_yticks(ax_zoom2.get_yticks()[:-1])
    # mark_inset(ax_main, ax_zoom2, loc1=3, loc2=4, fc='none', ec='0.5')
    #
    # legend_elements = [
    #     Line2D([0], [0], marker='o', color='w', label=r'$\beta_\mathrm{init}=1$, Generation 4000', markerfacecolor=plot_settings['color']['critical'][1],
    #            markersize=20, alpha=0.75),
    #     Line2D([0], [0], marker='o', color='w', label=r'$\beta_\mathrm{init}=1$, Generation 100', markerfacecolor=plot_settings['color']['critical'][0],
    #            markersize=20, alpha=0.75),
    #     Line2D([0], [0], marker='o', color='w', label=r'$\beta_\mathrm{init}=10$, Generation 4000', markerfacecolor=plot_settings['color']['sub_critical'][0],
    #            markersize=20, alpha=0.75),
    # ]
    #
    # ax_zoom2.legend(loc="upper left", bbox_to_anchor=(0.20, -0.23), handles=legend_elements, fontsize=16)

    save_name = 'genotype_phenotype_plot.png'
    save_folder = 'save/{}/figs/'.format(plot_settings['savefolder_name'])
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    plt.savefig(save_folder+save_name, bbox_inches='tight', dpi=300)


def plot_data(sim_data_list_each_folder, plot_settings, label_each_sim=False, y_upper_cut_off_label_sim=None, x_offset_boo=True, y_offset_boo=True, zoomed_axis=False):
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

        if plot_settings['divide_x_value_by_y_value']:
            list_of_avg_attr_list = divide_x_axis_by_y_axis(list_of_avg_attr_list, list_of_food_num_list)

        # food_num_list is not ordered yet, order both lists acc to food_num list for line plotting
        list_of_food_num_list, list_of_avg_attr_list = sort_lists_of_lists(list_of_food_num_list, list_of_avg_attr_list)

        if plot_settings['use_colormaps']:
            color_list_sims = create_color_list(list_of_food_num_list, list_of_avg_attr_list, sim_data_list, plot_settings)
        else:
            colors = plot_settings['color'][sim_data.key]
            try:
                color = colors[sim_data.dynamic_range_folder_includes_index]
            except IndexError:
                raise IndexError('Color list is out of bounds check whether dynamic_range_folder_includes_list is longer'
                                 ' than color lists in color dict')
            color_list_sims = [color for _ in sim_data_list]

        avg_of_avg_attr_list = []
        # This goes through all lists and takes averages of the inner nesting, such that instead of a list of lists
        # we have one list with average value of each entriy of the previous lists,
        # in future do this with np. array and define axis to take average over
        for i in range(len(list_of_avg_attr_list[0])):
            avg_of_avg_attr_list.append(np.mean([list_of_avg_attr_list[j][i] for j in range(len(list_of_avg_attr_list))]))

        marker = plot_settings['marker'][sim_data.folder_num_in_key]



        # Plot each simulation
        for food_num_list, avg_attr_list, color in zip(list_of_food_num_list, list_of_avg_attr_list, color_list_sims):
            plt.scatter(food_num_list, avg_attr_list, marker=marker, c=color, s=3, alpha=0.2)
        # Connect each simulation datapoint with lines
        for food_num_list, avg_attr_list, sim_data, color in zip(list_of_food_num_list, list_of_avg_attr_list, sim_data_list, color_list_sims):
            if sim_data.highlight_this_sim:
                if zoomed_axis:
                    alpha = 0.5
                    linewidth=2
                else:
                    alpha = 0.5
                    linewidth=2
                plt.plot(food_num_list, avg_attr_list, c=color, alpha=alpha, linewidth=linewidth)
            else:
                if zoomed_axis:
                    alpha = 0.5
                    linewidth=2
                else:
                    alpha = 0.5
                    linewidth=2
                plt.plot(food_num_list, avg_attr_list, c=color, alpha=alpha, linewidth=linewidth)

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

                x_offset = 300
                y_offset = 0
                if x_offset_boo:
                    try:
                        x_add_offset = plot_settings['x_offsets_for_labels'][sim_data.folder_name][sim_data.dynamic_range_folder_includes][sim_data.label]
                    except KeyError:
                        x_add_offset = 0
                else:
                    x_add_offset = 0

                if y_offset_boo:
                    try:
                        y_add_offset = plot_settings['y_offsets_for_labels'][sim_data.folder_name][sim_data.dynamic_range_folder_includes][sim_data.label]
                    except KeyError:
                        y_add_offset = 0
                else:
                    y_add_offset = 0

                y_offset += y_add_offset
                x_offset += x_add_offset
                # coordinates = (food_num_list[-1]+x_offset, avg_attr_list[-1]+y_offset)

                plot_this_label = True
                if y_upper_cut_off_label_sim is not None:
                    if avg_attr_list[-1] > y_upper_cut_off_label_sim:
                        plot_this_label = False

                label = sim_data.label #sim_data.sim_name[sim_data.sim_name.rfind('Run_')+4:]  # TODO check whether this is run number!
                if plot_settings['highlight_certain_sims']:
                    fontsize = 14
                else:
                    fontsize = 3
                # if (label is not None) and plot_this_label:
                #     plt.text(coordinates[0], coordinates[1], label, fontsize=fontsize, c=color)


def create_color_list(list_of_food_num_list, list_of_avg_attr_list, sim_data_list, plot_settings):
    if plot_settings['custom_colormaps']:
        plot_settings['colormaps'] = {critical_folder_name: {critical_low_gen_include_name: 'critical_low_gen',
                                                             critical_last_gen_include_name: 'critical_last_gen'},
                                      'sim-20201119-190204_parallel_b10_normal_run_g4000_t2000_54_sims':
                                          {sub_critical_last_gen_include_name: 'sub_critical_last_gen'}}
    # Extract colormap, that shall currently be used
    colormaps = plot_settings['colormaps']
    sim_data_folder = sim_data_list[0]
    for folder_name in colormaps:
        if folder_name == sim_data_folder.folder_name:
            include_name_cmap_dict = colormaps[folder_name]
            for include_name in include_name_cmap_dict:
                if include_name == sim_data_folder.dynamic_range_folder_includes:
                    colormap = include_name_cmap_dict[include_name]

    if plot_settings['custom_colormaps']:
        # The two colors of the custom color map can be adjusted
        # by changing the numbers of CustomCmap in the order of r,g,b
        if colormap == 'critical_low_gen':
            colormap = CustomCmap([0.1, 0.0, 0.0], [0.7, 0.0, 0.0])
        elif colormap == 'critical_last_gen':
            colormap = CustomCmap([0.00, 0.1, 0.0], [0.0, 0.7, 0.0])
        elif colormap == 'sub_critical_last_gen':
            colormap = CustomCmap([0.00, 0.00, 0.1], [0.0, 0.0, 0.7])

    list_of_avg_attr_list_arr = np.array(list_of_avg_attr_list)
    for food_num_list in list_of_food_num_list:
        if not food_num_list == list_of_food_num_list[0]:
            raise Exception('Different x_axis in loaded data sets. Cannot create colormap')
    food_num_list = list_of_food_num_list[0]
    i_where = food_num_list.index(plot_settings['trained_on_varying_parameter_value'])
    vals_at_trained_vary = list_of_avg_attr_list_arr[:, i_where]
    cmap = plt.get_cmap(colormap)

    norm = colors.Normalize(vmin=min(vals_at_trained_vary), vmax=max(vals_at_trained_vary))

    colors_list_sims = list(map(lambda x: cmap(norm(x)), vals_at_trained_vary))

    return colors_list_sims


def scatter_plot(sim_data_list_each_folder, plot_settings):
    fig = plt.figure(figsize=(10,10))
    max_var_param_list = []
    for sim_data_list in sim_data_list_each_folder:
        avg_attr_trained_var_param_list = []
        avg_attr_max_var_param_list = []
        for sim_data in sim_data_list:
            i_trained_var_param = sim_data.food_num_list.index(plot_settings['trained_on_varying_parameter_value'])
            max_var_param = np.max(sim_data.food_num_list)
            max_var_param_list.append(max_var_param)
            i_max_var_param = sim_data.food_num_list.index(max_var_param)
            avg_attr_trained_var_param = sim_data.avg_attr_list[i_trained_var_param]
            avg_attr_max_var_param = sim_data.avg_attr_list[i_max_var_param]

            avg_attr_trained_var_param_list.append(avg_attr_trained_var_param)
            avg_attr_max_var_param_list.append(avg_attr_max_var_param)


        colors = plot_settings['color'][sim_data.key]
        try:
            color = colors[sim_data.dynamic_range_folder_includes_index]
        except IndexError:
            raise IndexError('Color list is out of bounds check whether dynamic_range_folder_includes_list is longer'
                             ' than color lists in color dict')
        plt.scatter(avg_attr_trained_var_param_list, avg_attr_max_var_param_list, c=color, alpha=0.5)

    if not all_equal(max_var_param_list):
        raise BaseException('The maximal varying parameter (time steps) differs between some of the loaded simulation')
    xlim = plt.xlim()[1]

    # Unity line (check exactly what this does!)
    slope = max_var_param / plot_settings['trained_on_varying_parameter_value']

    x_arr = np.linspace(0, xlim, 1000)
    y_arr = [x*slope for x in x_arr]
    plt.plot(x_arr, y_arr, c='darkcyan', linestyle='dashed', alpha=0.5, linewidth=3)

    # Arbitrary threshold line of 5
    slope = 5

    x_arr = np.linspace(0, xlim, 1000)
    y_arr = [x*slope for x in x_arr]
    plt.plot(x_arr, y_arr, c='grey', linestyle='dashed', alpha=0.5, linewidth=3)


    # plt.plot([0, xlim], [0, xlim * slope])
    plt.ylabel(r'$\langle E_\mathrm{org} \rangle$ %s time steps' % (max_var_param))
    plt.xlabel(r'$\langle E_\mathrm{org} \rangle$ %s time steps' % (plot_settings['trained_on_varying_parameter_value']))

    plt.yscale('log')


    # Legend
    legend_elements = [
        Line2D([0], [0], color='b', lw=4, c='darkcyan', linestyle='dashed', alpha=0.7,
               label=r'$\frac{\langle E_\mathrm{org} \rangle \textrm{ } 50000 \textrm{ \small{time steps}}}{\langle E_\mathrm{org} \rangle \textrm{ } 2000 \textrm{ \small{time steps}}} = \frac{50000}{2000}$'),
        Line2D([0], [0], color='b', lw=4, c='grey', linestyle='dashed', alpha=0.7,
               label=r'arbitrary seperation $\frac{\langle E_\mathrm{org} \rangle \textrm{ } 50000 \textrm{ \small{time steps}}}{\langle E_\mathrm{org} \rangle \textrm{ } 2000 \textrm{ \small{time steps}}} = 5$'),
        Line2D([0], [0], marker='o', color='w', label=r'$\beta_\mathrm{init}=1$, Generation 4000', markerfacecolor=plot_settings['color']['critical'][1],
               markersize=20, alpha=0.75),
        Line2D([0], [0], marker='o', color='w', label=r'$\beta_\mathrm{init}=1$, Generation 100', markerfacecolor=plot_settings['color']['critical'][0],
               markersize=20, alpha=0.75),
        Line2D([0], [0], marker='o', color='w', label=r'$\beta_\mathrm{init}=10$, Generation 4000', markerfacecolor=plot_settings['color']['sub_critical'][0],
               markersize=20, alpha=0.75),
    ]

    plt.legend(loc="lower right", handles=legend_elements, fontsize=20)

    # mpl.patches.Ellipse((25, 50), 100, 100, angle=160)
    save_name = 'scatter_plot.png'
    save_folder = 'save/{}/figs/'.format(plot_settings['savefolder_name'])

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    plt.savefig(save_folder+save_name, bbox_inches='tight', dpi=300)


def load_dynamic_range_parameter(sim_data_list_each_folder, plot_settings):
    '''
    Updates sim_data_list_each_folder with dynamic range parameter
    '''
    for sim_data_list in sim_data_list_each_folder:
        for sim_data in sim_data_list:
            alternative = True
            try:
                alternative_folder_name = plot_settings['alternative_folder_name_for_heat_capacities'][sim_data.folder_name][sim_data.dynamic_range_folder_includes]

            except KeyError:
                complete_sim_name = sim_data.sim_name
                alternative = False
            if alternative:
                complete_sim_name = recreate_sim_name_for_alternativ_folder(sim_data, alternative_folder_name, plot_settings)
            calc_heat_cap_param_settings = {}
            mean_log_beta_distance_dict, log_beta_distance_dict, beta_distance_dict, beta_index_max, betas_max_gen_dict, heat_caps_max_dict, smoothed_heat_caps\
                = calc_heat_cap_param_main(complete_sim_name, calc_heat_cap_param_settings)
            dynamic_range_param_dict = mean_log_beta_distance_dict
            gens_dynamic_range_param_dict = list(dynamic_range_param_dict.keys())
            # !!!! Always loads in heat capacity from last detected generation !!!!
            index_last_heat_cap_gen = np.argsort(np.array(gens_dynamic_range_param_dict, dtype=int))[-1]
            last_heat_cap_gen = gens_dynamic_range_param_dict[index_last_heat_cap_gen]
            dynamic_range_param_last_heat_cap_gen = dynamic_range_param_dict[last_heat_cap_gen]

            sim_data.dynamic_range_param = dynamic_range_param_last_heat_cap_gen
    return sim_data_list_each_folder


def recreate_sim_name_for_alternativ_folder(sim_data, alternative_folder_name, plot_settings):
    sim_names = all_sim_names_in_parallel_folder(alternative_folder_name)

    for it_sim_name in sim_names:
        if 'Run_{}'.format(sim_data.sim_num) in it_sim_name:
            return it_sim_name
    raise Exception('No simulation found with "Run_{}" in {} out of {}'.format(sim_data.sim_num, alternative_folder_name, sim_names))


def dynamic_range_parameter_plot(sim_data_list_each_folder, plot_settings):
    '''
    Make sure, that dynamic range parameter has been loaded into sim_data_list_each_folder before calling this!
    '''



    color_list = []
    marker_list=[]
    fitness_at_largest_varying_param_list = []
    dynamic_range_param_list = []
    ratio_largest_trained_varying_param_list = []
    fitness_at_trained_varying_param_list = []
    color_at_trained_fitness_list = []

    largest_varying_param_list = []
    trained_varying_param_list = []
    for sim_data_list in sim_data_list_each_folder:
        for sim_data in sim_data_list:
            # Pick out fitness for highest varying parameter (food_num)
            index_largest_varying_param = np.argmax(sim_data.food_num_list)
            largest_varying_param = sim_data.food_num_list[index_largest_varying_param]
            fitness_at_largest_varying_param = sim_data.avg_attr_list[index_largest_varying_param]

            index_trained_varying_param = sim_data.food_num_list.index(plot_settings['trained_on_varying_parameter_value'])
            trained_varying_param = sim_data.food_num_list[index_trained_varying_param]
            fitness_at_trained_varying_param = sim_data.avg_attr_list[index_trained_varying_param]

            ratio_largest_trained_varying_param = (fitness_at_largest_varying_param / fitness_at_trained_varying_param)\
                                                  / (largest_varying_param / trained_varying_param)

            colors = plot_settings['color'][sim_data.key]
            markers = plot_settings['marker2'][sim_data.key]
            try:
                color = colors[sim_data.dynamic_range_folder_includes_index]
                marker = markers[sim_data.dynamic_range_folder_includes_index]
            except IndexError:
                raise IndexError('Color list is out of bounds check whether dynamic_range_folder_includes_list is longer'
                                 ' than color lists in color dict')
            color_list.append(color)
            marker_list.append(marker)
            fitness_at_largest_varying_param_list.append(fitness_at_largest_varying_param)
            dynamic_range_param_list.append(sim_data.dynamic_range_param)
            ratio_largest_trained_varying_param_list.append(ratio_largest_trained_varying_param)

            largest_varying_param_list.append(largest_varying_param)
            trained_varying_param_list.append(trained_varying_param)
            fitness_at_trained_varying_param_list.append(fitness_at_trained_varying_param)

    #
    cmap, norm = custom_color_map(colors=['xkcd:darkish blue', 'xkcd:darkish green', 'xkcd:darkish red'], min_val=min(fitness_at_trained_varying_param_list),
                                  max_val=max(fitness_at_trained_varying_param_list), cmap_name='fitness_cmap')
    # cmap = plt.get_cmap('turbo')
    cmap = plt.get_cmap('plasma')
    colors_for_trained_fitness = [cmap(norm(fitness)) for fitness in fitness_at_trained_varying_param_list]
    # Check whether all entries in largest_varying_param_list are equal
    if not len(set(largest_varying_param_list)) == 1:
        raise BaseException('Different largest_varying_params')



    fig, ax_lab = plt.subplots()
    plt.figure(figsize=(10, 10))

    ratio_largest_trained_varying_param_list_log = list(map(lambda x: np.log10(x), ratio_largest_trained_varying_param_list))

    # plt.scatter(fitness_at_largest_varying_param_list, dynamic_range_param_list, c=color_list, alpha=0.5)
    mscatter(ratio_largest_trained_varying_param_list, dynamic_range_param_list, c=colors_for_trained_fitness, alpha=0.5,
             m=marker_list)
    # plt.scatter(ratio_largest_trained_varying_param_list_log, dynamic_range_param_list, c=color_list, alpha=0.5)
    # plt.xscale('log')

    # plt.xlabel('{}/{}'.format(largest_varying_param_list[0], trained_varying_param_list[0]))
    # plt.xlabel(r'$\langle E_\mathrm{org} \rangle$ 50000 time steps $/$ $\langle E_\mathrm{org} \rangle$ 2000 time steps')
    plt.xlabel(r'Generalizability $\gamma_{t}$')
    plt.ylabel(r'Dynamical Regime $\langle \delta \rangle$')
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap))
    cbar.set_label(r'$\langle E_\mathrm{org} \rangle$ at Training Condition', rotation=270, labelpad=33)

    pad = -30
    color = 'dimgray'


    # plt.axvline(5, alpha=0.5, linestyle='dashed', color='grey', linewidth=3)

    plt.axvline(1, alpha=0.5, linestyle='dashed', color='darkcyan', linewidth=3)

    '''
    Legende:
    wieder critical gen 100, 4000 und sub-critical gen 4000
    
    graue linje: arbitratry threshold E time step 50000 / E time step 2000 = 5
    türquise Linje: Unity (?) E time step 50000 / E time step 2000 =  50000/2000 = 25
    '''

    ax_lab.annotate('Duration of $2$D-simulation', xy=(0, 0), xytext=(0, -ax_lab.xaxis.labelpad - pad),
                    xycoords=ax_lab.xaxis.label, textcoords='offset points',
                    size=15, ha='right', va='center', rotation=0, color=color)

    legend_elements = [
        Line2D([0], [0], color='b', lw=4, c='darkcyan', linestyle='dashed', alpha=0.7,
               label=r'Linear Generalizability'),

        Line2D([0], [0], marker=plot_settings['marker2']['critical'][0], color='w', label=r'Simulation $\beta_\mathrm{init}=1$', markerfacecolor='grey',
               markersize=20, alpha=0.75),
        Line2D([0], [0], marker=plot_settings['marker2']['sub_critical'][0], color='w', label=r'Simulation $\beta_\mathrm{init}=10$', markerfacecolor='grey',
               markersize=20, alpha=0.75),
    ]
    # Line2D([0], [0], color='b', lw=4, c='grey', linestyle='dashed', alpha=0.7,
           # label=r'arbitrary seperation \newline $\frac{\langle E_\mathrm{org} \rangle \textrm{ } 50000 \textrm{ \small{time steps}}}{\langle E_\mathrm{org} \rangle \textrm{ } 2000 \textrm{ \small{time steps}}} = 5$'),

    plt.legend(loc='upper left', handles=legend_elements, fontsize=32, bbox_to_anchor=(0, 1.32))

    save_name = 'fitness_largest_time_step_num_vs_dynamic_range_param.png'
    save_folder = 'save/{}/figs/'.format(plot_settings['savefolder_name'])

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    plt.savefig(save_folder+save_name, bbox_inches='tight', dpi=300)


def all_equal(lst):
    # Are all entries in list identical?
    return not lst or lst.count(lst[0]) == len(lst)


def divide_x_axis_by_y_axis(list_of_avg_attr_list, list_of_food_num_list):
    list_of_avg_attr_list_new = []
    for avg_attr_list, food_num_list in zip(list_of_avg_attr_list, list_of_food_num_list):
        avg_attr_list_new = list(np.array(avg_attr_list) / np.array(food_num_list))
        list_of_avg_attr_list_new.append(avg_attr_list_new)
    return list_of_avg_attr_list_new


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


def CustomCmap(from_rgb, to_rgb):

    # from color r,g,b
    r1,g1,b1 = from_rgb

    # to color r,g,b
    r2,g2,b2 = to_rgb

    cdict = {'red': ((0, r1, r1),
                     (1, r2, r2)),
             'green': ((0, g1, g1),
                       (1, g2, g2)),
             'blue': ((0, b1, b1),
                      (1, b2, b2))}

    cmap = LinearSegmentedColormap('custom_cmap', cdict)
    return cmap


if __name__ == '__main__':
    # In these dicts all folders, with parallel runs, that shall be loaded must be specified as keys.
    # The entry of each key is a list of all "dynamic_range_folder_includes", which is a string for each run of the
    # dynamic_range_parallel_pipline. This string is a characteristic substring of the folder name of the runs that
    # shall be loaded in the dynamic range folder of each simulation
    #

    critical_folder_name = 'sim-20210206-122918_parallel_b1_normal_run_g4000_t2000_54_sims' #'sim-20201210-200605_parallel_b1_dynamic_range_c_20_g4000_t2000_10_sims'
    # critical_folder_name = 'sim-20201210-200605_parallel_b1_genotype_phenotype_test'
    critical_low_gen_include_name = '_intermediate_run_res_40_gen_100d'
    critical_last_gen_include_name = 'second_big_10_runs_all_connectable_mutate_ALL_edges_0-005_5_repeats'

    sub_critical_folder_name = 'sim-20201119-190204_parallel_b10_normal_run_g4000_t2000_54_sims' #'sim-20201210-200613_parallel_b10_dynamic_range_c_20_g4000_t2000_10_sims'
    # sub_critical_folder_name = 'sim-20201210-200613_parallel_b10_genotype_phenotype_test'
    sub_critical_last_gen_include_name = 'second_big_10_runs_all_connectable_mutate_ALL_edges_0-005_5_repeats'

    # Plot with generation 100 critical:
    # critical_folder_name_dict = {critical_folder_name: [critical_low_gen_include_name, critical_last_gen_include_name]}
    critical_folder_name_dict = {critical_folder_name: [critical_last_gen_include_name]}
    sub_critical_folder_name_dict = {sub_critical_folder_name: [sub_critical_last_gen_include_name]}



    plot_settings = {}
    plot_settings['varying_parameter'] = 'time_steps'  # 'time_steps' or 'food'
    plot_settings['only_plot'] = True
    plot_settings['load_dynamic_range_parameter'] = True

    plot_settings['only_plot_folder_name'] = 'genotype_phenotype_plot_20210226-020759_time_steps_HUGE_RUN_2x54_SIMS'
    # plot_settings['only_plot_folder_name'] = 'response_plot_20201125-211925_time_steps_2000ts_fixed_CritGen100_3999_SubCritGen3999_huge_run_resolution_50_3_repeats'

    plot_settings['add_save_name'] = ''
    plot_settings['only_copied'] = True
    plot_settings['attr'] = 'avg_energy'
    # Colors for each dynamical regime. The color lists of each dynamical regime are chosen by the index of the
    # currently plotted entry of dynamic_range_folder_includes_list
    plot_settings['color'] = {'critical': ['olive', 'darkorange', 'turquoise'],
                              'sub_critical': ['royalblue', 'pink', 'magenta'],
                              'super_critical': ['maroon', 'red', 'steelblue']}

    plot_settings['marker2'] = {'critical': ['o'],
                               'sub_critical': ['X'],
                               'super_critical': []}
    # This setting defines the markers, which are used in the order that the folder names are listed
    plot_settings['marker'] = ['.', 'x', '+']
    # This feature looks for compressed ising-files and decompresses them
    plot_settings['compress_save_isings'] = True
    # This plots the means of all simulations in one folder for one value of the y-axis
    plot_settings['plot_means'] = False
    plot_settings['divide_x_value_by_y_value'] = False
    plot_settings['trained_on_varying_parameter_value'] = 2000
    plot_settings['critical_folder_name_dict'] = critical_folder_name_dict
    plot_settings['sub_critical_folder_name_dict'] = sub_critical_folder_name_dict

    # When heat capacity values are saved in a different folder name (parallel simulation) than where include_names are, specify here
    plot_settings['alternative_folder_name_for_heat_capacities'] = {critical_folder_name: {critical_low_gen_include_name: 'sim-20210117-224238_parallel_respone_plot_gen_100_heat_cap'}}

    #  This feature highlights certain simulation runs and relabels them. Those simulations, that shall be highlighted
    #  and relabeled have to be specified in plot_settings['label_highlighted_sims']. All other simulations are not
    #  labeled
    #  plot_settings['label_highlighted_sims'] is a dict of dicts of dicts with the following shape:
    #  {folder_name_1: {include_name_1: {simulation_number: new_label_1}, ...}, ...}
    #  The include name ("dynamic_range_folder_includes") has to be equal to the one used in the folder_name_dict s.
    plot_settings['highlight_certain_sims'] = True
    # plot_settings['label_highlighted_sims'] = {critical_folder_name: {critical_low_gen_include_name: {1: '1', 15: '15'}, critical_last_gen_include_name: {21: '21', 10: '10'}}, 'sim-20201119-190204_parallel_b10_normal_run_g4000_t2000_54_sims': {sub_critical_last_gen_include_name: {28: '28',3: '3', 53: '53', 7: '7', 39: '39', 48: '48'}}}
    # plot_settings['label_highlighted_sims'] = {critical_folder_name: {critical_low_gen_include_name: {1: '7', 15: '5'}, critical_last_gen_include_name: {21: '4', 10: '2'}}, 'sim-20201119-190204_parallel_b10_normal_run_g4000_t2000_54_sims': {sub_critical_last_gen_include_name: {28: '10',3: '8', 53: '6', 7: '9', 39: '3', 48: '1'}}}
    plot_settings['label_highlighted_sims'] = {critical_folder_name: {critical_low_gen_include_name: {1: '4', 15: '6'}, critical_last_gen_include_name: {21: '7', 10: '9'}}, sub_critical_folder_name: {sub_critical_last_gen_include_name: {28: '1',3: '3', 53: '5', 7: '2', 39: '8', 48: '10'}}}
    # plot_settings['label_highlighted_sims'] = {'sim-20201119-190135_parallel_b1_normal_run_g4000_t2000_27_sims': {'ds_res_10_try_2_gen_100d': {1: '1'}, 'gen4000_100foods_res_10_try_2dy': {21: '21'}},
    #                                            'sim-20201119-190204_parallel_b10_normal_run_g4000_t2000_54_sims': {'gen4000_100foods_res_10_try_2dy': {28: '28', 19: '19', 53: '53', 7: '7', 30: '30', 39: '39'}}}

    #  The offset label dicts give the offsets for a label of the simulation. !! Use the label of the simulation as key; not the number of the simulation !!
    plot_settings['x_offsets_for_labels'] = {critical_folder_name: {critical_low_gen_include_name: {}, critical_last_gen_include_name: {}}, sub_critical_folder_name: {sub_critical_last_gen_include_name: {'3': 800, '5': 800, '8': 800}}}
    plot_settings['y_offsets_for_labels'] = {critical_folder_name: {critical_low_gen_include_name: {'4': -10}, critical_last_gen_include_name: {}}, sub_critical_folder_name: {sub_critical_last_gen_include_name: {'2': -8, '3': -6.5}}}
    # plot_settings['label_highlighted_sims'] = {'sim-20201119-190135_parallel_b1_normal_run_g4000_t2000_27_sims': {'_intermediate_run_res_40_gen_100d': {1: '1'}, 'gen4000_100foods_intermediate_run_res_40d': {21: '21'}},
    #                                            'sim-20201119-190204_parallel_b10_normal_run_g4000_t2000_54_sims': {'gen4000_100foods_intermediate_run_res_40d': {28: '28', 19: '19', 53: '53', 7: '7', 30: '30', 39: '39'}}}

    plot_settings['customize_legend_labels'] = True
    # plot_settings['custom_legend_labels'] = {'sim-20201119-190135_parallel_b1_normal_run_g4000_t2000_27_sims': {'ds_res_10_try_2_gen_100d': 'Critical Generation 100', 'gen4000_100foods_res_10_try_2dy': 'Critical Generation 4000'},
    #                                            'sim-20201119-190204_parallel_b10_normal_run_g4000_t2000_54_sims': {'gen4000_100foods_res_10_try_2dy': 'Sub Critical Generation 4000'}}
    # plot_settings['custom_legend_labels'] = {'sim-20201119-190135_parallel_b1_normal_run_g4000_t2000_27_sims': {'_intermediate_run_res_40_gen_100d': 'Critical Generation 100', 'gen4000_100foods_intermediate_run_res_40d': 'Critical Generation 4000'},
    #                                          'sim-20201119-190204_parallel_b10_normal_run_g4000_t2000_54_sims': {'gen4000_100foods_intermediate_run_res_40d': 'Sub Critical Generation 4000'}}



    plot_settings['custom_legend_labels'] = {critical_folder_name: {critical_low_gen_include_name: 'Critical Generation 100', critical_last_gen_include_name: 'Critical Generation 4000'}, 'sim-20201119-190204_parallel_b10_normal_run_g4000_t2000_54_sims': {sub_critical_last_gen_include_name: 'Sub Critical Generation 4000'}}

    plot_settings['use_colormaps'] = False
    # 'custom_colormaps' can only be activated when 'use_colormaps' is active
    plot_settings['custom_colormaps'] = True
    # non-custom colormaps for when plot_settings['custom_colormaps'] = False
    # plot_settings['colormaps'] = {critical_folder_name: {critical_low_gen_include_name: 'autumn', critical_last_gen_include_name: 'summer'}, 'sim-20201119-190204_parallel_b10_normal_run_g4000_t2000_54_sims': {sub_critical_last_gen_include_name: 'winter'}}

    folder_name_dict = {'critical': critical_folder_name_dict, 'sub_critical': sub_critical_folder_name_dict}

    t1 = time.time()

    if plot_settings['only_plot']:
        print('Loading plot_data instead of ising files')
    else:
        print('Loading ising files')
    dynamic_range_main(folder_name_dict, plot_settings)

    t2 = time.time()
    print('total time:', t2-t1)
