#!/usr/bin/env python

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os import path, makedirs
from helper_functions.automatic_plot_helper import load_settings
import os
from helper_functions.heat_capacity_parameter import calc_heat_cap_param_main
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.colors as colors


def main(sim_name, settings, generation_list, recorded, plot_settings, draw_original_heat_cap_data=True, draw_dynamic_range_param=False, draw_legend=False, draw_critical=False, draw_smoothed_heat_caps=False, gaussian_kernel=False):
    '''
    generation list can be set to None
    recorded is a boolean defining whether we want to visualize recorded heat capacity or dream heat capacity
    '''

    # TODO: make these scripts take these as params
    loadfile = sim_name
    folder = 'save/' + loadfile

    R, thermal_time, beta_low, beta_high, beta_num, y_lim_high = settings['heat_capacity_props']
    #R = 10
    Nbetas = beta_num
    betas = 10 ** np.linspace(beta_low, beta_high, Nbetas)
    numAgents = settings['pop_size']
    size = settings['size']

    if generation_list is None:
        if recorded:
            generation_list = automatic_generation_generation_list(folder + '/C_recorded')
        else:
            generation_list = automatic_generation_generation_list(folder + '/C')
    iter_gen = generation_list

    # TODO: Repeat averaging seems to work
    C = np.zeros((R, numAgents, Nbetas, len(iter_gen)))


    print('Loading data...')
    for ii, iter in enumerate(iter_gen):
        #for bind in np.arange(0, 100):
        for bind in np.arange(1, Nbetas):
            if recorded:
                #  Depending on whether we are dealing with recorded or dream heat capacity
                filename = folder + '/C_recorded/C_' + str(iter) + '/C-size_' + str(size) + '-Nbetas_' + \
                           str(Nbetas) + '-bind_' + str(bind) + '.npy'
            else:
                filename = folder + '/C/C_' + str(iter) + '/C-size_' + str(size) + '-Nbetas_' + \
                           str(Nbetas) + '-bind_' + str(bind) + '.npy'
            C[:, :, bind, ii] = np.load(filename)
    print('Done.')

    plt.rc('text', usetex=True)
    font = {'family': 'serif', 'size': 30, 'serif': ['computer modern roman']}
    plt.rc('font', **font)
    plt.rc('legend', **{'fontsize': 20})

    b = 0.8
    alpha = 0.3

    print('Generating figures...')
    for ii, iter in enumerate(iter_gen):

        fig, ax = plt.subplots(1, 1, figsize=(14, 10), sharex=True)
        fig.text(0.51, 0.030, r'$\beta_{fac}$', ha='center', fontsize=30)
        fig.text(0.005, 0.5, r'$C_\mathrm{H}/N$', va='center', rotation='vertical', fontsize=30)
        title = 'Specific Heat of Foraging Community\n Generation: ' + str(iter)
        # fig.suptitle(title)
        if draw_dynamic_range_param:
            plot_dynamic_range_parameter_background(sim_name, betas, iter, gaussian_kernel)

        # CHANGE THIS TO CUSTOMIZE HEIGHT OF PLOT
        #upperbound = 1.5 * np.max(np.mean(np.mean(C[:, :, :-40, :], axis=0), axis=0))
        # upperbound = np.max(np.mean(np.mean(C, axis=0)), axis=0)
        #upperbound = 0.4

        # upperbound = y_lim_high / 100

        upperbound = 0.25
        upperbound = 0.25
        label = iter

        cm = plt.get_cmap('gist_earth')  # gist_ncar # gist_earth #cmocean.cm.phase
        ax.set_prop_cycle(color=[cm(1.*i/numAgents) for i in range(numAgents)])
        if draw_original_heat_cap_data:
            for numOrg in range(numAgents):

                ax.scatter(betas, np.mean(C[:, numOrg, :, ii], axis=0),
                           s=30, alpha=0.3, marker='.', label=label)  # color=[0, 0, 0],

        if draw_smoothed_heat_caps:
            module_settings={}
            mean_log_beta_distance_dict, log_beta_distance_dict, beta_distance_dict, beta_index_max, betas_max_gen_dict, heat_caps_max_dict, smoothed_heat_caps_dict \
                = calc_heat_cap_param_main(sim_name, module_settings, [iter], gaussian_kernel=gaussian_kernel)
            ax.set_prop_cycle(color=[cm(1.*i/numAgents) for i in range(numAgents)])
            for numOrg in range(numAgents):
                # c = np.dot(np.random.random(), [1, 1, 1])

                ax.scatter(betas, smoothed_heat_caps_dict[iter][numOrg], s=2, alpha=0.3, marker='o', label=label, color='grey')
                ax.plot(betas, smoothed_heat_caps_dict[iter][numOrg], linewidth=1, alpha=0.35, label=label, color='grey')

        if draw_dynamic_range_param:
            plot_dynamic_range_parameter(sim_name, betas, iter, draw_critical, gaussian_kernel, plot_settings)

        xticks = [0.01, 0.05, 0.1, 0.5, 1, 2, 10, 20, 100]
        ax.set_xscale("log") # , nonposx='clip'
        # This makes custom x ticks
        # ax.set_xticks(xticks)
        # This makes x-ticks
        # formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))
        # ax.get_xaxis().set_major_formatter(formatter)


        low_xlim = 10 ** beta_low
        high_xlim = 10 ** beta_high
        plt.axis([low_xlim, high_xlim, 0, upperbound])

        if draw_legend:
            plot_legend(cm, draw_smoothed_heat_caps)
            # plot_legend2(cm)

        # leg = plt.legend(loc=2, title='Generation')
        #
        # for lh in leg.legendHandles:
        #     lh.set_alpha(1)
        #     lh.set_sizes(30)
        if recorded:
            savefolder = folder + '/figs/C_recorded/'
        else:
            savefolder = folder + '/figs/C/'
        savefilename = savefolder + 'C-size_' + str(size) + '-Nbetas_' + \
                       str(Nbetas) + '-gen_' + str(iter) + '.png'
        if not path.exists(savefolder):
            makedirs(savefolder)

        plt.savefig(savefilename, bbox_inches='tight', dpi=300)
        plt.close()
        # plt.clf()
        savemsg = 'Saving ' + savefilename
        print(savemsg)
        # plt.show()
        # plt.pause(0.1)


def plot_legend(cmap, draw_smoothed_heat_caps):
    norm = colors.Normalize(vmin=1, vmax=4)
    legend_elements = [
        # Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(norm(1.3)),
        #        markersize=15, alpha=0.75),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(norm(2)),
               markersize=15, alpha=0.75, label=r'Organisms'),
        # Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(norm(3)),
        #        markersize=15, alpha=0.75),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='maroon',
               markersize=15, alpha=0.75, label=r'Maximum'),
        Patch(facecolor='maroon', edgecolor='w', label=r'$\mathrm{std}(\beta_\mathrm{fac}^\mathrm{crit})$', alpha=0.4),
        Line2D([0], [0], color='b', lw=3, c='maroon', linestyle='dashed', alpha=0.7, label=r'$\langle \beta_\mathrm{fac}^\mathrm{crit} \rangle$'),
        Line2D([0], [0], color='b', lw=3, c='darkorange', linestyle='dashed', alpha=0.7, label=r'$\beta_\mathrm{fac}^\mathrm{orig} = 1$'),
        Line2D([0], [0], color='b', lw=4, c='darkcyan', linestyle='dotted', alpha=0.7, label=r'$\langle \delta \rangle$'),
        # collections.CircleCollection((15,16,17), cmap=cmap, offsets=((150,0),(50,0),(100,0)), label='bla',transOffset=5)
    ]
    if draw_smoothed_heat_caps:
        add_element = Line2D([0], [0], color='b', lw=4, c='grey', alpha=0.7, label=r'Organism Smoothed')
        legend_elements.append(add_element)

    # handler_map = {('bla1','bla2'): HandlerTuple(ndivide=None)}
    plt.legend(loc="upper right", handles=legend_elements, fontsize=26.5)


# class HandlerColormap(HandlerBase):
#     def __init__(self, cmap, num_stripes=8, **kw):
#         HandlerBase.__init__(self, **kw)
#         self.cmap = cmap
#         self.num_stripes = num_stripes
#
#     def create_artists(self, legend, orig_handle,
#                        xdescent, ydescent, width, height, fontsize, trans):
#         stripes = []
#         for i in range(self.num_stripes):
#             s = Rectangle([xdescent + i * width / self.num_stripes, ydescent],
#                           width / self.num_stripes,
#                           height,
#                           fc=self.cmap((2 * i + 1) / (2 * self.num_stripes)),
#                           transform=trans)
#             stripes.append(s)
#         return stripes
#
#
# def plot_legend2(cmap):
#     legend_elements = [
#
#         Line2D([0], [0], marker='o', color='w', markerfacecolor='maroon',
#                markersize=15, alpha=0.75, label=r'$\mathrm{max}(C_\mathrm{H}/N)_\mathrm{org}$'),
#         Patch(facecolor='maroon', edgecolor='w', label=r'$\mathrm{std} (\mathrm{max}(C_\mathrm{H}/N)_\mathrm{org}) _ {\beta_\mathrm{fac}}$', alpha=0.4),
#         Line2D([0], [0], color='b', lw=3, c='maroon', linestyle='dashed', alpha=0.7, label=r'$\langle \mathrm{max}(C_\mathrm{H}/N)_\mathrm{org} \rangle _ {\beta_\mathrm{fac}}$'),
#         Line2D([0], [0], color='b', lw=3, c='darkorange', linestyle='dashed', alpha=0.7, label='Line'),
#         Line2D([0], [0], color='b', lw=4, c='darkcyan', linestyle='dotted', alpha=0.7, label='Line')
#
#     ]
#
#     cmaps = [cmap]
#     cmap_labels = ["bla"]
#     for line_obj in legend_elements:
#         cmap_labels.append(line_obj._label)
#     # create proxy artists as handles:
#     cmap_handles = [Rectangle((0, 0), 1, 1) for _ in cmaps]
#     handler_map = dict(zip(cmap_handles,
#                            [HandlerColormap(cm, num_stripes=8) for cm in cmaps]))
#
#     legend_elements = cmap_handles.append(legend_elements)
#
#     plt.legend(handles=legend_elements,
#                labels=cmap_labels,
#                handler_map=handler_map,
#                fontsize=12)


def plot_dynamic_range_parameter(sim_name, betas, generation, draw_critical, gaussian_kernel, plot_settings):
    module_settings = {}

    gen_list = [generation]
    mean_log_beta_distance_dict, log_beta_distance_dict, beta_distance_dict, beta_index_max, betas_max_gen_dict, heat_caps_max_dict, smoothed_heat_caps \
        = calc_heat_cap_param_main(sim_name, module_settings, gen_list, gaussian_kernel=gaussian_kernel)
    mean_log_beta_distance = mean_log_beta_distance_dict[generation]
    mean_beta_distance = np.mean(betas_max_gen_dict[generation])
    mean_max_betas = np.mean(betas_max_gen_dict[generation])
    std_max_betas = np.std(betas_max_gen_dict[generation])

    # Mark max beta values red
    plt.scatter(betas_max_gen_dict[generation], heat_caps_max_dict[generation], s=10, c='maroon')
    # Mean max beta values
    plt.axvline(mean_max_betas, c='maroon', linestyle='dashed', alpha=0.7, linewidth=2)
    # Mark hypothetical critical value
    plt.axvline(1, c='darkorange', linestyle='dashed', alpha=0.7, linewidth=2)
    # if mean_log_beta_distance > 1.1 and mean_log_beta_distance < 0.9:
    if draw_critical:
        text_y_pos = mean_beta_distance + (mean_beta_distance * 0.4)
        plt.text(text_y_pos, 0.21, r'$\langle \delta \rangle = %s$' % np.round(mean_log_beta_distance, decimals=2), fontsize=35)
        plt.title(plot_settings['title'])

    else:
        if mean_beta_distance < 1:
            x_min = mean_beta_distance
            x_max = 1
            text_y_pos = mean_beta_distance + (mean_beta_distance * 0.4)
            plt.text(text_y_pos, 0.21, r'$\langle \delta \rangle = %s$' % np.round(mean_log_beta_distance, decimals=2), fontsize=35)
            plt.title(plot_settings['title'])
        else:
            x_min = 1
            x_max = mean_beta_distance
            text_y_pos = 1 + (1 * 0.4)
            plt.text(text_y_pos, 0.21, r'$\langle \delta \rangle = %s$' % np.round(mean_log_beta_distance, decimals=2), fontsize=35)
            plt.title(plot_settings['title'])
        plt.hlines(0.2, x_min, x_max, linestyles='dotted', linewidths=5, colors='darkcyan')
    return smoothed_heat_caps




def plot_dynamic_range_parameter_background(sim_name, betas, generation, gaussian_kernel):
    module_settings = {}

    gen_list = [generation]
    mean_log_beta_distance_dict, log_beta_distance_dict, beta_distance_dict, beta_index_max, betas_max_gen_dict, heat_caps_max_dict, smoothed_heat_caps \
        = calc_heat_cap_param_main(sim_name, module_settings, gen_list, gaussian_kernel=gaussian_kernel)
    mean_log_beta_distance = mean_log_beta_distance_dict[generation]
    mean_max_betas = np.mean(betas_max_gen_dict[generation])
    std_max_betas = np.std(betas_max_gen_dict[generation])
    plt.axvspan(mean_max_betas - std_max_betas, mean_max_betas + std_max_betas, alpha=0.1, color='maroon')


def automatic_generation_generation_list(C_folder):
    C_gen_folders = [f.path for f in os.scandir(C_folder) if f.is_dir()]
    generation_list = get_generations(C_gen_folders)
    return generation_list


def get_generations(C_gen_folders):
    generation_list = []
    for C_gen_folder in C_gen_folders:
        if RepresentsInt(C_gen_folder.split('_')[-1]) is True:
            generation_list.append(C_gen_folder.split('_')[-1])
    return generation_list


def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

if __name__ == '__main__':
    # sim_name = 'sim-20201210-200605_parallel_b1_dynamic_range_c_20_g4000_t2000_10_sims_HEL_ONLY_PLOT/sim-20201210-200606-b_1_-g_4001_-t_2000_-compress_-noplt_-rec_c_20_-c_4_-subfolder_sim-20201210-200605_parallel_b1_dynamic_range_c_20_g4000_t2000_10_sims_-n_Run_1' #'sim-20201204-203157-g_2_-t_20_-rec_c_1_-c_props_100_10_-2_2_100_40_-c_20_-noplt_-n_heat_cap_test_default_setup' #'sim-20200916-192139-g_2_-t_2000_-rec_c_1_-c_props_10000_100_-2_2_300_40_-c_20_-noplt_-n_FINE_RESOLVED_HEAT_CAP_PLOT_THESIS_PLOT' # 'sim-20201207-145420-g_2_-b_10_-t_20_-rec_c_1_-c_props_100_10_-2_2_100_40_-c_20_-noplt_-n_heat_cap_TEST_default_setup'
    # sim_name = 'sim-20201210-200613_parallel_b10_dynamic_range_c_20_g4000_t2000_10_sims_HEL_ONLY_PLOT/sim-20201210-200614-b_10_-g_4001_-t_2000_-compress_-noplt_-rec_c_20_-c_4_-subfolder_sim-20201210-200613_parallel_b10_dynamic_range_c_20_g4000_t2000_10_sims_-n_Run_1'
    sim_name = 'sim-20201211-211021_parallel_b0_1_dynamic_range_c_20_g4000_t2000_10_sims_HEL_ONLY_PLOT/sim-20201211-211024-b_0.1_-g_4001_-t_2000_-compress_-noplt_-rec_c_20_-c_7_-subfolder_sim-20201211-211021_parallel_b0_1_dynamic_range_c_20_g4000_t2000_10_sims_-n_Run_1'
    plot_settings = {}
    plot_settings['title'] = r'$\beta_\mathrm{init} = 0.1$ Generation $4000$'
    generation_list = [0]
    settings = load_settings(sim_name)
    recorded = True
    draw_original_heat_cap_data = True
    draw_dynamic_range_param = True
    draw_legend = False
    draw_critical = False

    # Use gaussian kernel in order to smooth heat capacity curves before calculating maximum
    gaussian_kernel = True
    # Draw the smoothed kurves as well
    draw_smoothed_heat_caps = True
    main(sim_name, settings, None, recorded, plot_settings, draw_original_heat_cap_data, draw_dynamic_range_param, draw_legend, draw_critical, draw_smoothed_heat_caps, gaussian_kernel)
