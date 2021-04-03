import matplotlib as mpl
mpl.use('Agg') #For server use
from helper_functions.automatic_plot_helper import load_isings_specific_path
from helper_functions.automatic_plot_helper import load_isings_specific_path_decompress
from helper_functions.automatic_plot_helper import choose_copied_isings
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from helper_functions.automatic_plot_helper import all_folders_in_dir_with
import math
import matplotlib.gridspec as gridspec
import time


def main(plot_settings):
    plot_settings['number_individuals'] = 1

    font = {'family': 'serif', 'size': 45, 'serif': ['computer modern roman']}
    plt.rc('font', **font)
    plt.rc('legend', **{'fontsize': 45})

    plot_settings['savefolder_name'] = 'velocites_energies_plot_{}' \
        .format(time.strftime("%Y%m%d-%H%M%S"))
    save_folder = 'save/{}/figs/'.format(plot_settings['savefolder_name'])
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # plt.rcParams.update({'font.size': 22})
    plt.rc('text', usetex=True)

    label_highlighted_sims = plot_settings['label_highlighted_sims']

    num_subplots_page = plot_settings['max_number_of_plots_on_one_page']
    num_subplots_total = 0
    # Count number of plots:
    for folder_name in label_highlighted_sims:
        include_name_dict = label_highlighted_sims[folder_name]
        num_subplots_total += count_total_number_of_plots_in_folder(include_name_dict)

    num_pages = math.ceil(num_subplots_total / num_subplots_page)

    # custom_legend_labels = plot_settings['custom_legend_labels']
    # Making a new plot for each folder and a subplot for each dict in each include_name dict

    plot_info_dict = {}
    for page in range(num_pages):
        for folder_name in label_highlighted_sims:
            include_name_dict = label_highlighted_sims[folder_name]
            for include_name in include_name_dict:
                sim_num_dict = include_name_dict[include_name]
                for sim_num in sim_num_dict:
                    sim_label = sim_num_dict[sim_num]
                    try:
                        sim_label = int(sim_label)
                    except ValueError:
                        pass
                    # plot_info_dict[sim_label] = {'folder_name': folder_name, 'include_name': include_name, 'sim_num': sim_num}
                    plot_info_dict[sim_label] = (folder_name, include_name, sim_num)

    for page in range(num_pages):
        if page == num_pages -1:
            num_subplots_in_curr_plot = num_subplots_total % num_subplots_page
        else:
            num_subplots_in_curr_plot = num_subplots_page

        num_columns = plot_settings['number_columns_in_subplot']
        num_rows = math.floor(num_subplots_in_curr_plot / num_columns)
        fig = plt.figure(figsize=(20, 10*num_rows))
        outer_plot = gridspec.GridSpec(num_rows, num_columns, wspace=0.3, hspace=0.3)
        curr_subplot_num = 0

        sim_label_iter = sorted(list(plot_info_dict.keys()))[page*num_subplots_page : page*num_subplots_page + num_subplots_in_curr_plot]
        # sim_label_iter.reverse()
        for sim_label in sim_label_iter:
            folder_name, include_name, sim_num = plot_info_dict[sim_label]
            inner_plot = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_plot[curr_subplot_num], wspace=0.2, hspace=0.0)
            make_one_subplot(folder_name, include_name, sim_num, fig, inner_plot, plot_settings)
            curr_subplot_num += 1

        fig_name = 'velocities_p{}.png'.format(page+1)
        plt.savefig('{}{}'.format(save_folder, fig_name), dpi=300, bbox_inches='tight')

        # for folder_name in label_highlighted_sims:
        #     include_name_dict = label_highlighted_sims[folder_name]
        #     num_subplots = count_total_number_of_plots_in_folder(include_name_dict)
        #     num_columns = plot_settings['number_columns_in_subplot']
        #     num_rows = math.floor(num_subplots / num_columns)
        #
        #     fig = plt.figure(figsize=(20, 10*num_rows))
        #     outer_plot = gridspec.GridSpec(num_rows, num_columns, wspace=0.3, hspace=0.3)
        #
        #
        #     for include_name in include_name_dict:
        #         sim_num_dict = include_name_dict[include_name]
        #         for sim_num in sim_num_dict:
        #             sim_label = sim_num_dict[sim_num]
        #             sim_position_on_plot = int(sim_label)  - num_subplots_page*page
        #             inner_plot = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_plot[sim_position_on_plot], wspace=0.2, hspace=0.0)
        #             # Creating subplot command
        #             # subplot_input = int('{}{}{}'.format(num_subplots,
        #             #                                     math.floor(curr_subplot_num/2)+1,
        #             #                                     (curr_subplot_num % 2)+1))
        #             # plt.subplot(subplot_input)
        #             make_one_subplot(folder_name, include_name, sim_num, fig, inner_plot, plot_settings)
        #             curr_subplot_num += 1




def count_total_number_of_plots_in_folder(include_name_dict):
    count = 0
    for include_name in include_name_dict:
        sim_num_dict = include_name_dict[include_name]
        for sim_num in sim_num_dict:
            count += 1
    return count



def make_one_subplot(folder_name, include_name, sim_num, fig, inner_plot, plot_settings):

    I = load_all_sims_parallel_folder(folder_name, include_name, sim_num, plot_settings)
    plot_velocities_and_energies(I.energies, I.velocities, fig, inner_plot, folder_name, include_name, sim_num)


def load_all_sims_parallel_folder(folder_name, include_name, sim_num, plot_settings):
    folder_dir = 'save/{}'.format(folder_name)
    dir_list = all_folders_in_dir_with(folder_dir, 'sim')
    for dir in dir_list:
        sim_name = dir[(dir.rfind('save/')+5):]
        sim_num_str = sim_name[(sim_name.rfind('Run_')):]
        sim_num_str = sim_num_str[sim_num_str.rfind('Run_')+4:]
        if sim_num_str == str(sim_num):
            ising = load_from_dynamic_range_data_one_sim(sim_name, include_name, plot_settings)
            return ising


def load_from_dynamic_range_data_one_sim(sim_name, include_name, plot_settings):
    dir = 'save/{}/repeated_generations'.format(sim_name)
    dir_list = all_folders_in_dir_with(dir, include_name)
    plot_settings['plot_varying_number']
    if plot_settings['plot_largest_varying_number']:
        # find largest carying number
        plot_settings['plot_varying_number'] = np.max([get_int_end_of_str(dir) for dir in dir_list])

    # Find dirs that shall be plotted
    dirs_to_plot = []
    for dir in dir_list:
        if get_int_end_of_str(dir) == plot_settings['plot_varying_number']:
            dirs_to_plot.append(dir)

    if len(dirs_to_plot) > 1:
        print('Found more than one simulation in repeated generation folder! Choos ing first detected!')

    if len(dirs_to_plot) == 0:
        print('Did not find varying number (time step) {} in {}. Skip plotting this'
              .format(plot_settings['plot_varying_number'], sim_name))
        return None
    else:
        # TODO: Change loading all isings in this path as soon as we have more than one ising for speed
        if plot_settings['compress_save_isings']:
            # energies and velocitoes are saved in last isings, therefore [-1]
            isings = load_isings_specific_path_decompress(dirs_to_plot[0])[-1]
        else:
            isings = load_isings_specific_path(dirs_to_plot[0])[0]
        if plot_settings['only_copied_isings']:
            isings = choose_copied_isings(isings)

        plot_ind_num = np.random.randint(0, len(isings)-1)
    return isings[plot_ind_num]


def get_int_end_of_str(s):
    m = re.search(r'\d+$', s)
    return int(m.group()) if m else None


def plot_velocities_and_energies(energies_list_attr, velocities_list_attr, fig, inner_plot, folder_name, include_name, sim_num):
    sim_label = plot_settings['label_highlighted_sims'][folder_name][include_name][sim_num]
    color = plot_settings['colors'][folder_name][include_name]

    ax1 = plt.Subplot(fig, inner_plot[0])
    # Title switched off
    # ax1.set_title(r'Simulation ${}$'.format(sim_label), c=color, fontsize=35)
    x_axis_gens = np.arange(len(energies_list_attr))
    ax1.scatter(x_axis_gens, energies_list_attr, s=2, alpha=0.5, c=color)
    # ax1.set_xlabel('Time Step')
    ax1.set_xticks([])
    if not sim_num == 21:
        ax1.set_ylabel(r'$E$')
    ax1.axvline(plot_settings['data_set_trained_on_time_step'], linestyle='dashed', color=plot_settings['our_colors']['sred'], alpha=0.8, linewidth=5)


    ax2 = plt.Subplot(fig, inner_plot[1]) # , sharex=ax1
    x_axis_gens = np.arange(len(velocities_list_attr))
    ax2.scatter(x_axis_gens, velocities_list_attr, s=2, alpha=0.5, c=color)
    ax2.set_xlabel('Time Step, $t$')
    if not sim_num == 21:
        ax2.set_ylabel(r'$v$')
    ax2.axvline(plot_settings['data_set_trained_on_time_step'], linestyle='dashed', color=plot_settings['our_colors']['sred'], alpha=0.8, linewidth=5)

    # shared axis does not work.. also not in Subplot(sharex=ax1)
    # ax1.get_shared_x_axes().join(ax1, ax2)
    fig.add_subplot(ax1)
    fig.add_subplot(ax2)
    # Autocorreltaion
    # plt.subplot(313)
    # y_axis = autocorr(velocities_list_attr)
    # x_axis = np.arange(len(y_axis))
    # plt.scatter(x_axis, y_axis, s=2, alpha=0.5)

    # Fourier transform try1
    # plt.subplot(313)
    # f, Pxx = scipy.signal.welch(velocities_list_attr)
    # x_axis = np.arange(len(Pxx))
    # plt.semilogy(f, Pxx, linewidth=2, alpha=1)

    # Fourier transfor try2
    # The FFT of the signal
    # plt.subplot(313)
    # sig = velocities_list_attr
    # time_step = 0.2
    # sig_fft = fftpack.fft(sig)
    # # And the power (sig_fft is of complex dtype)
    # power = np.abs(sig_fft)
    # # The corresponding frequencies
    # sample_freq = fftpack.fftfreq(np.size(sig), d=time_step)
    # # Plot the FFT power
    # plt.plot(sample_freq, power)
    # plt.xlabel('Frequency [Hz]')
    # plt.ylabel('power')
    # plt.yscale('log')
    # plt.xscale('log')




def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result
    # return result[result.size/2:]


if __name__ == '__main__':

    plot_settings = {}

    critical_folder_name = 'sim-20201119-190135_parallel_b1_normal_run_g4000_t2000_27_sims'
    critical_low_gen_include_name = '_intermediate_run_res_40_gen_100d' # 'ds_res_10_try_2_gen_100d'
    critical_last_gen_include_name = 'gen4000_100foods_intermediate_run_res_40d' # 'gen4000_100foods_res_10_try_2dy'

    sub_critical_folder_name = 'sim-20201119-190204_parallel_b10_normal_run_g4000_t2000_54_sims'
    sub_critical_last_gen_include_name = 'gen4000_100foods_intermediate_run_res_40d' # 'gen4000_100foods_res_10_try_2dy'

    # The label highlighted sims dict is used to choose which velocities to plot
    # plot_settings['label_highlighted_sims'] = {critical_folder_name: {critical_low_gen_include_name: {1: '4', 15: '6'}, critical_last_gen_include_name: {21: '7', 10: '9'}}, 'sim-20201119-190204_parallel_b10_normal_run_g4000_t2000_54_sims': {sub_critical_last_gen_include_name: {28: '1',3: '3', 53: '5', 7: '2', 39: '8', 48: '10'}}}
    plot_settings['label_highlighted_sims'] = {critical_folder_name: {critical_low_gen_include_name: {}, critical_last_gen_include_name: {21: '7'}}, 'sim-20201119-190204_parallel_b10_normal_run_g4000_t2000_54_sims': {sub_critical_last_gen_include_name: {7: '2'}}}


    # plot_settings['colors'] = {critical_folder_name:
    #                                {critical_low_gen_include_name: 'darkorange', critical_last_gen_include_name: 'olive'},
    #                            sub_critical_folder_name: {sub_critical_last_gen_include_name: 'royalblue'}}
    plot_settings['colors'] = {critical_folder_name:
                                   {critical_low_gen_include_name: 'gray', critical_last_gen_include_name: 'gray'},
                               sub_critical_folder_name: {sub_critical_last_gen_include_name: 'gray'}}

    # The varying number is the number of the attribute which is changed in the response plots (foods and time steps)
    # Either the largest number is plotted or a specific number is plotted
    plot_settings['plot_largest_varying_number'] = True
    plot_settings['plot_varying_number'] = 50000
    # TODO: only copied könnte Probleme, geben, da 1. Generation...
    plot_settings['only_copied_isings'] = True

    plot_settings['compress_save_isings'] = True
    plot_settings['number_columns_in_subplot'] = 2

    plot_settings['data_set_trained_on_time_step'] = 2000

    plot_settings['max_number_of_plots_on_one_page'] = 6

    plot_settings['our_colors'] = {'lblue': '#8da6cbff', 'iblue': '#5e81b5ff', 'sblue': '#344e73ff',
                                   'lgreen': '#b6d25cff', 'igreen': '#8fb032ff', 'sgreen': '#5e7320ff',
                                   'lred': '#f2977aff', 'ired': '#eb6235ff', 'sred': '#c03e13ff'}

    #inds = [0]
    main(plot_settings)
