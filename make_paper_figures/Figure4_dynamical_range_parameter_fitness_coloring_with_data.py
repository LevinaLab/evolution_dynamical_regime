import matplotlib
matplotlib.use('Agg')
from helper_functions.automatic_plot_helper import all_sim_names_in_parallel_folder
from helper_functions.automatic_plot_helper import detect_all_isings
from helper_functions.automatic_plot_helper import load_isings_from_list
from helper_functions.automatic_plot_helper import choose_copied_isings
from helper_functions.heat_capacity_parameter import calc_heat_cap_param_main
from scipy.interpolate import interp1d
import numpy as np
# from statsmodels.nonparametric.kernel_regression import KernelReg
from scipy.signal import savgol_filter

import matplotlib.pyplot as plt
import os
import pickle
from matplotlib.lines import Line2D
import matplotlib.colors as colors_package
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm

class OneSimPlotData:
    def __init__(self, sim_name, delta_dict, delta_list_dict, fitness_at_given_generation,
                 generation_of_fitness):
        self.sim_name = sim_name
        self.delta_dict = delta_dict
        self.delta_list_dict = delta_list_dict
        self.fitness_at_given_generation = fitness_at_given_generation
        self.generation_of_fitness = generation_of_fitness


def main_plot_parallel_sims(folder_name, plot_settings):
    plt.rc('text', usetex=True)
    font = {'family': 'serif', 'size': 23, 'serif': ['computer modern roman']}
    plt.rc('font', **font)

    if not plot_settings['only_plot']:
        sim_plot_data_list = load_data_from_sims(folder_name, plot_settings)
        save_plot_data(folder_name, sim_plot_data_list, plot_settings)
    else:
        sim_plot_data_list = load_plot_data(folder_name, plot_settings)

    plot(sim_plot_data_list, plot_settings)


def save_plot_data(folder_name, attrs_lists, plot_settings):
    save_dir = 'save/{}/one_pop_plot_data/'.format(folder_name)
    save_name = 'plot_dynamic_range_param_data_with_fitness_last_gen.pickle'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    pickle_out = open(save_dir + save_name, 'wb')
    pickle.dump(attrs_lists, pickle_out)
    pickle_out.close()


def load_plot_data(folder_name, plot_settings):
    save_dir = 'save/{}/one_pop_plot_data/'.format(folder_name)
    save_name = 'plot_dynamic_range_param_data_with_fitness_last_gen.pickle'
    print('Load plot data from: {}{}'.format(save_dir, save_name))

    file = open(save_dir+save_name, 'rb')
    attrs_lists = pickle.load(file)
    file.close()

    return attrs_lists


def create_legend():
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=15, alpha=0.75, label=r'One Generation'),

    ]
    if plot_settings['smooth']:
        legend_elements.append(Line2D([0], [0], color='b', lw=4, c='grey', alpha=0.7, label='One Simulation\nSmoothed'))
    elif plot_settings['interpolate']:
        legend_elements.append(Line2D([0], [0], color='b', lw=4, c='grey', alpha=0.7, label='One Simulation\nInterpolated'))
    elif plot_settings['plot_line']:
        legend_elements.append(Line2D([0], [0], color='b', lw=4, c='grey', alpha=0.7, label='One Simulation'))

    plt.legend(handles=legend_elements, fontsize=17)


def colormap_according_to_delta(delta_dicts_all_sims, generation, plot_settings):
    delta_list_one_gen = []
    for delta_dict in delta_dicts_all_sims:
        # delta dict: mean delta of each generation

        delta_one_gen = delta_dict[str(generation)]
        delta_list_one_gen.append(delta_one_gen)

    colors = ['navy', plot_settings['colors']['b10'], plot_settings['colors']['b1'], plot_settings['colors']['b01']]
    cmap_name = 'custom_cmap'
    # cmap = plt.get_cmap('brg')
    cmap = LinearSegmentedColormap.from_list(
        cmap_name, colors)
    cmap = shiftedColorMap(cmap=cmap, start=0, midpoint=0.38, stop=1, name=cmap_name)
    norm = colors_package.Normalize(vmin=min(delta_list_one_gen), vmax=max(delta_list_one_gen))
    return cmap, norm


def plot(sim_plot_data_list, plot_settings):
    if plot_settings['new_fig']:
        fig_scaling = 1
        plt.figure(figsize=(10*fig_scaling, 5*fig_scaling))
        ax = plt.subplot()
        plt.grid()
    # plt.rcParams.update({
    #     'ytick.right': True,
    #     "ytick.labelright": True
    # })

    fitnesses = [sim_data.fitness_at_given_generation for sim_data in sim_plot_data_list]
    # cmap = sns.cubehelix_palette(as_cmap=True, dark=0.1, light=0.7, reverse=True, rot=-.4)
    # cmap = plt.get_cmap('plasma')
    # cmap = LinearSegmentedColormap.from_list('our_cmap', [plot_settings['our_colors']['our_violet'], plot_settings['our_colors']['our_orange']])
    cmap = LinearSegmentedColormap.from_list('our_cmap', ['xkcd:dark blue', plot_settings['our_colors']['our_violet'], plot_settings['our_colors']['our_orange'], 'xkcd:orangey yellow'])
    # cmap = LinearSegmentedColormap.from_list('our_cmap', [plot_settings['our_colors']['our_violet'], plot_settings['our_colors']['our_orange'], '#ffb871ff'])
    # cmap = shiftedColorMap(cmap, 0, 0.5, 0.7)
    # norm = colors_package.Normalize(vmin=min(fitnesses), vmax=max(fitnesses))
    norm = colors_package.Normalize(vmin=1.9, vmax=max(fitnesses))

    # cmap, norm = colormap_according_to_delta(delta_dicts_all_sims, plot_settings['color_according_to_delta_in_generation'],
    #                                          plot_settings)


    for sim_plot_data in sim_plot_data_list:
        # delta dict: mean delta of each generation
        # deltas_dict: delta of every individual
        delta_dict = sim_plot_data.delta_dict
        deltas_dict = sim_plot_data.delta_list_dict


        # Handle delta dict, which includes mean delta of each generation
        generations = list(delta_dict.keys())
        generations = np.array([int(gen) for gen in generations])
        sorted_gen_indecies = np.argsort(generations)
        generations = np.sort(generations)
        mean_attrs_list = np.array(list(delta_dict.values()))
        mean_attrs_list = mean_attrs_list[sorted_gen_indecies]

        # Handle deltas dict, which includes list of delta of each individual in a generation
        generations_ind = list(deltas_dict.keys())
        generations_ind = np.array([int(gen) for gen in generations_ind])
        sorted_gen_indecies_ind = np.argsort(generations_ind)
        generations_ind = np.sort(generations_ind)
        mean_attrs_list_ind = np.array(list(deltas_dict.values()))
        mean_attrs_list_ind = mean_attrs_list_ind[sorted_gen_indecies_ind]
        # We have a list of delta values for each generation. Unnest the lists and repeat the generations for each
        # individual, such that lists have same dimensions for plotting
        generations_unnested_ind = []
        mean_attr_list_ind_unnested = []
        for gen_ind, mean_attr_list_ind in zip(generations_ind, mean_attrs_list_ind):
            for mean_attr_ind in mean_attr_list_ind:
                generations_unnested_ind.append(gen_ind)
                mean_attr_list_ind_unnested.append(mean_attr_ind)




        color = cmap(norm(sim_plot_data.fitness_at_given_generation))




        if plot_settings['plot_line']:
            '''
            Trying to make some sort of regression, that smoothes and interpolates 
            Trying to find an alternative to moving average, where boundary values are cut off
            '''
            # smoothed_mean_attrs_list = gaussian_kernel_smoothing(mean_attrs_list)
            # Savitzky-Golay filter:
            if plot_settings['smooth']:
                smoothed_mean_attrs_list = savgol_filter(mean_attrs_list, plot_settings['smooth_window'], 3) # window size, polynomial order
            else:
                smoothed_mean_attrs_list = mean_attrs_list
            # plt.plot(generations, smoothed_mean_attrs_list, c=color)


            if plot_settings['interpolate']:
                f_interpolate = interp1d(generations, smoothed_mean_attrs_list, kind='cubic')
                x_interp = np.linspace(np.min(generations), np.max(generations), num=4000, endpoint=True)
                y_interp = f_interpolate(x_interp)
                plt.plot(x_interp, y_interp, c=color, alpha=plot_settings['line_alpha'])
            else:
                plt.plot(generations, smoothed_mean_attrs_list, c=color, alpha=plot_settings['line_alpha'])


        if plot_settings['plot_deltas_of_individuals']:
            plt.scatter(generations_unnested_ind,  mean_attr_list_ind_unnested, s=2, alpha=0.2, c=color)

        # plt.scatter(generations, mean_attrs_list, s=5, alpha=0.4, c=color, marker='.')

        if plot_settings['sliding_window']:
            slided_mean_attrs_list = moving_average(mean_attrs_list, plot_settings['sliding_window_size'])
            plt.plot(generations, slided_mean_attrs_list, alpha=0.8, linewidth=2, c=color)


    plt.xlabel('Generation')
    if plot_settings['label_y_axis']:
        plt.ylabel(r'$\langle \delta \rangle$')
    plt.ylim(plot_settings['ylim'])

    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap))
    cbar.set_label(r'$\langle E \rangle$ at Generation 4000', rotation=270, labelpad=26)


    # plt.text(-200, 1, 'hallo', fontsize=14)
    # plt.subplots_adjust(left=0.9)



    # plt.title(r'$\beta_\mathrm{init}=%s$' % plot_settings['beta_init_for_title'])

    if plot_settings['plot_legend']:
        create_legend()

    if plot_settings['new_fig'] and plot_settings['label_y_axis']:
        pad = -20
        color = 'dimgray'
        ax.annotate('Super-\nCritical', xy=(0, 1.5), xytext=(-ax.yaxis.labelpad - pad, 90),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size=18, ha='right', va='center', rotation=0, color=color)
        ax.annotate('Critical', xy=(0, 1.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size=18, ha='right', va='center', rotation=0, color=color)
        ax.annotate('Sub-\nCritical', xy=(0, 1.5), xytext=(-ax.yaxis.labelpad - pad, -100),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size=18, ha='right', va='center', rotation=0, color=color)

    if not plot_settings['label_y_axis']:
        plt.tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            left=False,      # ticks along the bottom edge are off
            labelleft=False) # labels along the bottom edge are off

    if plot_settings['save_fig']:


        save_dir = 'save/{}/figs/several_plots{}/'.format(folder_name, plot_settings['add_save_name'])
        save_name = 'delta_vs_generations_all_in_one'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        plt.savefig(save_dir+save_name+'.png', bbox_inches='tight', dpi=300)
        plt.savefig(save_dir+save_name+'.pdf', bbox_inches='tight')


def load_data_from_sims(folder_name, plot_settings):
    folder_dir = 'save/{}'.format(folder_name)
    sim_names = all_sim_names_in_parallel_folder(folder_name)
    sim_plot_data_list = []
    for sim_name in sim_names:
        module_settings = {}
        mean_log_beta_distance_dict, log_beta_distance_dict, beta_distance_dict, beta_index_max, betas_max_gen_dict, \
        heat_caps_max_dict, smoothed_heat_caps = calc_heat_cap_param_main(sim_name, module_settings, gaussian_kernel=plot_settings['gaussian_kernel'])
        delta_dict = mean_log_beta_distance_dict
        delta_list_dict = log_beta_distance_dict
        fitness_last_gen, last_gen = load_fitness_last_gen(sim_name, plot_settings)

        sim_plot_data_list.append(OneSimPlotData(sim_name=sim_name,
                                                 delta_dict=delta_dict,
                                                 delta_list_dict=delta_list_dict,
                                                 fitness_at_given_generation=fitness_last_gen,
                                                 generation_of_fitness=last_gen))

        # settings_list.append(load_settings(dir))
    # delta_dicts_all_sims --> men of each generation, deltas_dicts_all_sims --> each individual in a list
    return sim_plot_data_list


def load_fitness_last_gen(sim_name, plot_settings):
    generation = detect_all_isings(sim_name)[-1]
    isings_last_gen = load_isings_from_list(sim_name, [generation], decompress=plot_settings['decompress'])[0]
    isings_last_gen = choose_copied_isings(isings_last_gen)
    fitness_last_gen = np.mean([I.avg_energy for I in isings_last_gen])
    return fitness_last_gen, generation


def below_threshold_nan(isings_list, sim_settings):
    for i, isings in enumerate(isings_list):
        if isings[0].time_steps < plot_settings['min_ts_for_plot']:
            isings_list[i] = None
        if sim_settings['random_food_seasons']:
            if isings[0].food_in_env < plot_settings['min_food_for_plot']:
                isings_list[i] = None
    return isings_list


def slide_window(iterable, win_size):
    slided = []
    x_axis_gens = []
    n = 0
    while n+win_size < len(iterable)-1:
        mean = np.nanmean(iterable[n:n+win_size])
        slided.append(mean)
        x_axis_gens.append(n+int(win_size/2))
        n += 1
    return slided, x_axis_gens


def gaussian(x, mu, sigma):

    C = 1 / (sigma * np.sqrt(2*np.pi))

    return C * np.exp(-1/2 * (x - mu)**2 / sigma**2)


def gaussian_kernel_smoothing(x):
    '''
    Convolving with gaussian kernel in order to smoothen noisy heat cap data (before eventually looking for maximum)
    '''

    # gaussian kernel with sigma=2.25. mu=0 means, that kernel is centered on the data
    # kernel = gaussian(np.linspace(-3, 3, 15), 0, 2.25)
    kernel = gaussian(np.linspace(-3, 3, 15), 0, 6)
    smoothed_x = np.convolve(x, kernel, mode='same')
    return smoothed_x


def moving_average(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


if __name__ == '__main__':
    # folder_name = 'sim-20201020-181300_parallel_TEST'
    plot_settings = {}
    # Only plot loads previously saved plotting file instead of loading all simulations to save time
    plot_settings['only_plot'] = True

    plot_settings['add_save_name'] = ''
    # plot_settings['only_plot_fittest']

    plot_settings['ylim'] = (-1.8, 1.1)
    # This only plots individuals that have not been mutated in previous generation (thus were fittest in previous generation)
    plot_settings['sliding_window'] = False
    plot_settings['sliding_window_size'] = 10

    # smooth works only if plot_settings['interpolate'] = True
    plot_settings['plot_line'] = True
    plot_settings['smooth'] = True
    plot_settings['interpolate'] = True
    plot_settings['smooth_window'] = 7  # 21
    plot_settings['line_alpha'] = 1 # normal 0.6

    plot_settings['decompress'] = True

    plot_settings['plot_deltas_of_individuals'] = False

    plot_settings['gaussian_kernel'] = True

    plot_settings['kernel_regression'] = False
    plot_settings['color_according_to_delta_in_generation'] = 0

    plot_settings['colors'] = {'b1': 'olive', 'b01': 'maroon', 'b10': 'royalblue'}

    beta_inits = [1, 1]

    # folder_names = ['sim-20210302-215811_parallel_beta_linspace_rec_c20_TEST']
    # folder_names = ['sim-20210118-014339_parallel_beta_linspace_break_eat_rec_c40_30_sims']
    # folder_names = ['sim-20210118-014339_parallel_beta_linspace_break_eat_rec_c40_30_sims_HEL_ONLY_PLOT']
    # folder_names = ['sim-20201226-002401_parallel_beta_linspace_rec_c40_30_sims_HEL_ONLY_PLOT']
    # folder_names = ['sim-20201226-002401_parallel_beta_linspace_rec_c40_30_sims']
    folder_names = ['sim-20201226-002401_parallel_beta_linspace_rec_c40_30_sims_HEL_ONLY_PLOT', 'sim-20210118-014339_parallel_beta_linspace_break_eat_rec_c40_30_sims_HEL_ONLY_PLOT']

    plot_settings['our_colors'] = {'lblue': '#8da6cbff', 'iblue': '#5e81b5ff', 'sblue': '#344e73ff',
                                   'lgreen': '#b6d25cff', 'igreen': '#8fb032ff', 'sgreen': '#5e7320ff',
                                   'lred': '#f2977aff', 'ired': '#eb6235ff', 'sred': '#c03e13ff',
                                   'our_orange': '#e87a12ff', 'our_violet': '#3b3a7eff'}

    regimes = ['b1', 'b1']
    plot_settings['last_sim'] = False
    label_x_axis_list = [True, False]
    for i, (folder_name, beta_init, regime, label_x_axis) in enumerate(zip(folder_names, beta_inits, regimes, label_x_axis_list)):
        plot_settings['label_y_axis'] = label_x_axis
        plot_settings['regime'] = regime
        plot_settings['folder_name'] = folder_name
        plot_settings['beta_init_for_title'] = beta_init

        plot_settings['new_fig'] = True

        plot_settings['plot_legend'] = False
        plot_settings['save_fig'] = True


        main_plot_parallel_sims(folder_name, plot_settings)