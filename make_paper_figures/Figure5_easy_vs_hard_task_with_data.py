import numpy as np
from helper_functions.heat_capacity_parameter import calc_heat_cap_param_main
from helper_functions.automatic_plot_helper import all_sim_names_in_parallel_folder
import os
import pickle
import matplotlib.pyplot as plt
import time
import warnings
import seaborn as sns
from scipy.signal import savgol_filter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path



def comparison_main(folder_name_dict, plot_settings):
    path = Path(os.getcwd())
    os.chdir(path.parent.absolute())
    plt.rc('text', usetex=True)

    folders_delta_dict = {}
    folders_deltas_dict = {}
    for key, folder_name in folder_name_dict.items():
        if folder_name == '' or folder_name is None:
            pass
        else:
            if plot_settings['only_plot']:
                delta_dict, deltas_dict = load_plot_data(folder_name)
                folders_delta_dict[key] = delta_dict
                folders_deltas_dict[key] = deltas_dict
            else:
                if plot_settings['only_load_generation'] is None:
                    delta_dict, deltas_dict = load_dynamic_range_param(folder_name)
                else:
                    delta_dict, deltas_dict = load_dynamic_range_param(folder_name, [plot_settings['only_load_generation']])
                save_plot_data(folder_name, (delta_dict, deltas_dict))
                folders_delta_dict[key] = delta_dict
                folders_deltas_dict[key] = deltas_dict


    plot_settings['savefolder_name'] = 'dynamical_regime_comparison_plot_{}' \
        .format(time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs('save/{}/figs'.format(plot_settings['savefolder_name']))
    plot_with_inset(folders_delta_dict, folders_deltas_dict, folder_name_dict, plot_settings)



def plot_with_inset(folders_delta_dict, folders_deltas_dict, folder_name_dict, plot_settings):
    font = {'family': 'serif', 'size': 28, 'serif': ['computer modern roman']}
    plt.rc('font', **font)

    fig, ax_main = plt.subplots(figsize=(10, 6))
    ax_in = inset_axes(ax_main, width="45%", height="35%", loc=1)
    plot_continuous(ax_main, folders_delta_dict, plot_settings)
    plot_probability_density(ax_in, folders_delta_dict, folders_deltas_dict, folder_name_dict, plot_settings)


    save_dir = 'save/{}/figs/'.format(plot_settings['savefolder_name'])
    plt.savefig(save_dir + 'simple_vs_hard_task.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_dir + 'simple_vs_hard_task.pdf', bbox_inches='tight')
    plt.show()


def plot_probability_density(ax_in, folders_delta_dict, folders_deltas_dict, folder_name_dict, plot_settings):

    plot_keys = ['folder_simple_last_gen_delta', 'folder_hard_last_gen_delta']
    colors = [plot_settings['our_colors']['lgreen'], plot_settings['our_colors']['sgreen']]
    line_styles = ['solid', 'dashed']
    labels = ['Simple Task', 'Hard Task']
    alphas = [0.5, 0.3]
    # plot_keys.reverse()
    # colors.reverse()
    # labels.reverse()
    # alphas.reverse()
    # line_styles.reverse()
    for plot_key, color, label, line_style, alpha in zip(plot_keys, colors, labels, line_styles, alphas):
        try:
            delta_dict = folders_delta_dict[plot_key]
            if len(list(delta_dict.keys())) == 1:
                mean_delta_list_each_sim = delta_dict[list(delta_dict.keys())[0]]
            else:
                mean_delta_list_each_sim = delta_dict[str(plot_settings['compare_generation'])]
            ax_in.hist(mean_delta_list_each_sim, bins=15, alpha=alpha, color=color, label= label, density=True)
            sns.kdeplot(mean_delta_list_each_sim, ax=ax_in, data2=None, shade=False, vertical=False, color=color, linestyle=line_style, linewidth=3)
            ax_in.axvline(np.mean(mean_delta_list_each_sim), color=color, linestyle=line_style, linewidth=4) # , alpha=alpha*2
            # ax_in.axvline(np.mean(mean_delta_list_each_sim)+np.std(mean_delta_list_each_sim), color=color, alpha=alpha*2, linestyle='dotted', linewidth=3)
            # ax_in.axvline(np.mean(mean_delta_list_each_sim)-np.std(mean_delta_list_each_sim), color=color, alpha=alpha*2, linestyle='dotted', linewidth=3)
        except KeyError:
            warnings.warn('Simulation for {} not loaded'.format(plot_key))

    # plt.legend()
    pad = 5
    ax_in.annotate(r'$\langle \delta \rangle$', xy=(0, 1.5), xytext=(-88, -ax_in.xaxis.labelpad + 18),
                xycoords=ax_in.xaxis.label, textcoords='offset points', ha='right', va='center', rotation=0)
    plt.yticks([], [])
    plt.ylabel('Density')
    # plt.xlabel(r'$\langle \delta \rangle$')
    plt.xlabel('')
    # plt.title(r'$\beta_\mathrm{init}=1$, Generation 4000')




def plot_continuous(ax_main, folders_delta_dict, plot_settings):

    # change this back!!!
    plot_keys = ['folder_simple_continuous_delta', 'folder_hard_continuous_delta']
    # plot_keys = ['folder_simple_continuous_delta']
    # colors = ['olive', 'maroon']
    colors = [plot_settings['our_colors']['lgreen'], plot_settings['our_colors']['sgreen']]
    line_styles = ['solid', 'dashed']
    labels = ['Simple Task', 'Hard Task']
    for plot_key, color, label, line_style in zip(plot_keys, colors, labels, line_styles):
        delta_dict = folders_delta_dict[plot_key]
        plot_gens = []
        plot_mean_deltas = []
        plot_std_delta_low = []
        plot_std_delta_high = []
        for gen_str, mean_deltas_sims in delta_dict.items():
            gen = int(gen_str)
            plot_gens.append(gen)
            mean_delta = np.mean(mean_deltas_sims)
            std_delta = np.std(mean_deltas_sims)
            plot_mean_deltas.append(mean_delta)
            plot_std_delta_low.append(mean_delta - std_delta)
            plot_std_delta_high.append(mean_delta + std_delta)
        plot_gens = np.array(plot_gens)
        plot_mean_deltas = np.array(plot_mean_deltas)
        plot_std_delta_low = np.array(plot_std_delta_low)
        plot_std_delta_high = np.array(plot_std_delta_high)

        sort_gen_inds = np.argsort(plot_gens)
        plot_gens = plot_gens[sort_gen_inds]
        plot_mean_deltas = plot_mean_deltas[sort_gen_inds]
        plot_std_delta_low = plot_std_delta_low[sort_gen_inds]
        plot_std_delta_high = plot_std_delta_high[sort_gen_inds]

        plot_mean_deltas = smooth_svgal(plot_mean_deltas)
        plot_std_delta_low = smooth_svgal(plot_std_delta_low)
        plot_std_delta_high = smooth_svgal(plot_std_delta_high)



        ax_main.plot(plot_gens, plot_mean_deltas, color=color, label=label, linestyle=line_style, linewidth=3)
        ax_main.fill_between(plot_gens, plot_std_delta_low, plot_std_delta_high, alpha=0.5, color=color, linewidth=2)
    # plt.legend()
    ax_main.set_xlabel('Generation')
    ax_main.set_ylabel(r'Dynamical Regime,  $\langle \delta \rangle$', labelpad=10)




def smooth_svgal(x_vec):
    smoothed_x_vec = savgol_filter(x_vec, 11, 3)
    # smoothed_x_vec = x_vec
    return smoothed_x_vec

def converting_list_of_dicts_to_dict_of_lists(delta_dicts_all_sims):
    # converting to
    generations = list(delta_dicts_all_sims[0].keys())
    dict_sorted_by_generatons = {}
    for gen in generations:
        dict_sorted_by_generatons[gen] = []
    for delta_dict in delta_dicts_all_sims:
        for gen in generations:
            dict_sorted_by_generatons[gen].append(delta_dict[gen])
    return dict_sorted_by_generatons


def load_dynamic_range_param(folder_name, gen_list=None):

    sim_names = all_sim_names_in_parallel_folder(folder_name)
    delta_dicts_all_sims = []
    deltas_dicts_all_sims = []
    for sim_name in sim_names:
        module_settings = {}
        mean_log_beta_distance_dict, log_beta_distance_dict, beta_distance_dict, beta_index_max, betas_max_gen_dict, \
        heat_caps_max_dict, smoothed_heat_caps = calc_heat_cap_param_main(sim_name, module_settings, gen_list=gen_list,
                                                                          gaussian_kernel=True)
        delta_dict = mean_log_beta_distance_dict
        delta_list_dict = log_beta_distance_dict
        delta_dicts_all_sims.append(delta_dict)
        deltas_dicts_all_sims.append(delta_list_dict)


        # settings_list.append(load_settings(dir))
    # delta_dicts_all_sims --> men of each generation, deltas_dicts_all_sims --> each individual in a list
    delta_dict = converting_list_of_dicts_to_dict_of_lists(delta_dicts_all_sims)
    deltas_dict = converting_list_of_dicts_to_dict_of_lists(deltas_dicts_all_sims)
    return delta_dict, deltas_dict


def save_plot_data(folder_name, plot_data):
    save_dir = 'save/{}/dynamical_range_plot_data/'.format(folder_name)
    save_name = 'dynamical_range_data_compare_between_folders.pickle'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    pickle_out = open(save_dir + save_name, 'wb')
    pickle.dump(plot_data, pickle_out)
    pickle_out.close()


def load_plot_data(folder_name):
    save_dir = 'save/{}/dynamical_range_plot_data/'.format(folder_name)
    save_name = 'dynamical_range_data_compare_between_folders.pickle'
    print('Load plot data from: {}{}'.format(save_dir, save_name))

    file = open(save_dir+save_name, 'rb')
    attrs_lists = pickle.load(file)
    file.close()

    return attrs_lists

def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

if __name__ == '__main__':

    plot_settings = {}
    plot_settings['only_plot'] = True
    # Only works when 'only_plot' == False, should be None by default!!!!
    plot_settings['only_load_generation'] = None #None #4000
    plot_settings['compare_generation'] = 4000 #10
    plot_settings['our_colors'] = {'lblue': '#8da6cbff', 'iblue': '#5e81b5ff', 'sblue': '#344e73ff',
                                   'lgreen': '#b6d25cff', 'igreen': '#8fb032ff', 'sgreen': '#5e7320ff',
                                   'lred': '#f2977aff', 'ired': '#eb6235ff', 'sred': '#c03e13ff'}
    plot_settings['our_colors']['sgreen'] = adjust_lightness(plot_settings['our_colors']['sgreen'], amount=0.6)


    folder_name_dict = {}
    # folder_name_dict['folder_simple_continuous_delta'] = 'sim-20201210-200605_parallel_b1_dynamic_range_c_20_g4000_t2000_10_sims'
    # folder_name_dict['folder_simple_last_gen_delta'] = 'sim-20210226-023914_parallel_b1_default_task_significance_20_runs_delta_last_gen'
    # folder_name_dict['folder_hard_continuous_delta'] = 'sim-20210226-023745_parallel_b1_break_eat_significance_10_runs_delta_every_20_gen'
    # folder_name_dict['folder_hard_last_gen_delta'] = 'sim-20210226-023902_parallel_b1_break_eat_significance_20_runs_delta_last_gen'
    #
    # folder_name_dict['folder_simple_continuous_delta'] = ''
    # folder_name_dict['folder_simple_last_gen_delta'] = 'sim-20210226-023914_parallel_b1_default_task_significance_20_runs_delta_last_gen'
    # folder_name_dict['folder_hard_continuous_delta'] = ''
    # folder_name_dict['folder_hard_last_gen_delta'] = 'sim-20210226-023902_parallel_b1_break_eat_significance_20_runs_delta_last_gen'

    # folder_name_dict['folder_simple_continuous_delta'] = ''
    # folder_name_dict['folder_simple_last_gen_delta'] = 'sim-20210226-023914_parallel_b1_default_task_significance_20_runs_delta_last_gen_HEL_ONLY_PLOT'
    # folder_name_dict['folder_hard_continuous_delta'] = ''
    # folder_name_dict['folder_hard_last_gen_delta'] = 'sim-20210226-023902_parallel_b1_break_eat_significance_20_runs_delta_last_gen_HEL_ONLY_PLOT'
    # #
    # folder_name_dict['folder_simple_continuous_delta'] = ''
    # folder_name_dict['folder_simple_last_gen_delta'] = 'sim-20210302-215811_parallel_beta_linspace_rec_c20_TEST'
    # folder_name_dict['folder_hard_continuous_delta'] = ''
    # folder_name_dict['folder_hard_last_gen_delta'] = ''

    # folder_name_dict['folder_simple_continuous_delta'] = 'sim-20201210-200605_parallel_b1_dynamic_range_c_20_g4000_t2000_10_sims'
    # folder_name_dict['folder_simple_last_gen_delta'] = ''
    # folder_name_dict['folder_hard_continuous_delta'] = 'sim-20210226-023745_parallel_b1_break_eat_significance_10_runs_delta_every_20_gen'
    # folder_name_dict['folder_hard_last_gen_delta'] = ''

    # folder_name_dict['folder_simple_continuous_delta'] = 'sim-20201210-200605_parallel_b1_dynamic_range_c_20_g4000_t2000_10_sims_HEL_ONLY_PLOT'
    # folder_name_dict['folder_simple_last_gen_delta'] = 'sim-20210226-023914_parallel_b1_default_task_significance_20_runs_delta_last_gen_HEL_ONLY_PLOT'
    # folder_name_dict['folder_hard_continuous_delta'] = 'sim-20210226-023745_parallel_b1_break_eat_significance_10_runs_delta_every_20_gen_HEL_ONLY_PLOT'
    # folder_name_dict['folder_hard_last_gen_delta'] = 'sim-20210226-023902_parallel_b1_break_eat_significance_20_runs_delta_last_gen_HEL_ONLY_PLOT'

    # Only 10 cont simulations:

    # folder_name_dict['folder_simple_continuous_delta'] = 'sim-20201210-200605_parallel_b1_dynamic_range_c_20_g4000_t2000_10_sims_HEL_ONLY_PLOT'
    # folder_name_dict['folder_simple_last_gen_delta'] = 'sim-20201210-200605_parallel_b1_dynamic_range_c_20_g4000_t2000_10_sims_HEL_ONLY_PLOT'
    # folder_name_dict['folder_hard_continuous_delta'] = 'sim-20210226-023745_parallel_b1_break_eat_significance_10_runs_delta_every_20_gen_HEL_ONLY_PLOT'
    # folder_name_dict['folder_hard_last_gen_delta'] = 'sim-20210226-023745_parallel_b1_break_eat_significance_10_runs_delta_every_20_gen_HEL_ONLY_PLOT'
    #
    # Paper plot:
    folder_name_dict['folder_simple_continuous_delta'] = 'sim-20201210-200605_parallel_b1_dynamic_range_c_20_g4000_t2000_10_sims_HEL_ONLY_PLOT'
    folder_name_dict['folder_simple_last_gen_delta'] = 'sim-20210305-223257_parallel_b1_default_task_significance_44_runs_delta_last_gen_HEL_ONLY_PLOT'
    folder_name_dict['folder_hard_continuous_delta'] = 'sim-20210226-023745_parallel_b1_break_eat_significance_10_runs_delta_every_20_gen_HEL_ONLY_PLOT'
    folder_name_dict['folder_hard_last_gen_delta'] = 'sim-20210305-223243_parallel_b1_break_eat_significance_44_runs_delta_last_gen_HEL_ONLY_PLOT'

    # extended plot
    # folder_name_dict['folder_simple_continuous_delta'] = 'sim-20210309-131629_parallel_extend_b1_simple_task_by_4000_generations_HEL_ONLY_PLOT'
    # folder_name_dict['folder_simple_last_gen_delta'] = 'sim-20210305-223257_parallel_b1_default_task_significance_44_runs_delta_last_gen_HEL_ONLY_PLOT'
    # folder_name_dict['folder_hard_continuous_delta'] = 'sim-20210309-134302_parallel_extend_b1_hard_task_by_4000_generations_HEL_ONLY_PLOT'
    # folder_name_dict['folder_hard_last_gen_delta'] = 'sim-20210305-223243_parallel_b1_break_eat_significance_44_runs_delta_last_gen_HEL_ONLY_PLOT'

    # Second simple task iplementation (Other sampling rate!!)
    # folder_name_dict['folder_simple_continuous_delta'] = 'sim-20210310-002622_parallel_b1_dynamic_range_c_20_g8000_t2000_10_sims_UNTIL_GEN_4000_HEL_ONLY_PLOT'
    # folder_name_dict['folder_simple_last_gen_delta'] = 'sim-20210305-223257_parallel_b1_default_task_significance_44_runs_delta_last_gen_HEL_ONLY_PLOT'
    # folder_name_dict['folder_hard_continuous_delta'] = 'sim-20210226-023745_parallel_b1_break_eat_significance_10_runs_delta_every_20_gen_HEL_ONLY_PLOT'
    # folder_name_dict['folder_hard_last_gen_delta'] = 'sim-20210305-223243_parallel_b1_break_eat_significance_44_runs_delta_last_gen_HEL_ONLY_PLOT'
    #
    # folder_name_dict['folder_simple_continuous_delta'] = ''
    # folder_name_dict['folder_simple_last_gen_delta'] = 'sim-20210310-002622_parallel_b1_dynamic_range_c_20_g8000_t2000_10_sims_UNTIL_GEN_4000'
    # folder_name_dict['folder_hard_continuous_delta'] = ''
    # folder_name_dict['folder_hard_last_gen_delta'] = ''



    comparison_main(folder_name_dict, plot_settings)