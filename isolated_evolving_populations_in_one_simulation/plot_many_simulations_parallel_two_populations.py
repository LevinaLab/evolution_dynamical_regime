import os
import numpy as np
from helper_functions.automatic_plot_helper import all_folders_in_dir_with
from helper_functions.automatic_plot_helper import load_isings_attr
from helper_functions.automatic_plot_helper import load_isings_specific_path
from helper_functions.automatic_plot_helper import attribute_from_isings
from helper_functions.automatic_plot_helper import load_settings
from helper_functions.automatic_plot_helper import calc_normalized_fitness
import matplotlib.pylab as plt
import pickle

from helper_functions.isolated_population_helper import seperate_isolated_populations


def plot_any_simulations_parallel(folder_name, plot_settings, load_plot_data_only):
    add_name = '_seperated'
    if not load_plot_data_only:
        attrs_lists_all_sims_critical, attrs_lists_all_sims_sub_critical = load_seperated_simulations(folder_name, plot_settings)

        save_plot_data(folder_name, (attrs_lists_all_sims_critical, attrs_lists_all_sims_sub_critical), add_name)
    else:
        attrs_lists_all_sims_critical, attrs_lists_all_sims_sub_critical = load_plot_data(folder_name, add_name)

    plot_seperated_sims(attrs_lists_all_sims_critical, attrs_lists_all_sims_sub_critical, folder_name)

    plot_seperated_sims_2_plots(attrs_lists_all_sims_critical, attrs_lists_all_sims_sub_critical, folder_name)


def save_plot_data(folder_name, plot_data, add_name):
    save_folder = 'save/{}/plot_data/'.format(folder_name)
    save_name = '{}plot_data{}.pickle'.format(folder_name, add_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    pickle_out = open(save_folder+save_name, 'wb')
    pickle.dump(plot_data, pickle_out)
    pickle_out.close()


def load_plot_data(folder_name, add_name):
    load_folder = 'save/{}/plot_data/'.format(folder_name)
    load_name = '{}plot_data{}.pickle'.format(folder_name, add_name)
    file = open(load_folder+load_name, 'rb')
    plot_data = pickle.load(file)
    file.close()
    return plot_data


def plot_seperated_sims(attrs_lists_all_sims_critical, attrs_lists_all_sims_sub_critical, folder_name):
    plt.figure(figsize=(10, 7))
    generations = np.arange(len(attrs_lists_all_sims_critical[0]))
    for attrs_list_critical, attrs_list_sub_critical in zip(attrs_lists_all_sims_critical, attrs_lists_all_sims_sub_critical):


        mean_attrs_list_critical = [np.mean(gen_attrs) for gen_attrs in attrs_list_critical]
        mean_attrs_list_sub_critical = [np.mean(gen_attrs) for gen_attrs in attrs_list_sub_critical]

        # plt.plot(generations, attrs_list_critical, c=plot_settings['color']['critical'], linewidth=0.2, alpha=0.2)
        # plt.plot(generations, attrs_list_sub_critical, c=plot_settings['color']['sub_critical'], linewidth=0.2, alpha=0.2)

        plt.scatter(generations, mean_attrs_list_critical, c=plot_settings['color']['critical'], s=0.5, alpha=0.2)
        plt.scatter(generations, mean_attrs_list_sub_critical, c=plot_settings['color']['sub_critical'], s=0.5, alpha=0.2)
    plt.xlabel('Generation')
    plt.ylabel(plot_settings['attr'])
    save_dir = 'save/{}/figs/several_plots{}/'.format(folder_name, plot_settings['add_save_name'])
    save_name = 'several_sims_{}.png'.format(plot_settings['attr'])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_dir+save_name, bbox_inches='tight', dpi=300)


def plot_seperated_sims_2_plots(attrs_lists_all_sims_critical, attrs_lists_all_sims_sub_critical, folder_name):
    plt.figure(figsize=(10, 7))
    generations = np.arange(len(attrs_lists_all_sims_critical[0]))
    for attrs_list_critical in attrs_lists_all_sims_critical:


        mean_attrs_list_critical = [np.mean(gen_attrs) for gen_attrs in attrs_list_critical]


        # plt.plot(generations, mean_attrs_list_critical, linewidth=0.2, alpha=0.2)
        plt.scatter(generations, mean_attrs_list_critical, s=2, alpha=0.2)

        # plt.scatter(generations, mean_attrs_list_critical, c=plot_settings['color']['critical'], s=0.5, alpha=0.2)
        # plt.scatter(generations, mean_attrs_list_sub_critical, c=plot_settings['color']['sub_critical'], s=0.5, alpha=0.2)

    plt.xlabel('Generation')
    plt.ylabel(plot_settings['attr'])
    plt.ylim(plot_settings['ylim'])

    save_dir = 'save/{}/figs/several_plots{}/'.format(folder_name, plot_settings['add_save_name'])
    save_name = 'several_sims_criticial_{}.png'.format(plot_settings['attr'])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(save_dir+save_name, bbox_inches='tight', dpi=300)

    plt.figure(figsize=(10, 7))
    for attrs_list_sub_critical in attrs_lists_all_sims_sub_critical:
        mean_attrs_list_sub_critical = [np.mean(gen_attrs) for gen_attrs in attrs_list_sub_critical]
        # plt.plot(generations, mean_attrs_list_sub_critical, linewidth=0.2, alpha=0.2)
        plt.scatter(generations, mean_attrs_list_sub_critical, s=2, alpha=0.2)

    plt.xlabel('Generation')
    plt.ylabel(plot_settings['attr'])
    plt.ylim(plot_settings['ylim'])
    save_name = 'several_sims_sub_criticial_{}.png'.format(plot_settings['attr'])
    plt.savefig(save_dir+save_name, bbox_inches='tight', dpi=300)



# def avg_every_generation(attrs_list):
#     return [np.mean(gen_attrs) for gen_attrs in attrs_list]


def load_simulations_single_population(folder_name, plot_settings):
    '''
    Untested!!
    '''
    folder_dir = 'save/{}'.format(folder_name)
    dir_list = all_folders_in_dir_with(folder_dir, 'sim')
    attrs_lists_sims = []
    for sim_name in dir_list:
        attrs_list = load_isings_attr(sim_name, plot_settings['attr'])
        attrs_lists_sims.append(attrs_list)
    return attrs_lists_sims




def load_seperated_simulations(folder_name, plot_settings):
    folder_dir = 'save/{}'.format(folder_name)
    dir_list = all_folders_in_dir_with(folder_dir, 'sim')
    attrs_lists_all_sims_critical = []
    attrs_lists_all_sims_sub_critical = []
    for dir in dir_list:
        sim_name = dir[(dir.rfind('save/')+5):]
        isings_list = load_isings_specific_path('{}/isings'.format(dir))
        if plot_settings['attr'] == 'norm_avg_energy' or  plot_settings['attr'] == 'norm_food_and_ts_avg_energy':
            settings = load_settings(sim_name)
            calc_normalized_fitness(isings_list, plot_settings, settings)
        isings_list_seperated = seperate_isolated_populations(isings_list)
        isings_list_critical = isings_list_seperated[0]
        isings_list_sub_critical = isings_list_seperated[1]
        attrs_list_critical = [attribute_from_isings(isings_crit, plot_settings['attr']) for isings_crit in isings_list_critical]
        attrs_list_sub_critical = [attribute_from_isings(isings_sub, plot_settings['attr']) for isings_sub in isings_list_sub_critical]

        attrs_lists_all_sims_critical.append(attrs_list_critical)
        attrs_lists_all_sims_sub_critical.append(attrs_list_sub_critical)


    return attrs_lists_all_sims_critical, attrs_lists_all_sims_sub_critical

if __name__ == '__main__':
    load_plot_data_only = False

    folder_name = 'sim-20201019-153950_parallel_parallel_mean_4000_ts_b10_fixed_ts' #'sim-20201014-004324_parallel_g1000_random_ts' #'sim-20201014-004136_parallel_g1000_fixed_ts' #'sim-20201005-205252_parallel_g1000_rand_ts' # 'sim-20201012-220954_parallel_g1000_fixed_ts'
    plot_settings = {}
    plot_settings['add_save_name'] = ''
    plot_settings['attr'] = 'norm_avg_energy' # 'avg_energy'
    plot_settings['color'] = {'critical': 'darkorange', 'sub_critical': 'royalblue', 'super_critical': 'maroon'}
    plot_settings['ylim'] = (-0.001, 0.015)



    plot_any_simulations_parallel(folder_name, plot_settings, load_plot_data_only)