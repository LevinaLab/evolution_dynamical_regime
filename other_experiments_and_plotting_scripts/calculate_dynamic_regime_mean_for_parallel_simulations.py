import numpy as np
from helper_functions.heat_capacity_parameter import calc_heat_cap_param_main
from helper_functions.automatic_plot_helper import all_sim_names_in_parallel_folder
from scipy import stats


def dynamic_range_main(folder_name, plot_settings):
    delta_dicts_all_sims, deltas_dicts_all_sims = load_dynamic_range_param(folder_name, plot_settings['compare_generation'])

    delta_dict = converting_list_of_dicts_to_dict_of_lists(delta_dicts_all_sims)

    for gen, mean_delta_list_each_sim in delta_dict.items():
        mean_delta_per_sim = np.mean(mean_delta_list_each_sim)
        if plot_settings['compare_generation'] == gen:
            compare_mean_delta_list_per_sim = mean_delta_list_each_sim
        std_delta_per_sim = np.std(mean_delta_list_each_sim)
        print('---\n{}\n---\nGeneration {}:\nMean:{}\n Std:{}\n mean deltas OF INDIVIDUALS of all simulations:{}'.format(
            folder_name, gen, mean_delta_per_sim, std_delta_per_sim, mean_delta_list_each_sim))
    return compare_mean_delta_list_per_sim



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


def load_dynamic_range_param(folder_name, generation):
    folder_dir = 'save/{}'.format(folder_name)
    sim_names = all_sim_names_in_parallel_folder(folder_name)
    delta_dicts_all_sims = []
    deltas_dicts_all_sims = []
    for sim_name in sim_names:
        module_settings = {}
        mean_log_beta_distance_dict, log_beta_distance_dict, beta_distance_dict, beta_index_max, betas_max_gen_dict, \
        heat_caps_max_dict, smoothed_heat_caps = calc_heat_cap_param_main(sim_name, module_settings, gen_list=[generation], gaussian_kernel=True)
        delta_dict = mean_log_beta_distance_dict
        delta_list_dict = log_beta_distance_dict
        delta_dicts_all_sims.append(delta_dict)
        deltas_dicts_all_sims.append(delta_list_dict)


        # settings_list.append(load_settings(dir))
    # delta_dicts_all_sims --> men of each generation, deltas_dicts_all_sims --> each individual in a list
    return (delta_dicts_all_sims, deltas_dicts_all_sims)


if __name__ == '__main__':
    '''
    Calculates means of dynamic regimes between simulations and performs double sided t-test"
    '''

    # Thesis:
    # compare_folder_pairs = [['sim-20201210-200605_parallel_b1_dynamic_range_c_20_g4000_t2000_10_sims', 'sim-20210126-013412_parallel_break_for_food_heat_cap_b_1'],
    #                         ['sim-20201210-200613_parallel_b10_dynamic_range_c_20_g4000_t2000_10_sims', 'sim-20210126-013429_parallel_break_for_food_heat_cap_b_10']]
    # compare_generations = [['4000', '0'], ['4000', '0']]


    #'sim-20210126-013412_parallel_break_for_food_heat_cap_b_1' includes heat capacity calculations in generation 0 from 'sim-20201226-111318_parallel_b1_break_eat_v_eat_max_0_005_g4000_t2000_10_sims' in generation 4000
    # sam applies to 'sim-20210126-013429_parallel_break_for_food_heat_cap_b_10' and 'sim-20201226-111308_parallel_b10_break_eat_v_eat_max_0_005_g4000_t2000_10_sims'

    # delta 0, 0.25, 0.5
    # compare_folder_pairs = [['sim-20201210-200605_parallel_b1_dynamic_range_c_20_g4000_t2000_10_sims', 'sim-20210219-202921_parallel_b0-32_dynamic_range_c_50_g4000_t2000_10_sims'],
    #                         ['sim-20210219-202921_parallel_b0-32_dynamic_range_c_50_g4000_t2000_10_sims', 'sim-20210219-202936_parallel_b0-56_dynamic_range_c_50_g4000_t2000_10_sims'],
    #                         ['sim-20201210-200605_parallel_b1_dynamic_range_c_20_g4000_t2000_10_sims', 'sim-20210219-202936_parallel_b0-56_dynamic_range_c_50_g4000_t2000_10_sims']]

    # compare_folder_pairs = [['sim-20201210-200605_parallel_b1_dynamic_range_c_20_g4000_t2000_10_sims', 'sim-20201226-111318_parallel_b1_break_eat_v_eat_max_0_005_g4000_t2000_10_sims'],
    #                         ['sim-20201210-200613_parallel_b10_dynamic_range_c_20_g4000_t2000_10_sims', 'sim-20201226-111308_parallel_b10_break_eat_v_eat_max_0_005_g4000_t2000_10_sims']]
    # compare_folder_pairs = [['sim-20210226-023914_parallel_b1_default_task_significance_20_runs_delta_last_gen', 'sim-20210226-023902_parallel_b1_break_eat_significance_20_runs_delta_last_gen']]
    compare_folder_pairs = [['sim-20210305-223257_parallel_b1_default_task_significance_44_runs_delta_last_gen', 'sim-20210305-223243_parallel_b1_break_eat_significance_44_runs_delta_last_gen']]


    compare_generations = [['4000', '4000']]

# compare_generations = [['2000', '100'], ['2000', '100']]
    plot_settings = {}
    for compare_folders, generations in zip(compare_folder_pairs, compare_generations):

        compare_samples = []
        for i, (folder_name, gen) in enumerate(zip(compare_folders, generations)):
            plot_settings['compare_generation'] = gen
            compare_sample = dynamic_range_main(folder_name, plot_settings)
            compare_samples.append(compare_sample)
        t_stat, p_value = stats.ttest_ind(compare_samples[0], compare_samples[1], equal_var=False)
        # latest scipy version: alternative='less' --> normal has lower delta (more sub-critical) than break for food
        print('+++\nCompare folders {} and {}\n+++\nT_statistic:{}\np_vlaue:{}'.format(
            compare_folders[0], compare_folders[1], t_stat, p_value))
