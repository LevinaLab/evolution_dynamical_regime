import train
import copy
from helper_functions.automatic_plot_helper import detect_all_isings
from helper_functions.automatic_plot_helper import load_settings
from helper_functions.automatic_plot_helper import all_sim_names_in_parallel_folder
import ray
import numpy as np



def dynamic_pipeline_all_sims(folder_names, pipeline_settings):
    for folder_name in folder_names:
        sim_names = all_sim_names_in_parallel_folder(folder_name)

        if not pipeline_settings['parallelize_each_sim']:
            for i, sim_name in enumerate(sim_names):
                if pipeline_settings['only_plot_certain_num_of_simulations'] is None:
                    dynamic_pipeline_one_sim(sim_name, pipeline_settings)
                elif pipeline_settings['only_plot_certain_num_of_simulations'] > i:
                    dynamic_pipeline_one_sim(sim_name, pipeline_settings)
        else:
            all_sim_names = np.array([])
            for folder_name in folder_names:
                sim_names = all_sim_names_in_parallel_folder(folder_name)
                all_sim_names = np.append(all_sim_names, sim_names)

            ray.init(num_cpus=pipeline_settings['cores'])
            if pipeline_settings['specify_memory_usage']:
                ray_funcs = [dynamic_pipeline_one_sim_remote_memory.remote(sim_name, pipeline_settings)for sim_name in all_sim_names]
            else:
                ray_funcs = [dynamic_pipeline_one_sim_remote.remote(sim_name, pipeline_settings)for sim_name in all_sim_names]
            ray.get(ray_funcs)
            ray.shutdown()

@ray.remote
def dynamic_pipeline_one_sim_remote(sim_name, pipeline_settings):

    original_settings = load_settings(sim_name)
    settings = create_settings_for_repeat(original_settings, sim_name, pipeline_settings)
    run_all_repeats(settings, original_settings, pipeline_settings)


# Exact copy of run_repeat_remote but with specific memory usage. Memory usage par task!!
@ray.remote(memory=1500 * 1024 * 1024)
def dynamic_pipeline_one_sim_remote_memory(sim_name, pipeline_settings):

    original_settings = load_settings(sim_name)
    settings = create_settings_for_repeat(original_settings, sim_name, pipeline_settings)
    run_all_repeats(settings, original_settings, pipeline_settings)

# Exact copy of run_repeat_remote but without ray.remote decorator
def dynamic_pipeline_one_sim(sim_name, pipeline_settings):

    original_settings = load_settings(sim_name)
    settings = create_settings_for_repeat(original_settings, sim_name, pipeline_settings)
    run_all_repeats(settings, original_settings, pipeline_settings)


def create_settings_for_repeat(settings, sim_name, pipeline_settings):
    # settings['TimeSteps'] = 5
    if pipeline_settings['varying_parameter'] == 'time_steps':
        settings['random_time_steps'] = False
    elif pipeline_settings['varying_parameter'] == 'food':
        settings['random_food_seasons'] = False

    settings = copy.deepcopy(settings)

    complete_sim_folder = sim_name
    settings['loadfile'] = complete_sim_folder

    if pipeline_settings['load_last_generation']:
        settings['iter'] = detect_all_isings(complete_sim_folder)[-1]
        pipeline_settings['load_generation'] = detect_all_isings(complete_sim_folder)[-1]
    else:
        settings['iter'] = pipeline_settings['load_generation']
    settings['LoadIsings'] = True
    settings['switch_off_evolution'] = True
    settings['save_data'] = False
    settings['switch_seasons_repeat_pipeline'] = True
    settings['dynamic_range_pipeline'] = True
    # Animations:
    settings['plot_generations'] = pipeline_settings['animation_for_repeats']
    settings['repeat_pipeline_switched_boo'] = None
    settings['random_time_steps_power_law'] = False
    settings['commands_in_folder_name'] = False
    settings['plot_pipeline'] = False
    # switches off animation:
    settings['plot'] = False
    settings['save_energies_velocities_last_gen'] = False

    settings['compress_save_isings'] = pipeline_settings['compress_save_isings']

    return settings


def run_all_repeats(settings, original_settings, pipeline_settings):
    # WATCH OUT !!! PARAMETERS WITH "FOOD" IN THEM CAN ALSO BECOME TIME STEPS !!!
    if pipeline_settings['varying_parameter'] == 'time_steps':
        if not original_settings['random_time_steps']:
            original_mean_food_num = original_settings['TimeSteps']
        else:
            original_mean_food_num = (settings['random_time_step_limits'][0] + settings['random_time_step_limits'][1]) / 2

        # if original_settings['random_time_steps_power_law']:
        #     print('!!! random_time_steps_power_law is not supported !!!')


    elif pipeline_settings['varying_parameter'] == 'food':
        if not original_settings['random_food_seasons']:
            original_mean_food_num = original_settings['food_num']
        else:
            original_mean_food_num = (settings['rand_food_season_limits'][0] + settings['rand_food_season_limits'][1]) / 2


    lowest_food_num = original_mean_food_num * (pipeline_settings['lowest_food_percent'] / 100.0)
    if lowest_food_num < 1:
        lowest_food_num = 1
    highest_food_num = original_mean_food_num * (pipeline_settings['highest_food_percent'] / 100.0)


    resolution = pipeline_settings['resolution']

    food_num_arr = l
    # Append food_num of original simulation if not already in list
    if not original_mean_food_num in food_num_arr:
        food_num_arr = np.append(food_num_arr, original_mean_food_num)
        food_num_arr = np.sort(food_num_arr)

    if pipeline_settings['parallelize_run_repeats']:
        ray.init(num_cpus=pipeline_settings['cores']) #, ignore_reinit_error=True
        ray_funcs = [run_repeat_remote.remote(food_num, settings, pipeline_settings, food_num_arr, original_mean_food_num) for food_num in food_num_arr]

        ray.get(ray_funcs)
        ray.shutdown()
    else:
        [run_repeat(food_num, settings, pipeline_settings, food_num_arr, original_mean_food_num) for food_num in food_num_arr]
    # run_repeat(20, settings, pipeline_settings)

@ray.remote
def run_repeat_remote(num_foods, settings, pipeline_settings, food_num_arr, original_mean_food_num):

    if pipeline_settings['varying_parameter'] == 'time_steps':
        settings['TimeSteps'] = num_foods
        # Activate saving of energies and velocities during life time for simulation with similar varying param as
        # original simulation and for largest varying param
        if num_foods == original_mean_food_num or num_foods == np.max(food_num_arr):
            settings['save_energies_velocities_last_gen'] = True
        print(num_foods)
    elif pipeline_settings['varying_parameter'] == 'food':
        settings['food_num'] = num_foods

    if pipeline_settings['varying_parameter'] == 'food':
        settings['dynamic_range_pipeline_save_name'] = '{}dynamic_range_run_foods_{}'.format(pipeline_settings['add_save_file_name'], num_foods)
    elif pipeline_settings['varying_parameter'] == 'time_steps':
        settings['dynamic_range_pipeline_save_name'] = '{}dynamic_range_run_time_step_{}'.format(pipeline_settings['add_save_file_name'], num_foods)
    Iterations = pipeline_settings['num_repeats']
    train.run(settings, Iterations)

# Exact copy of run_repeat_remote but without ray.remote decorator
def run_repeat(num_foods, settings, pipeline_settings, food_num_arr, original_mean_food_num):

    if pipeline_settings['varying_parameter'] == 'time_steps':
        settings['TimeSteps'] = num_foods
        # Activate saving of energies and velocities during life time for simulation with similar varying param as
        # original simulation and for largest varying param
        if num_foods == original_mean_food_num or num_foods == np.max(food_num_arr):
            settings['save_energies_velocities_last_gen'] = True
        print(num_foods)
    elif pipeline_settings['varying_parameter'] == 'food':
        settings['food_num'] = num_foods

    if pipeline_settings['varying_parameter'] == 'food':
        settings['dynamic_range_pipeline_save_name'] = '{}dynamic_range_run_foods_{}'.format(pipeline_settings['add_save_file_name'], num_foods)
    elif pipeline_settings['varying_parameter'] == 'time_steps':
        settings['dynamic_range_pipeline_save_name'] = '{}dynamic_range_run_time_step_{}'.format(pipeline_settings['add_save_file_name'], num_foods)
    Iterations = pipeline_settings['num_repeats']
    train.run(settings, Iterations)

if __name__=='__main__':
    '''
    BETTER NAME: FOOD or TIME STEP DENSITY RESPONSE CURVE
    This module explores the dynamic range of random food simulations: 
    It expects a file with with random food season parameter active
    It then takes the last generation of that simulation and puts it into different environments with fixed amount of 
    foods. There the organisms do not evolve but the experiment is repeated from scratch a given amount of times, which
    is defined by "num_repeats" to get statistically meaningful results.
    Cores should be about equal to the resolution, which should also be int
    '''

    pipeline_settings = {}
    pipeline_settings['varying_parameter'] = 'time_steps'  # 'food'
    pipeline_settings['cores'] = 58
    pipeline_settings['num_repeats'] = 3
    if pipeline_settings['varying_parameter'] == 'food':
        pipeline_settings['lowest_food_percent'] = 1
        pipeline_settings['highest_food_percent'] = 1000
    elif pipeline_settings['varying_parameter'] == 'time_steps':
        pipeline_settings['lowest_food_percent'] = 1
        pipeline_settings['highest_food_percent'] = 2500
    pipeline_settings['resolution'] = 40
    # !!!!!!!! add_save_file_name has to be unique each run and must not be a substring of previous run !!!!!!!!!
    # !!!!!!!! otherwise runs are indistringuishible !!!!!!!!!
    pipeline_settings['add_save_file_name'] = 'res_40_3_repeats_gen_4000' #'resulotion_80_hugeres_3_repeats_gen_100' # 'resulotion_80_hugeres_3_repeats_last_gen'
    # list of repeats, that should be animated, keep in mind, that this Creates an animation for each REPEAT!
    # If no animations, just emtpy list, if an animation should be created f.e. [0]
    pipeline_settings['animation_for_repeats'] = []
    # This loads last / highest generation from trained simulation
    pipeline_settings['load_last_generation'] = False
    # Otherwise specify generation, that shall be loaded, make sure thsi generation exists in all loaded simulations:
    pipeline_settings['load_generation'] = 4000
    # The following command allows to only plot a certain number of simulations in each parallel simulations folder
    # If all simulations in those folders shall be plotted, set to None
    pipeline_settings['only_plot_certain_num_of_simulations'] = None
    # The following settings define the level of parallelization. Use 'parallelize_run_repeats' for low level
    # parallelization when plotting few simulations. use high level parallelization with 'parallelize_each_sim' when
    # plotting many simulations. Both does not work at the same time. 'parallelize_each_sim' particularly recommended
    # when varying time steps
    pipeline_settings['parallelize_each_sim'] = True
    pipeline_settings['parallelize_run_repeats'] = False

    # Specific memory usage per parallel task has to be specified in dynamic_pipeline_one_sim_remote_memory
    # only works for pipeline_settings['parallelize_each_sim'] = True
    pipeline_settings['specify_memory_usage'] = True

    pipeline_settings['compress_save_isings'] = True
    # folder_names = ['sim-20201022-184145_parallel_TEST_repeated']
    # folder_names = ['sim-20201022-190553_parallel_b1_normal_seas_g4000_t2000', 'sim-20201022-190615_parallel_b10_normal_seas_g4000_t2000']#, 'sim-20201105-202455_parallel_b1_random_ts_2000_lim_100_3900', 'sim-20201105-202517_parallel_b10_random_ts_2000_lim_100_3900']
    # folder_names = ['sim-20201026-224639_parallel_b1_fixed_4000ts_', 'sim-20201026-224709_parallel_b10_fixed_4000ts_', 'sim-20201022-190553_parallel_b1_normal_seas_g4000_t2000', 'sim-20201022-190615_parallel_b10_normal_seas_g4000_t2000', 'sim-20201026-224655_parallel_b1_random_100-7900ts_', 'sim-20201026-224722_parallel_b10_random_100-7900ts_', 'sim-20201105-202455_parallel_b1_random_ts_2000_lim_100_3900', 'sim-20201105-202517_parallel_b10_random_ts_2000_lim_100_3900']
    folder_names = ['sim-20210206-122918_parallel_b1_normal_run_g4000_t2000_54_sims']#, 'sim-20201119-190204_parallel_b10_normal_run_g4000_t2000_54_sims']
    dynamic_pipeline_all_sims(folder_names, pipeline_settings)
