import train
import copy
from helper_functions.automatic_plot_helper import load_settings
from helper_functions.automatic_plot_helper import all_sim_names_in_parallel_folder
from helper_functions.automatic_plot_helper import load_isings_from_list
import ray
import numpy as np

# TODO: !!!!This chooses mutate function!!!!!
# from mutate_genotype import mutate_genotype_main
# from mutate_genotype_from_unconnected import mutate_genotype_main
from helper_functions.mutate_genotype_all_edges import mutate_genotype_main

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

            ray_funcs = [dynamic_pipeline_one_sim_remote_memory.remote(sim_name, pipeline_settings)for sim_name in all_sim_names]

            ray.get(ray_funcs)
            ray.shutdown()


# Exact copy of run_repeat_remote but with specific memory usage. Memory usage par task!!
@ray.remote(memory=1500 * 1024 * 1024)
def dynamic_pipeline_one_sim_remote_memory(sim_name, pipeline_settings):

    original_settings = load_settings(sim_name)
    settings = create_settings_for_repeat(original_settings, sim_name, pipeline_settings)
    run_all_repeats(settings, original_settings, pipeline_settings, sim_name)

# Exact copy of run_repeat_remote but without ray.remote decorator
def dynamic_pipeline_one_sim(sim_name, pipeline_settings):

    original_settings = load_settings(sim_name)
    settings = create_settings_for_repeat(original_settings, sim_name, pipeline_settings)
    run_all_repeats(settings, original_settings, pipeline_settings, sim_name)


def create_settings_for_repeat(settings, sim_name, pipeline_settings):
    # settings['TimeSteps'] = 5
    if pipeline_settings['varying_parameter'] == 'time_steps':
        settings['random_time_steps'] = False
    elif pipeline_settings['varying_parameter'] == 'food':
        settings['random_food_seasons'] = False

    settings = copy.deepcopy(settings)

    complete_sim_folder = sim_name
    settings['loadfile'] = complete_sim_folder


    settings['LoadIsings'] = False
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
    settings['beta_linspace'] = None

    settings['compress_save_isings'] = pipeline_settings['compress_save_isings']

    return settings


def run_all_repeats(settings, original_settings, pipeline_settings, sim_name):

    resolution = pipeline_settings['resolution']
    isings_orig = load_isings_from_list(sim_name, [pipeline_settings['load_generation']],
                                        decompress=pipeline_settings['decompress_loaded_ising'])[0]


    gen_perturb_arr = np.linspace(pipeline_settings['lowest_genetic_perturbation'], pipeline_settings['largest_genetic_perturbation'],
                pipeline_settings['resolution']).astype(int)  # astype int kann theoretisch auch weg

    if pipeline_settings['parallelize_run_repeats']:
        ray.init(num_cpus=pipeline_settings['cores']) #, ignore_reinit_error=True
        ray_funcs = [run_repeat_remote.remote(gen_perturb, isings_orig, settings, pipeline_settings)
                     for gen_perturb in gen_perturb_arr]

        ray.get(ray_funcs)
        ray.shutdown()
    else:

        [run_repeat(gen_perturb, isings_orig, settings, pipeline_settings)
         for gen_perturb in gen_perturb_arr]


    # run_repeat(20, settings, pipeline_settings)

@ray.remote
def run_repeat_remote(gene_perturb, isings_orig, settings, pipeline_settings):

    settings['save_energies_velocities_last_gen'] = pipeline_settings['save_energies_velocities']
    print('Genetic perturbation with factor {}'.format(gene_perturb))

    settings['dynamic_range_pipeline_save_name'] = '{}genotype_phenotype_mapping_{}'.format(pipeline_settings['add_save_file_name'], gene_perturb)

    perturbed_isings = mutate_genotype_main(isings_orig, gene_perturb, pipeline_settings['genetic_perturbation_constant']
                                            , pipeline_settings['number_of_edges_to_perturb'], settings)

    settings['set_isings'] = perturbed_isings
    Iterations = pipeline_settings['num_repeats']
    train.run(settings, Iterations)

# Exact copy of run_repeat_remote but without ray.remote decorator
def run_repeat(gene_perturb, isings_orig, settings, pipeline_settings):

    settings['save_energies_velocities_last_gen'] = pipeline_settings['save_energies_velocities']
    print('Genetic perturbation with factor {}'.format(gene_perturb))

    settings['dynamic_range_pipeline_save_name'] = '{}genotype_phenotype_mapping_{}'.format(pipeline_settings['add_save_file_name'], gene_perturb)

    perturbed_isings = mutate_genotype_main(isings_orig, gene_perturb, pipeline_settings['genetic_perturbation_constant']
                                            , pipeline_settings['number_of_edges_to_perturb'], settings)

    settings['set_isings'] = perturbed_isings
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
    pipeline_settings['cores'] = 60
    pipeline_settings['num_repeats'] = 10 # (!!!!!)
    pipeline_settings['lowest_genetic_perturbation'] = 0
    pipeline_settings['largest_genetic_perturbation'] = 300

    pipeline_settings['genetic_perturbation_constant'] = 0.005 #0.005
    pipeline_settings['number_of_edges_to_perturb'] = 10 #5


    pipeline_settings['resolution'] = 80
    # !!!!!!!! add_save_file_name has to be unique each run and must not be a substring of previous run !!!!!!!!!
    # !!!!!!!! otherwise runs are indistringuishible !!!!!!!!!
    pipeline_settings['add_save_file_name'] = 'big_run_10_runs_all_connectable_mutate_ALL_edges_0-005_only_1_repeat' #'resulotion_80_hugeres_3_repeats_gen_100' # 'resulotion_80_hugeres_3_repeats_last_gen'
    # list of repeats, that should be animated, keep in mind, that this Creates an animation for each REPEAT!
    # If no animations, just emtpy list, if an animation should be created f.e. [0]
    pipeline_settings['animation_for_repeats'] = []
    # This loads last / highest generation from trained simulation
    pipeline_settings['load_last_generation'] = False
    # Otherwise specify generation, that shall be loaded, make sure thsi generation exists in all loaded simulations:
    pipeline_settings['load_generation'] = 4000
    pipeline_settings['decompress_loaded_ising'] = True
    # The following command allows to only plot a certain number of simulations in each parallel simulations folder
    # If all simulations in those folders shall be plotted, set to None
    pipeline_settings['only_plot_certain_num_of_simulations'] = None
    # The following settings define the level of parallelization. Use 'parallelize_run_repeats' for low level
    # parallelization when plotting few simulations. use high level parallelization with 'parallelize_each_sim' when
    # plotting many simulations. Both does not work at the same time. 'parallelize_each_sim' particularly recommended
    # when varying time steps
    pipeline_settings['parallelize_each_sim'] = False
    pipeline_settings['parallelize_run_repeats'] = True

    pipeline_settings['save_energies_velocities'] = False


    # Specific memory usage per parallel task has to be specified in dynamic_pipeline_one_sim_remote_memory
    # only works for pipeline_settings['parallelize_each_sim'] = True


    pipeline_settings['compress_save_isings'] = True

    # folder_names = ['sim-20210216-001754_parallel_-b_10_-g_10_-t_200_-noplt_-n_test_genotype_phenotype2', 'sim-20210215-235355_parallel_-g_10_-t_200_-noplt_-n_test_genotype_phenotype2']#, 'sim-20201119-190204_parallel_b10_normal_run_g4000_t2000_54_sims']
    # folder_names = ['sim-20201210-200605_parallel_b1_genotype_phenotype_test', 'sim-20201210-200613_parallel_b10_genotype_phenotype_test']
    folder_names = ['sim-20210206-122918_parallel_b1_normal_run_g4000_t2000_54_sims', 'sim-20201119-190204_parallel_b10_normal_run_g4000_t2000_54_sims']
    #   TODO: !!!!!! CHECK WHETHER YOU IMPORTED CORRECT MUTATE FUNTION!!!!!!
    # folder_names = ['sim-20201210-200605_parallel_b1_dynamic_range_c_20_g4000_t2000_10_sims', 'sim-20201210-200613_parallel_b10_dynamic_range_c_20_g4000_t2000_10_sims']
    dynamic_pipeline_all_sims(folder_names, pipeline_settings)
