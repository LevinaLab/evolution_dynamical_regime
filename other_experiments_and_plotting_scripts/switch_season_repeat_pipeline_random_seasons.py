from multiprocessing import Pool
import argparse
import train
import copy
from helper_functions.automatic_plot_helper import detect_all_isings
from helper_functions.automatic_plot_helper import load_isings_from_list
import time
import ray
from other_experiments_and_plotting_scripts.switch_season_repeat_plotting import plot_pipeline
import pickle
from helper_functions.run_combi import RunCombi
import numpy as np
processes = ('-g 5 -t 200', '-g 20 -t 200')


def main(num_repeats, same_repeats, food_summer, food_winter, folder_add, only_fittest, cores):

    folder_add = 'num_rep_{}_same_rep_{}_f_sum_{}_f_win{}_{}'.format(num_repeats, same_repeats, food_summer, food_winter
                                                                     , folder_add)

    run_combis, first_subfolder = run_all_combinations(num_repeats, same_repeats, food_summer, food_winter,
                                                       folder_add, only_fittest, cores)
    plot_pipeline(run_combis, first_subfolder, 'avg_energy')


def run_all_combinations(num_repeats, same_repeats, food_summer, food_winter, folder_add, only_fittest, cores):
    '''
    main function for running simulations
    num_repeats: the number of times last generation is repeated
    same_repeats: Number of times the same simulation is run
    '''
    # TODO Parallelize all combinations!

    #same_repeats = 4  # Number of times the same simulation is run

    settings, Iterations = train.create_settings()
    #num_repeats = 5  # 200 # num repeats: the number of times last generation is repeated
    first_subfolder = 'switch_seasons_{}_{}'.format(time.strftime("%Y%m%d-%H%M%S"), folder_add)
    run_combis = make_combinations(settings, same_repeats, food_summer, food_winter)

    #ray.init(num_cpus=19 ,memory=9*10**9, object_store_memory=7*10**9)
    ray.init(num_cpus=cores)

    ray_funcs = [run_one_combination.remote(run_combi, first_subfolder, Iterations, num_repeats, food_summer, food_winter, only_fittest)
                 for run_combi in run_combis]
    # ray_funcs = [run_one_combination(run_combi, first_subfolder, Iterations, num_repeats, food_summer, food_winter, only_fittest)
    #              for run_combi in run_combis]
    ray.get(ray_funcs)

    save_run_combis(run_combis, first_subfolder)
    return run_combis, first_subfolder


def save_run_combis(run_combis, first_subfolder):
    savefolder = 'save/{}/run_combis.pickle'.format(first_subfolder)
    pickle_out = open(savefolder, 'wb')
    pickle.dump(run_combis, pickle_out)
    pickle_out.close()



def make_combinations(settings, same_repeats, food_summer, food_winter):
    '''
    creates all combinations of runs
    same_repeats: int - Defines how many times the simulation with same parameter is "repeated"
    (for statistical significance)
    '''
    run_combis = []
    for food in [food_summer, food_winter]:
        for beta in [1, 10]:
            for repeat in range(same_repeats):
                run_combis.append(RunCombi(settings, food, beta, repeat, same_repeats, food_summer, food_winter))
    return run_combis


@ray.remote
def run_one_combination(run_combi, first_subfolder, Iterations, num_repeats, food_summer, food_winter, only_fittest):
    second_subfolder = run_combi.subfolder
    save_subfolder = '{}/{}'.format(first_subfolder, second_subfolder)
    settings = run_combi.settings
    settings['commands_in_folder_name'] = False

    run_sim_and_create_repeats(save_subfolder, settings, Iterations, num_repeats, food_summer, food_winter, only_fittest)


def run_sim_and_create_repeats(save_subfolder, settings, Iterations, num_repeats, food_summer, food_winter, only_fittest):
    settings['save_subfolder'] = save_subfolder
    settings['random_food_seasons'] = True
    sim_name = train.run(settings, Iterations)
    create_repeats(sim_name, save_subfolder, settings, num_repeats, food_summer, food_winter, only_fittest)


def only_fittest_individuals(sim_name, save_subfolder):
    '''
    This function takes last generation of run, selects the fittest individuals and saves them as additional last generation
    (This generation thus has fewer individuals)
    '''

    complete_sim_folder = '{}/{}'.format(save_subfolder, sim_name)
    last_generation_num = detect_all_isings(complete_sim_folder)[-1]
    last_isings = load_isings_from_list(complete_sim_folder, [last_generation_num])[0]
    fitness_list = [I.avg_energy for I in last_isings]
    # Sort isings according to fitness:
    # sorted_isings = [I for _, I in np.sort(zip(fitness_list, last_isings))]
    # sorted_zip = zip(fitness_list, last_isings)
    # sorted_zip = sorted(sorted_zip)
    # sorted_isings = [I for f, I in sorted_zip]
    # Choose first third of the fittes isings:
    sorted_isings = sort_list_acc_to_other_list(last_isings, fitness_list)
    fittest_isings = sorted_isings[:int(len(sorted_isings)/3)]
    pop_size = len(fittest_isings)
    #save/switch_seasons_20200524-030645_num_rep_200_same_rep_1_f_sum_100_f_win10_default_food/b10_summer_0/sim-20200524-030647/isings
    ising_save_folder = 'save/{}/isings/gen[{}]-isings.pickle'.format(complete_sim_folder, last_generation_num + 1)
    pickle_out = open(ising_save_folder, 'wb')
    pickle.dump(fittest_isings, pickle_out)
    pickle_out.close()
    return pop_size

def sort_list_acc_to_other_list(main_list, other_list):
    '''
    Sorts main_list according to sorted other list in descending order
    '''
    sorted_index = np.argsort(other_list)
    sorted_index = sorted_index[::-1]
    main_list_sorted = [main_list[i] for i in sorted_index]
    return main_list_sorted

def create_repeats(sim_name, save_subfolder, settings, num_repeats, food_summer, food_winter, only_fittest):
    settings = copy.deepcopy(settings)

    if only_fittest:
        pop_size = only_fittest_individuals(sim_name, save_subfolder)
        settings['pop_size'] = pop_size

    complete_sim_folder = '{}/{}'.format(save_subfolder, sim_name)
    settings['loadfile'] = complete_sim_folder

    settings['iter'] = detect_all_isings(complete_sim_folder)[-1]
    settings['LoadIsings'] = True
    settings['switch_off_evolution'] = True
    settings['save_data'] = False
    settings['switch_seasons_repeat_pipeline'] = True
    # Animations:
    settings['plot_generations'] = [1]



    #  Number of repeats
    # Iterations = 200
    Iterations = num_repeats

    settings['repeat_pipeline_switched_boo'] = False
    train.run(settings, Iterations)

    #  switch seasons
    if settings['food_num'] == food_summer:
        settings['food_num'] = food_winter
    elif settings['food_num'] == food_winter:
        settings['food_num'] = food_summer


    settings['repeat_pipeline_switched_boo'] = True
    settings['random_food_seasons'] = False
    train.run(settings, Iterations)


def create_repeats_parallel(sim_name, settings):
    settings['loadfile'] = sim_name
    settings['iter'] = detect_all_isings(sim_name)[-1]
    pool = Pool(processes=2)
    pool.map(_run_process, processes)


def _run_process(process, settings):
    #os.system('python3 train {}'.format(process))

    train.run(settings)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', dest='commands', help='''Commands that are passed to evolution simulation''')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    '''
    input arguments of train.py can be passed just as usual. This way f.e. the number of time steps as well as number of
    generations in first simulation can be adjusted
    recommended:
    -g 2000 -t 2000 -dream_c 0 -nat_c 0 -ref 0 -rec_c 0 -noplt
    
    The parameters below specify the pipeline specific parameters. The following parameters are recommented:
    folder_add = 'test_run'
    num_repeats = 200
    same_repeats = 4 (four times as many cores are required, so in this case 16)
    food_summer = 100
    food_winter = 10
    
    only_fittest = False
    
    '''
    folder_add = 'test'
    num_repeats = 20#200
    same_repeats = 4
    food_summer = 100
    food_winter = 4

    cores = 5
    only_fittest = False

    main(num_repeats, same_repeats, food_summer, food_winter, folder_add, only_fittest, cores)
