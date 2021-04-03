#!/usr/bin/env python



import ray
from train import run
from train import create_settings
import time
import os
import numpy as np
from helper_functions.automatic_plot_helper import detect_all_isings
from helper_functions.automatic_plot_helper import load_isings_from_list


def test_hyperparams(iterations):
    #ray.init()
    hyperparam_configs = [generate_hyperparams() for _ in range(20)]

    results = []
    for hyperparams in hyperparam_configs:
        results.append(run_hyperparams(iterations, hyperparams))

    accuracies = results

    save_dir = 'save/hyperparam_tuning/'.format(time.strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_dir + 'hyper_param_results_{}.csv', 'w') as f:
        for tup in accuracies:
            f.write('{}: {}\n'.format(tup[0], tup[1]))
    f.close()


def main_hyperparams(iterations, cores):
    ray.init()
    hyperparam_configs = [generate_hyperparams() for _ in range(cores)]


    settings, iterations_not_used = create_settings()
    results = [run_hyperparams.remote(iterations, hyperparams, settings) for hyperparams in hyperparam_configs]

    accuracies = ray.get(results)

    save_dir = 'save/hyperparam_tuning/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_dir + 'hyper_param_results_{}.csv'.format(time.strftime("%Y%m%d-%H%M%S")), 'w') as f:
        for tup in accuracies:
            f.write('{}: {}\n'.format(tup[1], tup[0]))
    f.close()


@ray.remote
def run_hyperparams(iterations, settings_tune, settings):


    #settings['continuous_species'] = settings_tune['continuous_species']
    #Constants for delta formula c_top, c_weight, c_beta
    #settings['shared_fitness_constants'] = settings_tune['shared_fitness_constants']
    settings['delta_threshold_speciation'] = settings_tune['delta_threshold_speciation']
    #settings['add_save_name'] = '_HYPERPARAM_con_species_{}_delta_{}'.format(settings_tune['continuous_species'], settings_tune['delta_threshold_speciation'])
    settings['add_save_name'] = 'delta_{}'.format(settings_tune['delta_threshold_speciation'])
    sim_name = run(settings, iterations)
    last_ising = detect_all_isings(sim_name)[-1]
    isings_last_gen = load_isings_from_list(sim_name, [last_ising])[0]
    all_avg_energies = [I.avg_energy for I in isings_last_gen]
    avg_fitness_last_gen = np.average(all_avg_energies)
    settings_tune['sim_name'] = sim_name
    return settings_tune, avg_fitness_last_gen

def generate_hyperparams():

    # settings_tune = {'continuous_species': bool(np.random.randint(0, 1)),
    #                  'delta_threshold_speciation': 10 ** np.random.uniform(-1, 2),
    #                  'shared_fitness_constants': (10 ** np.random.uniform(-1, 1), 10 ** np.random.uniform(-1, 1), 10 ** np.random.uniform(-1, 1))
    #                  }

    settings_tune = {
                     'delta_threshold_speciation': 10 ** np.random.uniform(-1, 2),

                     }

    return settings_tune



# --- RUN ----------------------------------------------------------------------+

if __name__ == '__main__':
    '''
    Time steps and other settings can be adjusted just like in train using argparse
    '''
    cores = 20
    generations = 300
    main_hyperparams(generations, cores)

    #test_hyperparams(3)
    # settings, Iterations = create_settings()
    # t1 = time.time()
    # sim_name = run(settings, Iterations)
    # t2 = time.time()
    # print('total time:', t2-t1)
    # if settings['save_data'] and settings['plot_pipeline']:
    #     automatic_plotting.main(sim_name)




# --- END ----------------------------------------------------------------------+
