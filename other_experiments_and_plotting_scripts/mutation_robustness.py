from helper_functions.automatic_plot_helper import load_isings_from_list
from helper_functions.automatic_plot_helper import detect_all_isings
from helper_functions.automatic_plot_helper import load_settings
import numpy as np
import copy
from embodied_ising import EvolutionLearning
from embodied_ising import food
import matplotlib.pyplot as plt
import time
import os
import pickle


def main(sim_name, animate_reps=[], num_evaluate_different_inds=4, repeat_mutations=20, load_generation=False, pop_size=50,
         time_steps=2000, attr_name='avg_energy', load_results_boo=False):


    save_dir = 'save/mutation_robustness/{}/'.format(sim_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_name = '{}ts_{}inds_{}repeats_{}'.format(time_steps, num_evaluate_different_inds, repeat_mutations, attr_name)
    save_sim_name = save_name+'.pickle'
    save_plot_name = save_name+str(time.strftime("%Y%m%d-%H%M%S"))+'.png'

    if not load_results_boo:
        all_inds_evaluated = test_robustness(sim_name, save_dir, animate_reps, num_evaluate_different_inds, repeat_mutations, load_generation, pop_size,
                                             time_steps, attr_name)
        save_results(all_inds_evaluated, save_dir, save_sim_name)
    else:
        all_inds_evaluated = load_results(save_dir, save_sim_name)

    plot(all_inds_evaluated, attr_name, sim_name, save_dir, save_plot_name)


def save_results(all_inds_evaluated, save_dir, save_name):
    pickle_out = open(save_dir+save_name, 'wb')
    pickle.dump(all_inds_evaluated, pickle_out)
    pickle_out.close()


def load_results(save_dir, save_name):
    file = open(save_dir+save_name, 'rb')
    all_inds_evaluated = pickle.load(file)
    file.close()
    return all_inds_evaluated


def test_robustness(sim_name, save_dir, animate_reps, num_evaluate_different_inds=4, repeat_mutations=20, load_generation=False, pop_size=50,
                    time_steps=2000, attr_name='avg_energy'):
    if not load_generation:
        load_generation = detect_all_isings(sim_name)[-1]
    settings = load_settings(sim_name)
    settings['pop_size'] = pop_size
    settings['TimeSteps'] = time_steps
    settings['save_data'] = False
    settings['plot_pipeline'] = False
    settings['fading_traces_animation'] = True

    # Das ist nur zu test-pfuschzwecken da kann irgendwann weg:_
    settings['random_time_steps'] = False
    foods = []
    for i in range(0, settings['food_num']):
        foods.append(food(settings))

    isings = load_isings_from_list(sim_name, [load_generation], wait_for_memory=False)[0]

    all_inds_evaluated = []
    for ind in range(num_evaluate_different_inds):
        # Only take first 20 inds, those are the not recently mutated
        chosen_I = copy.deepcopy(isings[ind])
        one_agend_all_repeats_list = []
        for rep in range(repeat_mutations):
            if rep in animate_reps:
                settings['mutation_robustness_path'] = save_dir
                settings['plot_generations'] = [0]
            else:
                settings['mutation_robustness_path'] = ''
                settings['plot_generations'] = []

            chosen_I_copy = copy.deepcopy(chosen_I)
            # Mutation!
            if not rep == 0:
                network_edge_positions = np.argwhere(chosen_I.maskJ==True)
                rand_pos = np.random.randint(0, len(network_edge_positions))
                mutate_edge_pos = network_edge_positions[rand_pos]

                # Don't use chosen_I_copy if with each iteration there shouls be more mutations

                chosen_I_copy.J[mutate_edge_pos[0], mutate_edge_pos[1]] = np.random.uniform(-1, 1) * chosen_I_copy.max_weights
            clone_isings = []
            for i in range(pop_size):
                clone_isings.append(copy.deepcopy(chosen_I_copy))


            blub, evaluated_isings = EvolutionLearning(clone_isings, foods, settings, Iterations=1)

            one_agend_all_repeats_list.append(evaluated_isings)
        all_inds_evaluated.append(one_agend_all_repeats_list)

    return all_inds_evaluated

def plot(all_inds_evaluated, attr_name, sim_name, save_dir, save_name):
    plt.figure(figsize=(40, 10))
    for ind_num, one_agend_all_repeats_list in enumerate(all_inds_evaluated):
        colour = np.random.rand(3,)
        for num_simulation_repeat, isings_simulation_repeat in enumerate(one_agend_all_repeats_list):
            # PLotting parameters adjusted for 4 individuals
            attributes_to_plot = []
            for I in isings_simulation_repeat:
                exec('attributes_to_plot.append(I.{})'.format(attr_name))
            x = [float(num_simulation_repeat) - 0.3 + 0.2*ind_num + np.random.uniform(-0.075, 0.075) for i in range(len(attributes_to_plot))]
            plt.scatter(x, attributes_to_plot, c=colour, alpha=0.5)

    plt.savefig(save_dir+save_name, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    sim_name = 'Energies_Velocities_saved_during_2d_sim_random_time_steps_cut_off_animations/sim-20200604-235433-g_2000_-t_2000_-b_10_-dream_c_0_-nat_c_0_-ref_0_-rec_c_0_-n_energies_velocities_saved'
    #sim_name = 'Energies_Velocities_saved_during_2d_sim_random_time_steps_cut_off_animations/sim-20200604-235424-g_2000_-t_2000_-b_1_-dream_c_0_-nat_c_0_-ref_0_-rec_c_0_-n_energies_velocities_saved'
    num_evaluate_diff_inds = 4
    repeat_mutations = 5
    time_steps = 2000
    attr_name = 'avg_energy'
    load_results_boo = False
    animate_reps = [] # [0,1,2,3]
    main(sim_name, animate_reps, num_evaluate_different_inds=num_evaluate_diff_inds, repeat_mutations=repeat_mutations,
         time_steps=time_steps, attr_name=attr_name, load_results_boo=load_results_boo)
    # test_robustness(sim_name, num_evaluate_different_inds=num_evaluate_diff_inds, repeat_mutations=repeat_mutations,
    #                 time_steps=time_steps, attr_name=attr_name)
