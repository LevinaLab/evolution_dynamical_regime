from helper_functions.automatic_plot_helper import load_isings_from_list
from helper_functions.automatic_plot_helper import load_settings

import itertools
import numpy as np
from sklearn import manifold
import matplotlib.pyplot as plt
from os import path, makedirs
import matplotlib.colors as colors



# !!!! These functions are also used in heat capacity calculations !!!!

def main(sim_name, gen):
    '''
    This module plots a tsne- representation of the fitness landscape of an ising-network
    '''
    settings = load_settings(sim_name)
    I = load_ising(sim_name, gen)
    sensor_vals = I.s[0:(settings['nSensors'])]
    s_list, s_list_sensors = all_states(I, settings, sensor_vals)
    energies = calculate_energies(I, settings, s_list_sensors)
    s_tsne = calc_tsne(s_list)
    plot_tsne(s_tsne, energies, sim_name)


def load_ising(sim_name, gen):
    isings = load_isings_from_list(sim_name, iter_list=[gen], wait_for_memory=False)[0]
    I = isings[0]
    return I


def calculate_energies(I, settings, s_list):

    energies = []
    h = I.h
    J = I.J
    energies = [calc_energy(s, h, J) for s in s_list]

    return energies


def calc_tsne(s_list):
    # tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, perplexity=30, n_iter=2000)
    tsne = manifold.TSNE(n_components=2)
    s_arr = np.array(s_list)
    s_tsne = tsne.fit_transform(s_arr)
    return s_tsne

def plot_tsne(s_tsne, energies, sim_name):
    plt.figure(figsize=(12, 12))
    cmap = plt.get_cmap('jet')
    energies = normalize_energy_values_positive(energies)
    norm = colors.LogNorm(vmin=min(energies), vmax=max(energies))

    energy_colors = list(map(lambda x: cmap(norm(x)), energies))
    plt.scatter(s_tsne[:, 0], s_tsne[:, 1], c=energy_colors)

    save_folder = 'save/{}/figs/ising_tsne/'.format(sim_name)
    if not path.exists(save_folder):
        makedirs(save_folder)
    save_name = 'ising_tsne.png'
    plt.savefig(save_folder+save_name, bbox_inches='tight', dpi=300)
    plt.show()

def normalize_energy_values_positive(energies):
    '''
    Do this for norm function which is used for coloring
    all energy values are added with the minimal energy +1 such that all energy values are >0
    '''
    add = np.abs(min(energies)) + 1
    energies_positive = list(map(lambda x: x + add, energies))
    return energies_positive

def calc_energy(s, h, J):
    E = -(np.dot(s, h) + np.dot(np.dot(s, J), s))
    return E

# def calc_solution_space(I, )


def all_states(I, settings, sensor_vals):
    # all combinations:
    permutated_states = list(itertools.product([-1, 1], repeat=len(I.s) - settings['nSensors']))
    permutated_states_with_sensors = [list(sensor_vals) + list(en) for en in permutated_states]
    return permutated_states, permutated_states_with_sensors


if __name__ == '__main__':
    sim_name = 'sim-20201119-202501-g_2_-t_5_-num_neurons_10_-noplt_-n_6_non_sensory_neurons_energy_landscape_graph'
    generation = 0
    main(sim_name, generation)