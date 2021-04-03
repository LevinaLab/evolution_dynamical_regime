from helper_functions.automatic_plot_helper import load_isings_from_list
from helper_functions.automatic_plot_helper import load_settings

import itertools
import numpy as np
from sklearn import manifold
import matplotlib.pyplot as plt
from os import path, makedirs
import matplotlib.colors as colors
import networkx as nx



# !!!! These functions are also used in heat capacity calculations !!!!


def normal_graph(sim_name, gen, ising_num):
    '''
    This module plots a tsne- representation of the fitness landscape of an ising-network
    '''
    settings = load_settings(sim_name)
    I = load_ising(sim_name, gen, ising_num)
    sensor_vals = I.s[0:(settings['nSensors'])]
    s_list, s_list_sensors = all_states(I, settings, sensor_vals)
    energies = calculate_energies(I, settings, s_list_sensors)
    h_graph = create_hamming_graph(s_list)
    plot_graph(h_graph, energies, sim_name)
    # s_tsne = calc_tsne(s_list)
    # plot_tsne(s_tsne, energies, sim_name)


def graph_for_network_without_sensors(sim_name, gen, ising_num, only_1_in_J=True):
    settings = load_settings(sim_name)
    I = load_ising(sim_name, gen, ising_num)
    if only_1_in_J:
        I.J = convert_non_zero_to_1(I.J)
    s_list_12 = all_states_for_s_without_sensors(I)
    energies = calculate_energies(I, settings, s_list_12)
    h_graph = create_hamming_graph(s_list_12)
    plot_graph(h_graph, energies, sim_name, save_suffix='_no_sensors_1_J')


def tsne_without_sensors(sim_name, gen, ising_num, only_1_in_J=True):
    settings = load_settings(sim_name)
    I = load_ising(sim_name, gen, ising_num)
    if only_1_in_J:
        I.J = convert_non_zero_to_1(I.J)
    s_list_12 = all_states_for_s_without_sensors(I)
    energies = calculate_energies(I, settings, s_list_12)
    s_tsne = calc_tsne(s_list_12)
    plot_tsne(s_tsne, energies, sim_name)




def convert_to_1(x):
    if x != 0:
        return 1.0
    else:
        return 0.0
convert_non_zero_to_1 = np.vectorize(convert_to_1)


def load_ising(sim_name, gen, ising_num):
    isings = load_isings_from_list(sim_name, iter_list=[gen], wait_for_memory=False)[0]
    I = isings[ising_num]
    return I





def calculate_energies(I, settings, s_list):

    energies = []
    h = I.h
    J = I.J
    energies = [calc_energy(s, h, J) for s in s_list]

    return energies


def plot_graph(h_graph, energies, sim_name, save_suffix=''):

    cmap = plt.get_cmap('jet')
    energies = normalize_energy_values_positive(energies)
    norm = colors.LogNorm(vmin=min(energies), vmax=max(energies))
    energy_colors = list(map(lambda x: cmap(norm(x)), energies))

    # nx.draw_kamada_kawai(h_graph, node_size=3, with_labels=False, width=0.1, style='dotted', node_color=energy_colors)
    # nx.draw_spring(h_graph, node_size=3, with_labels=False, width=0.2, style='dotted')
    nx.draw_networkx(h_graph, node_size=1, with_labels=False, width=0.2)

    save_folder = 'save/{}/figs/energy_landscape/'.format(sim_name)
    if not path.exists(save_folder):
        makedirs(save_folder)
    save_name = 'energy_landscape_graph{}.png'.format(save_suffix)
    plt.savefig(save_folder+save_name, bbox_inches='tight', dpi=300)
    plt.show()


def create_hamming_graph(s_list):
    h_graph = nx.MultiGraph()
    h_graph.add_nodes_from(s_list)
    h_tuples = create_hamming_tuples(s_list)
    h_graph.add_edges_from(h_tuples)
    return h_graph


def create_hamming_tuples(s_list):
    hamming_tuples = []
    for i, s1 in enumerate(s_list):
        for j in range(i + 1, len(s_list)):
            s2 = s_list[j]
            if is_hamming_1(s1, s2):
                hamming_tuples.append((s1, s2))
    return hamming_tuples



def is_hamming_1(s1, s2):
    return 1 == len(s1) - np.sum(np.array(s1)==np.array(s2))

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
    # plt.scatter(s_tsne[:, 0], s_tsne[:, 1], c=energy_colors, s=2, alpha=1)
    plt.scatter(s_tsne[:, 0], s_tsne[:, 1], c=energy_colors, s=0.05, alpha=1)

    save_folder = 'save/{}/figs/ising_tsne/'.format(sim_name)
    if not path.exists(save_folder):
        makedirs(save_folder)
    save_name = 'ising_tsne_graph.png'
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

def all_states_for_s_without_sensors(I):
    '''
    For the version where we assume no sensors at all
    '''
    permutated_states = list(itertools.product([-1, 1], repeat=len(I.s)))
    return permutated_states

def all_states(I, settings, sensor_vals):
    # all combinations:
    permutated_states = list(itertools.product([-1, 1], repeat=len(I.s) - settings['nSensors']))
    permutated_states_with_sensors = [list(sensor_vals) + list(en) for en in permutated_states]
    return permutated_states, permutated_states_with_sensors


if __name__ == '__main__':

    sim_name = 'sim-20201119-202501-g_2_-t_5_-num_neurons_10_-noplt_-n_6_non_sensory_neurons_energy_landscape_graph'
    generation = 0
    ising_num = 0
    normal_graph(sim_name, generation, ising_num)
    # graph_for_network_without_sensors(sim_name, generation, ising_num)
    # tsne_without_sensors(sim_name, generation, ising_num)