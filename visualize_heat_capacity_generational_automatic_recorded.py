#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from os import path, makedirs
from helper_functions.automatic_plot_helper import load_settings
import os



def main(sim_name, settings, generation_list, recorded):
    '''
    generation list can be set to None
    recorded is a boolean defining whether we want to visualize recorded heat capacity or dream heat capacity
    '''

    # TODO: make these scripts take these as params
    loadfile = sim_name
    folder = 'save/' + loadfile


    R = 10
    Nbetas = 102
    betas = 10 ** np.linspace(-1, 1, Nbetas)
    numAgents = settings['pop_size']
    size = settings['size']

    if generation_list is None:
        if recorded:
            generation_list = automatic_generation_generation_list(folder + '/C_recorded')
        else:
            generation_list = automatic_generation_generation_list(folder + '/C')
    iter_gen = generation_list

    C = np.zeros((R, numAgents, Nbetas, len(iter_gen)))


    print('Loading data...')
    for ii, iter in enumerate(iter_gen):
        #for bind in np.arange(0, 100):
        for bind in np.arange(1, 100):
            if recorded:
                #  Depending on whether we are dealing with recorded or dream heat capacity
                filename = folder + '/C_recorded/C_' + str(iter) + '/C-size_' + str(size) + '-Nbetas_' + \
                           str(Nbetas) + '-bind_' + str(bind) + '.npy'
            else:
                filename = folder + '/C/C_' + str(iter) + '/C-size_' + str(size) + '-Nbetas_' + \
                           str(Nbetas) + '-bind_' + str(bind) + '.npy'
            C[:, :, bind, ii] = np.load(filename)
    print('Done.')

    plt.rc('text', usetex=True)
    font = {'family': 'serif', 'size': 28, 'serif': ['computer modern roman']}
    plt.rc('font', **font)
    plt.rc('legend', **{'fontsize': 20})

    b = 0.8
    alpha = 0.3

    print('Generating figures...')
    for ii, iter in enumerate(iter_gen):

        fig, ax = plt.subplots(1, 1, figsize=(11, 10), sharex=True)
        fig.text(0.51, 0.035, r'$\beta$', ha='center', fontsize=28)
        fig.text(0.005, 0.5, r'$C/N$', va='center', rotation='vertical', fontsize=28)
        title = 'Specific Heat of Foraging Community\n Generation: ' + str(iter)
        fig.suptitle(title)

        # CHANGE THIS TO CUSTOMIZE HEIGHT OF PLOT

        upperbound = 0.4
        upperbound = 1

        label = iter

        for numOrg in range(numAgents):
            c = np.dot(np.random.random(), [1, 1, 1])
            ax.scatter(betas, np.mean(C[:, numOrg, :, ii], axis=0),
                       color=[0, 0, 0], s=30, alpha=alpha, marker='x', label=label)

        xticks = [0.1, 0.5, 1, 2, 4, 10]
        ax.set_xscale("log", nonposx='clip')
        ax.set_xticks(xticks)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

        # This is bounding plot!!
        # plt.axis([0.1, 10, 0, upperbound])


        if recorded:
            savefolder = folder + '/figs/C_recorded/'
        else:
            savefolder = folder + '/figs/C/'
        savefilename = savefolder + 'C-size_' + str(size) + '-Nbetas_' + \
                       str(Nbetas) + '-gen_' + str(iter) + '_experimental.png'
        if not path.exists(savefolder):
            makedirs(savefolder)

        plt.savefig(savefilename, bbox_inches='tight')
        plt.close()
        # plt.clf()
        savemsg = 'Saving ' + savefilename
        print(savemsg)
        # plt.show()
        # plt.pause(0.1)

def automatic_generation_generation_list(C_folder):
    C_gen_folders = [f.path for f in os.scandir(C_folder) if f.is_dir()]
    generation_list = get_generations(C_gen_folders)
    return generation_list

def get_generations(C_gen_folders):
    generation_list = []
    for C_gen_folder in C_gen_folders:
        if RepresentsInt(C_gen_folder.split('_')[-1]) is True:
            generation_list.append(C_gen_folder.split('_')[-1])
    return generation_list

def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

if __name__ == '__main__':
    sim_name = 'sim-20200703-141831-g_2_-t_2000_-rec_c_1_-c_9_-n_sensors_remain_fixed_no_therm_to_equilib'
    generation_list = [0, 4000]
    settings = load_settings(sim_name)
    main(sim_name, settings, None, recorded=True)
