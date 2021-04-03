#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
mpl.use('Agg') #For server use
import matplotlib.pyplot as plt


from matplotlib import colors
from matplotlib.lines import Line2D
import pickle
from os import makedirs, path
from helper_functions.automatic_plot_helper import detect_all_isings
from helper_functions.automatic_plot_helper import load_isings
import sys
'''
loadfiles = ['beta_experiment/beta-0-1/sim-20180512-105719',
             'beta_experiment/beta-1/sim-20180511-163319',
             'beta_experiment/beta-10/sim-20180512-105824']
'''


def main(loadfiles, plot_var, settings=None, isings_lists=None, autoLoad=True,
         sim_labels=[r'$\beta_i = 0.1$', r'$\beta_i = 1$', r'$\beta_i = 10$'], scatter=True, name_extension=''):

    '''
    Can either plot one or multiple simulations in a combined plot
    :param loadfile: save names of simulations; list of strings
    :param plot_var: isings attribute to ne plotted over generation
    :param isings_lists: list of isings list (one isings list for each simulation to be plotted)
    :param autoLoad: If previously plotted should npz file be loaded to speed up process?
    :param sim_labels: Labels of simulation in plot in case multi_sim = True
    '''

    if type(loadfiles) == str:
        loadfiles = [loadfiles]
    if type(isings_lists) == str:
        isings_lists == [isings_lists]

    #loadfiles = [loadfile]#loadfiles = ['sim-20191114-000009_server']
    #Boo shows whether there are multiple simulations in one plot
    multiple_sim = len(loadfiles) > 1

    if settings is None:
        energy_model = settings['energy_model']

    #autoLoad = True
    saveFigBool = True
    fixGen2000 = False

    new_order = [2, 0, 1]

    labels = sim_labels

    cmap = plt.get_cmap('seismic')
    norm = colors.Normalize(vmin=0, vmax=len(loadfiles))  # age/color mapping

    a = 0.15  # alpha

    ###########################
    FOODS = []
    for loadfile, isings_list in zip(loadfiles, isings_lists):
        iter_list = detect_all_isings(loadfile)  # iter_list = np.arange(0, 2000, 1)
        #settings = load_settings(loadfile)
        numAgents = len(isings_list[0]) # settings['pop_size']
        f = fitness(loadfile, iter_list, isings_list, numAgents, autoLoad, saveFigBool, plot_var)
        # FIX THE DOUBLE COUNTING PROBLEM
        if f.shape[0] > 2000 and fixGen2000:
            print('Fixing Double Counting at Gen 2000')
            f[2000, :] = f[2000, :] - f[1999, :]
        FOODS.append(f)

    # FIX THE DOUBLE COUNTING OF THE FITNESS


    plt.rc('text', usetex=True)
    font = {'family': 'serif', 'size': 28, 'serif': ['computer modern roman']}
    plt.rc('font', **font)
    plt.rc('legend', **{'fontsize': 20})

    fig, ax = plt.subplots(1, 1, figsize=(19, 10))
    fig.text(0.51, 0.035, r'Generation', ha='center', fontsize=20)
    # fig.text(0.07, 0.5, r'$Avg. Food Consumed$', va='center', rotation='vertical', fontsize=20)
    fig.text(0.07, 0.5, r'%s' %plot_var.replace('_',' '), va='center', rotation='vertical', fontsize=20)
    title = '' #'Food consumed per organism'
    fig.suptitle(title)


    for i, FOOD in enumerate(FOODS):

        c = cmap(norm(i))

        muF = np.mean(FOOD, axis=1)
        if scatter:
            ax.scatter(iter_list, muF, color=c, label=labels[i], alpha=0.15)
        else:
            ax.plot(iter_list, muF, color=c, label=labels[i])

        if not scatter:
            sigmaF = FOOD.std(axis=1)
            ax.fill_between(iter_list, muF + sigmaF, muF - sigmaF,
                            color=c, alpha=a
                            )

    custom_legend = [Line2D([0], [0], marker='o', color='w',
                            markerfacecolor=cmap(norm(0)), markersize=15),
                     Line2D([0], [0], marker='o', color='w',
                            markerfacecolor=cmap(norm(1)), markersize=15),
                     Line2D([0], [0], marker='o', color='w',
                            markerfacecolor=cmap(norm(2)), markersize=15),]

    #Custom legend for multiple runs in one plot removed:
    if multiple_sim:
        ax.legend(custom_legend, sim_labels, loc='upper left')
        #ax.legend(custom_legend, [r'$\beta = 10$', r'$\beta = 1$', r'$\beta = 0.1$'], loc='upper left')


    if multiple_sim:
        savefolder = 'multi_sim_plots/'
        for loadfile in loadfiles:
            savefolder += loadfile[0:18] + '__'
        savefolder += '/'
    else:
        folder = 'save/' + loadfile
        savefolder = folder + '/figs/' + plot_var + '_line/'


    #savefilename = savefolder + plot_var + '_gen' + str(iter_list[0]) + '-' + str(iter_list[-1]) + '.png'
    savefilename = savefolder + '{}_gen{}-{}-total-inds{}_{}.png'.format(
        plot_var, str(iter_list[0]), str(iter_list[-1]), len(isings_list), name_extension)
    if not path.exists(savefolder):
        makedirs(savefolder)

    # TODO:THIS line is temporary!!!
    #plt.xlim((0, 50))
    if saveFigBool:
        plt.savefig(savefilename, bbox_inches='tight', dpi=300)
        # plt.close()

        savemsg = 'Saving ' + savefilename
        print(savemsg)

    plt.show()
    #  Trying to fix memory leak with this:
    plt.cla()
    plt.clf()
    plt.close('all')
    ax.clear()

def upper_tri_masking(A):
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:, None] < r
    return A[mask]

def fitness(loadfile, iter_list, isings_list, numAgents, autoLoad, saveFigBool, plot_var):

    folder = 'save/' + loadfile

    folder2 = folder + '/figs/' + plot_var + '/' # + '_line/'
    fname2 = folder2 + plot_var + \
             str(iter_list[0]) + '-' + str(iter_list[1] - iter_list[0]) + '-' + str(iter_list[-1]) + \
             '.npz'


    if path.isfile(fname2) and autoLoad:
        #Loading previously saved files
        txt = 'Loading: ' + fname2
        print(txt)
        data = np.load(fname2)
        FOOD = data['FOOD']
    elif not isings_list is None:
        #Loading directly from isings_list in case it has been passed
        FOOD = np.zeros((len(iter_list), numAgents))
        for ii, isings in enumerate(isings_list):
            food = []
            for i, I in enumerate(isings):
                exec('food.append(I.%s)' % plot_var)
            FOOD[ii, :] = food
        if not path.exists(folder2):
            makedirs(folder2)
        np.savez(fname2, FOOD=FOOD)
    else:
        #Otherwise load file directly
        FOOD = np.zeros((len(iter_list), numAgents))
        for ii, iter in enumerate(iter_list):
            filename = 'save/' + loadfile + '/isings/gen[' + str(iter) + ']-isings.pickle'
            startstr = 'Loading simulation:' + filename
            print(startstr)

            try:
                isings = pickle.load(open(filename, 'rb'))
            except Exception:
                print("Error while loading %s. Skipped file" % filename)
                #Leads to the previous datapoint being drawn twice!!


            food = []


            for i, I in enumerate(isings):
                exec('food.append(I.%s)' % plot_var)

            # food = np.divide(food, 6)
            FOOD[ii, :] = food

        if not path.exists(folder2):
            makedirs(folder2)

        np.savez(fname2, FOOD=FOOD)
    return FOOD

if __name__ == '__main__':

    loadfile = sys.argv[1]
    plot_var = sys.argv[2] #plot_var = 'v'
    isings_list = [load_isings(loadfile)]
    main(loadfile, plot_var, isings_lists=isings_list)

