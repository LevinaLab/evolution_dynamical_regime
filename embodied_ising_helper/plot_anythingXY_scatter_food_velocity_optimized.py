#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg') #For server use
from matplotlib import colors
from os import makedirs, path
from helper_functions.automatic_plot_helper import detect_all_isings
from helper_functions.automatic_plot_helper import load_settings
from helper_functions.automatic_plot_helper import load_isings

'''
loadfiles = ['beta_experiment/beta-0-1/sim-20180512-105719',
             'beta_experiment/beta-1/sim-20180511-163319',
             'beta_experiment/beta-10/sim-20180512-105824']
'''
def main(loadfile, settings, isings_list, plot_var_x, plot_var_y, s=0.8, alpha=0.13, autoLoad=True, x_lim=None,
         y_lim=None, log=True, y_noise=True):


    loadfiles = [loadfile]#loadfiles = ['sim-20191114-000009_server']
    iter_list = detect_all_isings(loadfile) #  iter_list = np.arange(0, 2000, 1)
    #
    energy_model = settings['energy_model']
    numAgents = settings['pop_size']
    #autoLoad = True
    saveFigBool = True
    fixGen2000 = False

    new_order = [2, 0, 1]

    labels = [r'$\beta_i = 0.1$', r'$\beta_i = 1$', r'$\_i = 10$']

    cmap = plt.get_cmap('seismic')
    norm = colors.Normalize(vmin=0, vmax=len(loadfiles))  # age/color mapping
    # norm = [[194, 48, 32, 255],
    #         [146, 49, 182, 255],
    #         [44, 112, 147, 255]
    #         ]
    # norm = np.divide(norm, 255)
    a = 0.15 # alpha

    x_pars_list, y_pars_list = fitness(loadfile, iter_list, isings_list, numAgents, autoLoad, saveFigBool, plot_var_x,
                                       plot_var_y)
    #fig, ax = plt.subplots()
    cmap = plt.get_cmap('plasma')
    norm = colors.Normalize(vmin=0, vmax=len(iter_list))
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 10}

    plt.rc('font', **font)
    plt.figure()
    for gen, (x_pars, y_pars) in enumerate(zip(x_pars_list, y_pars_list)):
        c = cmap(norm(gen))
        if y_noise:
            y_pars = y_pars.astype(float)
            y_pars = y_pars + np.random.rand(np.shape(y_pars)[0]) - 0.5
        ax = plt.scatter(x_pars, y_pars, s = s, alpha = alpha, c=c)
        if y_noise:
            plt.ylim(1,1000)
        if log:
            plt.xscale('log')
            plt.yscale('log')
        #TODO:colour acc to generation!!
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.xlabel('{}'.format(plot_var_x.replace('_', ' ')))
    plt.ylabel('{}'.format(plot_var_y.replace('_', ' ')))

    folder = 'save/' + loadfile
    savefolder = folder + '/figs/' + plot_var_x + '_vs_' + plot_var_y + '_line/'
    savefilename = savefolder + plot_var_x + '_vs_' + plot_var_y + '_gen' + str(iter_list[0]) + '-' + str(
        iter_list[-1]) + '.png'
    if not path.exists(savefolder):
        makedirs(savefolder)

    if saveFigBool:
        plt.savefig(savefilename, bbox_inches='tight', dpi=500)


    plt.show()
    #  Trying to fix memory leak with this:
    plt.cla()
    plt.clf()
    plt.close('all')





    '''
    ###########################
    FOODS = []
    for loadfile in loadfiles:
        x_pars_list, y_pars_list = fitness(loadfile, iter_list, isings_list, numAgents, autoLoad, saveFigBool, plot_var_x, plot_var_y)
        FOODS.append((x_pars_list, y_pars_list))
        # FIX THE DOUBLE COUNTING PROBLEM
        # if f.shape[0] > 2000 and fixGen2000:
        #     print('Fixing Double Counting at Gen 2000')
        #     f[2000, :] = f[2000, :] - f[1999, :]


    # FIX THE DOUBLE COUNTING OF THE FITNESS


    plt.rc('text', usetex=True)
    font = {'family': 'serif', 'size': 28, 'serif': ['computer modern roman']}
    plt.rc('font', **font)
    plt.rc('legend', **{'fontsize': 20})

    fig, ax = plt.subplots(1, 1, figsize=(19, 10))
    fig.text(0.51, 0.035, r'${}$'.format(plot_var_x.replace('_',' ')), ha='center', fontsize=20)
    # fig.text(0.07, 0.5, r'$Avg. Food Consumed$', va='center', rotation='vertical', fontsize=20)
    fig.text(0.07, 0.5, r'${}$'.format(plot_var_y.replace('_',' ')), va='center', rotation='vertical', fontsize=20)
    title = 'Food consumed per organism'
    fig.suptitle(title)


    for i, FOOD in enumerate(FOODS):

        # for i in range(0, numAgents):
        #     ax.scatter(iter_list, FOOD[:, i], color=[0, 0, 0], alpha=0.2, s=30)
        c = cmap(norm(new_order[i]))
        # c = norm[i]
        # c = norm[IC[i]]

        muF = np.mean(FOOD, axis=1)
        ax.plot(iter_list, muF, color=c, label=labels[new_order[i]])

        # for numOrg in range(FOOD.shape[1]):
        #     ax.scatter(iter_list, FOOD[:, numOrg],
        #                alpha=0.01, s=8, color=c,  label=labels[new_order[i]])

        # maxF = np.max(FOOD, axis=1)
        # minF = np.min(FOOD, axis=1)
        # ax.fill_between(iter_list, maxF, minF,
        #                 color=np.divide(c, 2), alpha=a)

        sigmaF = FOOD.std(axis=1)
        ax.fill_between(iter_list, muF + sigmaF, muF - sigmaF,
                        color=c, alpha=a
                        )

    custom_legend = [Line2D([0], [0], marker='o', color='w',
                            markerfacecolor=cmap(norm(1)), markersize=15),
                     Line2D([0], [0], marker='o', color='w',
                            markerfacecolor=cmap(norm(0)), markersize=15),
                     Line2D([0], [0], marker='o', color='w',
                            markerfacecolor=cmap(norm(2)), markersize=15),]

    # custom_legend = [Circle((0, 0), 0.001,
    #                         facecolor=cmap(norm(1))),
    #                  Circle((0, 0), 1,
    #                         facecolor=cmap(norm(0))),
    #                  Circle((0, 0), 1,
    #                         facecolor=cmap(norm(2)))]

    ax.legend(custom_legend, [r'$\beta = 10$', r'$\beta = 1$', r'$\beta = 0.1$'], loc='upper left')

    # plt.legend(loc=2)

    # yticks = np.arange(0, 150, 20)
    # ax.set_yticks(yticks)


        # xticks = [0.1, 0.5, 1, 2, 4, 10, 50, 100, 200, 500, 1000, 2000]
        # ax.set_xscale("log", nonposx='clip')
        # ax.set_xticks(xticks)
        # ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    folder = 'save/' + loadfile
    savefolder = folder + '/figs/' + plot_var_x + '_vs_'+ plot_var_y + '_line/'
    savefilename = savefolder + plot_var_x + '_vs_'+ plot_var_y + '_gen' + str(iter_list[0]) + '-' + str(iter_list[-1]) + '.png'
    if not path.exists(savefolder):
        makedirs(savefolder)

    if saveFigBool:
        plt.savefig(savefilename, bbox_inches='tight', dpi=300)
        # plt.close()

        savemsg = 'Saving ' + savefilename
        print(savemsg)

    # if saveFigBool:
    #     savefolder = folder + '/figs/fitness/'
    #     savefilename = savefolder + 'fitness_gen_' + str(iter_list[0]) + '-' + str(iter_list[-1]) + '.png'
    #     plt.savefig(bbox_inches = 'tight', dpi = 300)
    plt.show()
    '''
def upper_tri_masking(A):
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:, None] < r
    return A[mask]

def fitness(loadfile, iter_list, isings_list, numAgents, autoLoad, saveFigBool, plot_var_x, plot_var_y):

    folder = 'save/' + loadfile

    folder2 = folder + '/figs/' + plot_var_x + '_vs_'+ plot_var_y + '/'
    fname2 = folder2 + plot_var_x + '_vs_'+ plot_var_y + \
             str(iter_list[0]) + '-' + str(iter_list[1] - iter_list[0]) + '-' + str(iter_list[-1]) + \
             '.npz'

    # if path.isfile(fname2) and autoLoad:
    #     txt = 'Loading: ' + fname2
    #     print(txt)
    #     data = np.load(fname2)
    #     FOOD = data['FOOD']
    if path.isfile(fname2) and autoLoad:
        #Loading previously saved files
        txt = 'Loading: ' + fname2
        print(txt)
        data = np.load(fname2)
        x_pars_list = data['x_pars_list']
        y_pars_list = data['y_pars_list']
    else:
        #Loading directly from isings_list in case it has been passed
        x_pars_list = np.zeros((len(iter_list), numAgents))
        y_pars_list = np.zeros((len(iter_list), numAgents))
        for ii, isings in enumerate(isings_list):
            x_pars = []
            y_pars = []
            for i, I in enumerate(isings):
                exec('x_pars.append(I.%s)' % plot_var_x)
                exec('y_pars.append(I.%s)' % plot_var_y)
            x_pars_list[ii, :] = x_pars
            y_pars_list[ii, :] = y_pars
        if not path.exists(folder2):
            makedirs(folder2)
        np.savez(fname2, x_pars_list=x_pars_list)
    return x_pars_list, y_pars_list
    # else:
    #     #Otherwise load file directly
    #     FOOD = np.zeros((len(iter_list), numAgents))
    #     for ii, iter in enumerate(iter_list):
    #         filename = 'save/' + loadfile + '/isings/gen[' + str(iter) + ']-isings.pickle'
    #         startstr = 'Loading simulation:' + filename
    #         print(startstr)
    #
    #         try:
    #             isings = pickle.load(open(filename, 'rb'))
    #         except Exception:
    #             print("Error while loading %s. Skipped file" % filename)
    #             #Leads to the previous datapoint being drawn twice!!
    #
    #
    #         food = []


        #     for i, I in enumerate(isings):
        #         exec('food.append(I.%s)' % plot_var)
        #
        #     # food = np.divide(food, 6)
        # x_pars_list[ii, :] = x_pars
        #
        # if not path.exists(folder2):
        #     makedirs(folder2)

        #np.savez(fname2, FOOD=x_pars_list)


if __name__ == '__main__':

    #loadfile = sys.argv[1]
    #plot_var = sys.argv[2] #plot_var = 'v'
    loadfile = 'sim-20200103-170556-ser_-s_-b_1_-ie_2_-a_0_500_1000_1500_1999'
    plot_var_x = 'avg_energy'
    plot_var_y = 'avg_velocity'#'food'
    isings_list = load_isings(loadfile)
    settings = load_settings(loadfile)
    #TODO: add something that detetcts .npz file and skips loading isings in that case

    main(loadfile, settings, isings_list, plot_var_x, plot_var_y, autoLoad=False, x_lim=(-1,20), y_lim=(-0.1, 0.8), alpha = 0.05)
    #TODO: Evt. PCA oder decision trees um herauszufinden welche eigenschaften wichtig sind fuer hohe avg energy?

