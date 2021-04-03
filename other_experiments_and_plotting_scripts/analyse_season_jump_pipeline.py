import numpy as np
from helper_functions.automatic_plot_helper import load_isings_from_list
from helper_functions.automatic_plot_helper import detect_all_isings
import matplotlib.pylab as plt
from os import makedirs, path
import seaborn as sns
import os
import pandas as pd
import copy
from matplotlib.lines import Line2D

def extract_attr(isings_list, attr):
    val_list = []
    for isings in isings_list:
        for I in isings:
            exec('val_list.append(I.{})'.format(attr))
    return val_list

def load_trained_vals(trained_sim_name, attr, n_last_gens = 100):
    load_gens_trained = detect_all_isings(trained_sim_name)
    load_gens_trained = load_gens_trained[-(n_last_gens):]
    trained_isings_list = load_isings_from_list(trained_sim_name, load_gens_trained)
    trained_vals = extract_attr(trained_isings_list, attr)
    # trained_avg = np.avg(trained_vals)
    # trained_std = np.std(trained_vals)
    return trained_vals

# def load_switched_vals(switched_sim_name, attr, n_last_gens = 100):
#     # load_gens_switched = detect_all_isings(switched_sim_name)
#     # load_gens_switched = load_gens_switched[-(n_last_gens + 1):]
#     # switched
#     #switched_isings_list = load_isings(switched_sim_name)
#
#     switched_vals = extract_attr(switched_isings_list, attr)
#     # switched_avg = np.avg(switched_vals)
#     # switched_std = np.std(switched_vals)
#     return switched_vals

def load_plot_data(trained_sim_names, switched_sim_names, attr):
    '''
    :param trained_sim_names: names of one parameter set
    :param switched_sim_names: names of one parameter set
    :return: all values of sets concatenated
    '''

    all_trained_vals = []
    for trained_sim_name in trained_sim_names:
        all_trained_vals.append(load_trained_vals(trained_sim_name, attr))
    # trained_avg = np.avg(all_trained_vals)
    # trained_std = np.std(all_trained_vals)

    all_switched_vals = []
    for switched_sim_name in switched_sim_names:
        #all_switched_vals.append(load_switched_vals(switched_sim_name, attr))
        all_switched_vals.append(load_trained_vals(switched_sim_name, attr))
    # switched_avg = np.avg(all_switched_vals)
    # switched_std = np.std(all_switched_vals)

    return all_trained_vals, all_switched_vals

# def plot(trained_sets, switched_sets):
#     trained_avgs = []
#     trained_stds = []
#     switched_avgs = []
#     switched_stds = []
#     for trained_set, switched_set in zip(trained_sets):
#         all_trained_vals, all_switched_vals = calc_plot_data(trained_set, switched_set)
#         trained_avgs.append(np.avg(all_trained_vals))
#         trained_stds.append(np.std(all_trained_vals))

def add_folder_name(sets, folder):
    new_sets = []
    for set in sets:
        new_set = []
        for sim_name in set:
            new_set.append(folder + sim_name)
        new_sets.append(new_set)
    return new_sets

def create_DF(all_data, labels, sims_per_label=4):
    columns = []
    for label in labels:
        for i in range(sims_per_label):
            columns.append(label + str(i))
    all_data = np.array([np.array(data) for data in all_data])
    all_data = np.asarray(all_data)
    all_data = np.stack(all_data, axis=0)
    df = pd.DataFrame(all_data, index=columns)
    df = df.transpose()
    return df, columns

def create_violin_colors(color_list, repeat = 4):
    out_color_list = []
    for color in color_list:
        for i in range(repeat):
            out_color_list.append(color)
    return out_color_list


def plot(trained_sets, switched_sets, attr, labels, new_order_labels, trained_folder=None, switched_folder=None,
         auto_load=False, yscale='linear', ylim=None, save_addition='', xlim=None):
    #for some reason auto_load stopped working, did not look for bug yet

    if not trained_folder is None:
        trained_sets = add_folder_name(trained_sets, trained_folder)
        switched_sets = add_folder_name(switched_sets, switched_folder)


    # -----PLot concetinated data
    # trained_sets = [j for sub in trained_sets for j in sub]
    # switched_sets = [j for sub in switched_sets for j in sub]


    npz_name = 'save/{}figs/{}_boxplot.npz'.format(switched_folder, attr)

    if path.isfile(npz_name) and auto_load:
        txt = 'Loading: ' + npz_name
        print(txt)
        data = np.load(npz_name)
        all_data = data['all_data']
        data = data['data']
    else:
        data = []
        all_data = []
        for trained_set, switched_set in zip(trained_sets, switched_sets):
            trained_vals, switched_vals = load_plot_data(trained_set, switched_set, attr)

            for trained_single_sim in trained_vals:
                all_data.append(trained_single_sim)
            for switched_single_sim in switched_vals:
                all_data.append(switched_single_sim)

            trained_vals_concat = [j for sub in trained_vals for j in sub]
            switched_vals_concat = [j for sub in switched_vals for j in sub]
            data.append(trained_vals_concat)
            data.append(switched_vals_concat)



        # plt.boxplot(data)
        # plt.xticks(np.arange(1, len(labels) + 1), labels, rotation='vertical')
        # plt.show()

    savefolder = 'save/{}figs/{}_'.format(switched_folder, attr)
    if not path.exists(savefolder):
        makedirs(savefolder)
    np.savez(npz_name, all_data=all_data, data=data)

    df, names = create_DF(all_data, labels)
    df = reorder_df(df, new_order_labels)
    all_data_reordered = df_to_nested_list(df)

    # plt.figure(figsize=(25, 5))
    # chart = sns.violinplot(data=df, width=0.8, inner='quartile', scale='width', linewidth=0.01)  # inner='quartile'
    # chart.set_xticklabels(chart.get_xticklabels(), rotation=70)
    # df.mean().plot(style='*')
    # plt.savefig('{}violin_df_neworder{}.png'.format(savefolder, save_addition), dpi=300, bbox_inches='tight')
    # plt.show()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
              '#17becf']
    violin_colors = create_violin_colors(colors)


    # LEGEND

    legend_elements = [Line2D([0], [0], marker='_', color='black', label='mean', markerfacecolor='g', markersize=10)]


    plt.figure(figsize=(25, 10))
    chart = sns.violinplot(data=df, width=0.8, inner='quartile', scale='width', linewidth=0.05,
                           palette=violin_colors)  # inner='quartile'
    df.mean().plot(style='_', c='black', ms=30)
    chart.set_xticklabels(chart.get_xticklabels(), rotation=70)
    plt.yscale(yscale)
    plt.gca().set_ylim(top=20)
    plt.legend(handles=legend_elements)
    plt.savefig('{}violin_df{}.png'.format(savefolder, save_addition), dpi=300, bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots()
    col_i = 0
    for i, d in enumerate(all_data_reordered):
        color = colors[col_i]
        noisy_x = i * np.ones((1, len(d))) + np.random.random(size=len(d)) * 0.5
        ax.scatter(noisy_x[0, :], d, alpha=0.6, s=0.01, c=color)
        if (i + 1) % 4 == 0:
            col_i += 1

    mean_series = df.mean()
    mean_series.plot(style='_', c='black', ms=7)

    ax.set_xticks(np.arange(32))
    ax.set_yscale(yscale)
    #plt.ylabel('median energy')
    plt.ylabel(attr)
    plt.xticks(np.arange(1, len(new_order_labels) * 4 + 1, 4), new_order_labels, rotation=70)
    plt.legend(handles=legend_elements)
    plt.savefig('{}scatter{}.png'.format(savefolder, save_addition), dpi=300, bbox_inches='tight')
    plt.show()
    # plt.boxplot(data, showmeans=True)
    # plt.xticks(np.arange(1, len(labels) + 1), labels, rotation='vertical')
    # plt.ylabel(attr)
    # plt.savefig('{}boxplot.png'.format(savefolder), dpi=200, bbox_inches='tight')
    # plt.show()

    # plt.boxplot(all_data_reordered, showmeans=True)
    # plt.xticks(np.arange(1, len(new_order_labels)*4 + 1, 4), new_order_labels, rotation='vertical')
    # plt.ylabel(attr)
    # plt.savefig('{}boxplot_all.png'.format(savefolder), dpi=200, bbox_inches='tight')
    # plt.show()

    plt.figure(figsize=(20, 5))
    plt.violinplot(all_data_reordered, showmeans=True, showextrema=False, widths=0.8)
    plt.xticks(np.arange(1, len(new_order_labels)*4 + 1, 4), new_order_labels, rotation=70)
    plt.yscale(yscale)
    #plt.ylabel('median energy')
    plt.ylabel(attr)
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.savefig('{}violin_all{}.png'.format(savefolder, save_addition), dpi=300, bbox_inches='tight')
    plt.show()










    # fig, ax = plt.subplots()
    # for i, s in enumerate(df):
    #     d = df[s].tolist()
    #     noisy_x = i * np.ones((1, len(d))) + np.random.random(size=len(d)) * 0.5
    #     ax.scatter(noisy_x[0, :], d, alpha=0.2, s=0.1)
    # ax.set_xticks(np.arange(32))
    # ax.set_yscale('log')
    #
    # plt.xticks(np.arange(1, len(labels) * 4 + 1, 4), labels, rotation=70)
    # plt.savefig('{}scatter{}.png'.format(savefolder, save_addition), dpi=300, bbox_inches='tight')
    # plt.show()

def df_to_nested_list(df):
    df = copy.deepcopy(df)
    out_list = []
    for col in df:
        out_list.append(df[col].tolist())
    return out_list

def which(trained_sim, switched_sets, two_dim = True, trained_also_simulated_again = True):
    if trained_also_simulated_again:
        start = trained_sim.find('-l_sim')+4
        end = start + 19
        trained_sim = trained_sim[start:end]
    if two_dim:
        switched_sets_1D = [j for sub in switched_sets for j in sub]
    else:
        switched_sets_1D = switched_sets
    for switched_sim in switched_sets_1D:
        if trained_sim in switched_sim:
            return switched_sim
    #raise FileNotFoundError('No switched simulation found for the trained simulation {}'.format(trained_sim))



def sort_switched_sets(trained_sets, switched_sets, two_dim = True):
    '''Sort switched sets according to trained sets'''
    sorted_switched_sets = []
    for trained_set in trained_sets:
        sorted_switched_set = []
        for trained_sim in trained_set:
            acc_switched_sim = which(trained_sim, switched_sets, two_dim)
            sorted_switched_set.append(acc_switched_sim)
        sorted_switched_sets.append(sorted_switched_set)
    return sorted_switched_sets

def load_switched_sets_sorted(switched_folder, trained_sets):
    directory_list = [f.path for f in os.scandir('save/{}'.format(switched_folder)) if f.is_dir()]
    switched_set = []
    for sim_name in directory_list:
        sim_name = sim_name.split('/')[-1]
        if 'sim-' in sim_name:
            switched_set.append(sim_name)
    return sort_switched_sets(trained_sets, switched_set, two_dim = False)

def sort_sets(order_list, sets):
    '''sort sets according to order given in order list.
    :param order_list: list of ints, where each int gives place in new order
    '''
    ordered_sets = [x for _, x in sorted(zip(order_list, sets))]
    #ordered_labels = [x for _, x in sorted(zip(order_list, labels))]
    return ordered_sets

def reorder_df(df, new_order_labels):
    old_cols = df.columns.tolist()
    new_cols = []
    for new_label in new_order_labels:
        for old_col in old_cols:
            if new_label in old_col:
                new_cols.append(old_col)
    df_new = copy.deepcopy(df)
    df_new = df_new[new_cols]
    return df_new




if __name__ == '__main__':
    #Sort labels according to order of trained sets!!!!!
    trained_folder = '4th_run_same_season/'
    switched_folder = '4th_run_switched_season/'

    #attr = 'avg_energy'
    #attr = 'avg_velocity'
    attr = 'food'

    #labels in order of trained sets
    labels = ['b10 summer', 'b10 switched to winter', 'b10 winter', 'b10 switched to summer',
                'b1 summer', 'b1 switched to winter', 'b1 winter', 'b1 switched to summer']

    #labels in wanted plot order
    new_order_labels = ['b1 summer', 'b1 switched to summer', 'b10 summer', 'b10 switched to summer', 'b1 winter',
              'b1 switched to winter', 'b10 winter', 'b10 switched to winter']

    #To list all files in a certain folder in windows command shell: dir /B > filelist.txt

    trained_sets =  [['sim-20200214-180611-l_sim-20200209-124814-ser_-b_10_-f_100_-n_1_-r_200_-ser_-f_100_-n_1same',
                    'sim-20200214-180611-l_sim-20200209-124814-ser_-b_10_-f_100_-n_2_-r_200_-ser_-f_100_-n_2same',
                    'sim-20200214-180611-l_sim-20200209-124814-ser_-b_10_-f_100_-n_3_-r_200_-ser_-f_100_-n_3same',
                    'sim-20200214-180611-l_sim-20200209-124814-ser_-b_10_-f_100_-n_4_-r_200_-ser_-f_100_-n_4same'],
                    ['sim-20200214-180611-l_sim-20200209-124814-ser_-b_10_-f_10_-n_1_-r_200_-ser_-f_10_-n_1same',
                    'sim-20200214-180611-l_sim-20200209-124814-ser_-b_10_-f_10_-n_2_-r_200_-ser_-f_10_-n_2same',
                    'sim-20200214-180611-l_sim-20200209-124814-ser_-b_10_-f_10_-n_3_-r_200_-ser_-f_10_-n_3same',
                    'sim-20200214-180611-l_sim-20200209-124814-ser_-b_10_-f_10_-n_4_-r_200_-ser_-f_10_-n_4same'],
                    ['sim-20200214-180611-l_sim-20200209-124814-ser_-b_1_-f_100_-n_1_-r_200_-ser_-f_100_-n_1same',
                    'sim-20200214-180611-l_sim-20200209-124814-ser_-b_1_-f_100_-n_2_-r_200_-ser_-f_100_-n_2same',
                    'sim-20200214-180611-l_sim-20200209-124814-ser_-b_1_-f_100_-n_3_-r_200_-ser_-f_100_-n_3same',
                    'sim-20200214-180611-l_sim-20200209-124814-ser_-b_1_-f_100_-n_4_-r_200_-ser_-f_100_-n_4same'],
                    ['sim-20200214-180611-l_sim-20200209-124814-ser_-b_1_-f_10_-n_1_-r_200_-ser_-f_10_-n_1same',
                    'sim-20200214-180611-l_sim-20200209-124814-ser_-b_1_-f_10_-n_2_-r_200_-ser_-f_10_-n_2same',
                    'sim-20200214-180611-l_sim-20200209-124814-ser_-b_1_-f_10_-n_3_-r_200_-ser_-f_10_-n_3same',
                    'sim-20200214-180611-l_sim-20200209-124814-ser_-b_1_-f_10_-n_4_-r_200_-ser_-f_10_-n_4same']]


    # trained_sets = [['sim-20200121-213309-ser_-cfg_2000_100_-b_1_-nmb',
    #                 'sim-20200121-213313-ser_-cfg_2000_100_-b_1_-nmb',
    #                 'sim-20200121-213321-ser_-cfg_2000_100_-b_1_-nmb',
    #                 'sim-20200121-213347-ser_-cfg_2000_100_-b_1_-nmb_-a_200_1999_2190',],
    #                 ['sim-20200121-213356-ser_-cfg_2000_100_-b_10_-nmb',
    #                  'sim-20200121-213400-ser_-cfg_2000_100_-b_10_-nmb',
    #                  'sim-20200121-213403-ser_-cfg_2000_100_-b_10_-nmb',
    #                  'sim-20200121-213424-ser_-cfg_2000_100_-b_10_-nmb_-a_200_1999_2190'],
    #                  ['sim-20200121-213437-ser_-f_10_-cfg_2000_100_-b_1_-nmb',
    #                   'sim-20200121-213441-ser_-f_10_-cfg_2000_100_-b_1_-nmb',
    #                   'sim-20200121-213446-ser_-f_10_-cfg_2000_100_-b_1_-nmb',
    #                   'sim-20200121-213458-ser_-f_10_-cfg_2000_100_-b_1_-nmb_-a_200_1999_2190'],
    #                 ['sim-20200121-213512-ser_-f_10_-cfg_2000_100_-b_10_-nmb',
    #                  'sim-20200121-213520-ser_-f_10_-cfg_2000_100_-b_10_-nmb',
    #                  'sim-20200121-213524-ser_-f_10_-cfg_2000_100_-b_10_-nmb',
    #                  'sim-20200121-213537-ser_-f_10_-cfg_2000_100_-b_10_-nmb_-a_200_1999_2190'
    #                  ]]


    switched_sets = load_switched_sets_sorted(switched_folder, trained_sets)

    # new_order_trained_sets = [,,0]
    # new_order_switched_sets = [0,,1 ]
    # switched_sets =

    # switched_sets = [['sim-20200130-205401-ser_-b_1_-f_10_-r_200_-li_1999_-a_5_-l_sim-20200121-213347-ser_-cfg_2000_100_-b_1_-nmb_-a_200_1999_2190',
    #                 'sim-20200130-205401-ser_-b_1_-f_10_-r_200_-li_1999_-l_sim-20200121-213309-ser_-cfg_2000_100_-b_1_-nmb',
    #                 'sim-20200130-205401-ser_-b_1_-f_10_-r_200_-li_1999_-l_sim-20200121-213313-ser_-cfg_2000_100_-b_1_-nmb',
    #                 'sim-20200130-205401-ser_-b_1_-f_10_-r_200_-li_1999_-l_sim-20200121-213321-ser_-cfg_2000_100_-b_1_-nmb'],
    #                 ['sim-20200130-205401-ser_-f_100_-b_10_-r_200_-li_1999_-a_5_-l_sim-20200121-213537-ser_-f_10_-cfg_2000_100_-b_10_-nmb_-a_200_1999_2190',
    #                 'sim-20200130-205401-ser_-f_100_-b_10_-r_200_-li_1999_-l_sim-20200121-213512-ser_-f_10_-cfg_2000_100_-b_10_-nmb',
    #                 'sim-20200130-205401-ser_-f_100_-b_10_-r_200_-li_1999_-l_sim-20200121-213520-ser_-f_10_-cfg_2000_100_-b_10_-nmb',
    #                 'sim-20200130-205401-ser_-f_100_-b_10_-r_200_-li_1999_-l_sim-20200121-213524-ser_-f_10_-cfg_2000_100_-b_10_-nmb',
    #                 'sim-20200130-205401-ser_-f_100_-b_1_-r_200_-li_1999_-a_5_-l_sim-20200121-213458-ser_-f_10_-cfg_2000_100_-b_1_-nmb_-a_200_1999_2190'],
    #                 ['sim-20200130-205401-ser_-f_100_-b_1_-r_200_-li_1999_-l_sim-20200121-213437-ser_-f_10_-cfg_2000_100_-b_1_-nmb',
    #                 'sim-20200130-205401-ser_-f_100_-b_1_-r_200_-li_1999_-l_sim-20200121-213441-ser_-f_10_-cfg_2000_100_-b_1_-nmb',
    #                 'sim-20200130-205401-ser_-f_100_-b_1_-r_200_-li_1999_-l_sim-20200121-213446-ser_-f_10_-cfg_2000_100_-b_1_-nmb',
    #                 'sim-20200130-205401-ser_-f_10_-b_10_-r_200_-li_1999_-a_5_-l_sim-20200121-213424-ser_-cfg_2000_100_-b_10_-nmb_-a_200_1999_2190'],
    #                 ['sim-20200130-205401-ser_-f_10_-b_10_-r_200_-li_1999_-l_sim-20200121-213356-ser_-cfg_2000_100_-b_10_-nmb',
    #                 'sim-20200130-205401-ser_-f_10_-b_10_-r_200_-li_1999_-l_sim-20200121-213400-ser_-cfg_2000_100_-b_10_-nmb',
    #                 'sim-20200130-205401-ser_-f_10_-b_10_-r_200_-li_1999_-l_sim-20200121-213403-ser_-cfg_2000_100_-b_10_-nmb']]

    switched_sets = sort_switched_sets(trained_sets, switched_sets)

    # switched_sets = [[
    #
    #                 'sim-20200129-212115-ser_-b_1_-f_10_-r_200_-li_1999_-a_5_-l_sim-20200121-213347-ser_-cfg_2000_100_-b_1_-nmb_-a_200_1999_2190',
    #                 'sim-20200129-212115-ser_-b_1_-f_10_-r_200_-li_1999_-l_sim-20200121-213309-ser_-cfg_2000_100_-b_1_-nmb',
    #                 'sim-20200129-212115-ser_-b_1_-f_10_-r_200_-li_1999_-l_sim-20200121-213313-ser_-cfg_2000_100_-b_1_-nmb',
    #                 'sim-20200129-212115-ser_-b_1_-f_10_-r_200_-li_1999_-l_sim-20200121-213321-ser_-cfg_2000_100_-b_1_-nmb'],
    #                 ['sim-20200129-212115-ser_-f_100_-b_10_-r_200_-li_1999_-a_5_-l_sim-20200121-213537-ser_-f_10_-cfg_2000_100_-b_10_-nmb_-a_200_1999_2190',
    #                 'sim-20200129-212115-ser_-f_100_-b_10_-r_200_-li_1999_-l_sim-20200121-213512-ser_-f_10_-cfg_2000_100_-b_10_-nmb',
    #                 'sim-20200129-212115-ser_-f_100_-b_10_-r_200_-li_1999_-l_sim-20200121-213520-ser_-f_10_-cfg_2000_100_-b_10_-nmb',
    #                 'sim-20200129-212115-ser_-f_100_-b_10_-r_200_-li_1999_-l_sim-20200121-213524-ser_-f_10_-cfg_2000_100_-b_10_-nmb'],
    #                 ['sim-20200129-212115-ser_-f_100_-b_1_-r_200_-li_1999_-a_5_-l_sim-20200121-213424-ser_-cfg_2000_100_-b_10_-nmb_-a_200_1999_2190',
    #                 'sim-20200129-212115-ser_-f_100_-b_1_-r_200_-li_1999_-l_sim-20200121-213403-ser_-cfg_2000_100_-b_10_-nmb',
    #                 'sim-20200129-212115-ser_-f_100_-b_1_-r_200_-li_1999_-l_sim-20200121-213437-ser_-f_10_-cfg_2000_100_-b_1_-nmb',
    #                 'sim-20200129-212115-ser_-f_100_-b_1_-r_200_-li_1999_-l_sim-20200121-213441-ser_-f_10_-cfg_2000_100_-b_1_-nmb'],
    #                 ['sim-20200129-212115-ser_-f_10_-b_10_-r_200_-li_1999_-l_sim-20200121-213356-ser_-cfg_2000_100_-b_10_-nmb',
    #                 'sim-20200129-212115-ser_-f_10_-b_10_-r_200_-li_1999_-l_sim-20200121-213400-ser_-cfg_2000_100_-b_10_-nmb',
    #                 'sim-20200129-212115-ser_-f_10_-b_10_-r_200_-li_1999_-l_sim-20200121-213403-ser_-cfg_2000_100_-b_10_-nmb',]]



    plot(trained_sets, switched_sets, attr, labels, new_order_labels, trained_folder, switched_folder, yscale='linear', ylim=None,
         save_addition='_nolog', xlim=None, auto_load=False)
    #yscale='symlog'







# def analyse(trained_sim_name, switched_sim_name, n_last_gens = 100):
#     load_gens_trained = detect_all_isings(trained_sim_name)
#     load_gens_trained = load_gens_trained[-(n_last_gens+1)]
#     trained_isings_list = load_isings_from_list(trained_sim_name, load_gens_trained)
#     switched_isings_list = load_isings(switched_sim_name)
#     trained_vals = extract_attr(trained_isings_list, 'avg_energy')
#     switched_vals = extract_attr(switched_isings_list, 'avg_energy')
#     trained_avg = np.avg(trained_vals)
#     switched_avg = np.avg(switched_vals)


