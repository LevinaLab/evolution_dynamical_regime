import os
import numpy as np
from helper_functions.automatic_plot_helper import load_isings_specific_path
from helper_functions.automatic_plot_helper import attribute_from_isings
import copy
import pandas as pd
import pickle
import matplotlib.pylab as plt
from matplotlib.lines import Line2D
import seaborn as sns


def load_and_plot(runs_name, attr):
    '''
    This function is for manual use, to load simulation results and plot them
    runs_folder is the main folder that all runs are saved in
    '''
    run_combis = load_run_combis(runs_name)
    plot_pipeline(run_combis, runs_name, attr)


def plot_pipeline(run_combis, runs_name, attr):
    '''
    This is the function called by switch_season_repeat_pipeline right after the runs are finished
    '''
    # Total number of repeats (of same simulation)
    tot_same_repeats = run_combis[0].tot_same_repeats

    unordered_object_df = create_df(run_combis, runs_name, attr)
    new_order_labels = ['b1 summer', 'b1 switched to summer', 'b10 summer', 'b10 switched to summer', 'b1 winter',
                        'b1 switched to winter', 'b10 winter', 'b10 switched to winter']
    ordered_df = reorder_df(unordered_object_df, new_order_labels)
    ordered_list = df_to_nested_list(ordered_df)
    scatter_plot(ordered_df, ordered_list, new_order_labels, runs_name, attr, tot_same_repeats)
    violin_plot(ordered_df, runs_name, attr, tot_same_repeats)
    pass
    # TODO: Create plotting functions


def create_df(run_combis, runs_name, attr):
    data = []
    labels = []
    for run_combi in run_combis:
        switched_repeat_isings, same_repeat_isings = extract_isings(run_combi, runs_name)


        #Create data frame label
        if run_combi.season == 'summer':
            switched_to = 'winter'
        elif run_combi.season == 'winter':
            switched_to = 'summer'

        #switched_label = "b{} switched from {} to {}".format(run_combi.beta, run_combi.season, switched_to)
        switched_label = "b{} switched to {} {}".format(run_combi.beta, switched_to, run_combi.same_repeat)

        same_label = "b{} {} {}".format(run_combi.beta, run_combi.season, run_combi.same_repeat)

        # Make the currently 2d "repeat_isings" list 1d, which means that all ising objects from all repeated generations are in one big list
        switched_repeat_isings_1d = make_2d_list_1d(switched_repeat_isings)
        same_repeat_isings_1d = make_2d_list_1d(same_repeat_isings)
        del switched_repeat_isings
        del same_repeat_isings

        # Extract attributes from isings
        switched_repeat_isings_1d_attr = attribute_from_isings(switched_repeat_isings_1d, attr)
        same_repeat_isings_1d_attr = attribute_from_isings(same_repeat_isings_1d, attr)
        del switched_repeat_isings_1d
        del same_repeat_isings_1d


        # Append stuff to lists that will be converted to df
        data.append(same_repeat_isings_1d_attr)
        data.append(switched_repeat_isings_1d_attr)
        labels.append(same_label)
        labels.append(switched_label)



    sims_per_label = run_combi.tot_same_repeats
    unordered_df = list_to_df(data, labels, sims_per_label)
    return unordered_df


def make_2d_list_1d(in_list):
    out_list = []
    for sub_list in in_list:
        for en in sub_list:
            out_list.append(en)
    return out_list

def load_run_combis(runs_folder):
    path = 'save/{}/run_combis.pickle'.format(runs_folder)
    file = open(path, 'rb')
    run_combis = pickle.load(file)
    file.close()
    return run_combis


def extract_isings(run_combi, runs_name):
    path = 'save/{}/{}/'.format(runs_name, run_combi.subfolder)
    both_repeat_isings = all_files_in(path, 'repeat_isings')
    if len(both_repeat_isings) != 2:
        raise Exception('Found more than two folders that include "repeat_isings" in {}'.format(path))
    for repeat_isings in both_repeat_isings:
        if 'switched' in repeat_isings:
            path_switched_repeat_isings = repeat_isings
        elif 'same' in repeat_isings:
            path_same_repeat_isings = repeat_isings
    switched_repeat_isings = load_isings_specific_path(path_switched_repeat_isings)
    same_repeat_isings = load_isings_specific_path(path_same_repeat_isings)

    return switched_repeat_isings, same_repeat_isings#


def all_files_in(path, sub_str):
    '''
    Returns directories of all files in folder and sub-folders of path including sub_str
    '''
    #filenames = []
    files = []
    # r=root, d=directories, f=files
    for r, d, f in os.walk(path):
        for dir in d:
            if sub_str in dir:
                files.append(os.path.join(r, dir))
    return files
    # for filename in glob.iglob(path + '**/*', recursive=True):
    #     if sub_str in filename:
    #         filenames.append(filename)
    # return filenames



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


def list_to_df(all_data, labels, sims_per_label=4):
    # all_data = np.array([np.array(data) for data in all_data])
    # all_data = np.asarray(all_data)
    # all_data = np.stack(all_data, axis=0)
    df = pd.DataFrame(all_data, index=labels)
    df = df.transpose()
    return df


def df_to_nested_list(df):
    df = copy.deepcopy(df)
    out_list = []
    for col in df:
        out_list.append(df[col].tolist())
    return out_list


def generate_common_plotting_params(runs_name):
    savefolder = 'save/{}/figs/'.format(runs_name)
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
              '#17becf']

    legend_elements = [Line2D([0], [0], marker='_', color='black', label='mean', markerfacecolor='g', markersize=10)]

    return savefolder, colors, legend_elements


def violin_plot(df, runs_name, attr, tot_same_repeats, yscale='linear'):

    savefolder, colors, legend_elements = generate_common_plotting_params(runs_name)

    plt.rcParams.update({'font.size': 22})


    violin_colors = create_violin_colors(colors, repeat=tot_same_repeats)

    plt.figure(figsize=(25, 10))

    if attr == 'avg_energy':
        plt.axhline(y=2, linewidth=1, color='r')
    elif attr== 'avg_velocity':
        plt.axhline(y=0.05, linewidth=1, color='r')

    chart = sns.violinplot(data=df, width=0.8, inner='quartile', scale='width', linewidth=0.05,
                           palette=violin_colors)  # inner='quartile'
    df.mean().plot(style='_', c='black', ms=30)
    chart.set_xticklabels(chart.get_xticklabels(), rotation=70)
    plt.yscale(yscale)
    #plt.gca().set_ylim(top=20)
    plt.legend(handles=legend_elements)
    plt.title(attr)
    plt.savefig('{}violin_{}.png'.format(savefolder, attr), dpi=300, bbox_inches='tight')
    plt.show()

def scatter_plot(df, all_data_reordered, new_order_labels, runs_name, attr, tot_same_repeats, yscale='linear'):
    '''
    Creates scatter plot of data
    df: data frame of all data
    all_data_reordered: the same data frame as list
    new_order_labels = The labels that were used to sort the labels... not one label per run but one label for all
    repeated runs
    '''


    savefolder, colors, legend_elements = generate_common_plotting_params(runs_name)

    plt.rcParams.update({'font.size': 11})

    fig, ax = plt.subplots()

    if attr == 'avg_energy':
        plt.axhline(y=2, linewidth=0.5, color='r')
    elif attr== 'avg_velocity':
        plt.axhline(y=0.05, linewidth=0.5, color='r')

    col_i = 0
    for i, d in enumerate(all_data_reordered):
        color = colors[col_i]
        noisy_x = i * np.ones((1, len(d))) + np.random.random(size=len(d)) * 0.5
        ax.scatter(noisy_x[0, :], d, alpha=0.6, s=0.01, c=color)
        if (i + 1) % tot_same_repeats == 0:
            col_i += 1

    mean_series = df.mean()
    mean_series.plot(style='_', c='black', ms=7)



    ax.set_xticks(np.arange(8 * tot_same_repeats))
    ax.set_yscale(yscale)
    #plt.ylabel('median energy')
    plt.ylabel(attr)
    if tot_same_repeats == 1:
        plt.xticks(np.arange(0, len(new_order_labels) * tot_same_repeats, tot_same_repeats), new_order_labels,
                   rotation=70)
    else:
        plt.xticks(np.arange(1, len(new_order_labels) * tot_same_repeats + 1, tot_same_repeats), new_order_labels,
                   rotation=70)
    plt.legend(handles=legend_elements)
    plt.title(attr)
    plt.savefig('{}scatter_{}.png'.format(savefolder, attr), dpi=300, bbox_inches='tight')
    plt.show()


def create_violin_colors(color_list, repeat=4):
    out_color_list = []
    for color in color_list:
        for i in range(repeat):
            out_color_list.append(color)
    return out_color_list


if __name__ == '__main__':
    # runs_folder = 'switch_seasons_20200524-031338_num_rep_100_same_rep_1_f_sum_100_f_win10_middle_long_test'
    # attr = 'food'
    # load_and_plot(runs_folder, attr)

    runs_folders = [
        'switch_seasons_20200526-124526_num_rep_200_same_rep_4_f_sum_100_f_win10_huge_run'

    ]

    #runs_folders = ['switch_seasons_20200524-135607_num_rep_200_same_rep_1_f_sum_100_f_win10_Other_energy_payment_REAL']
    #attrs = 'avg_energy', 'food', 'avg_velocity'
    attrs = ['Beta']
    for runs_folder in runs_folders:
        for attr in attrs:
            load_and_plot(runs_folder, attr)

