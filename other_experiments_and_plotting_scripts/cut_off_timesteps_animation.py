from helper_functions.automatic_plot_helper import load_isings_attr
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
import pandas as pd
import seaborn as sns
import sys


def main(sim_name, list_attr, len_cut_off, n_th_frame, fps):
    #isings = load_isings(sim_name)
    attrs_list = load_isings_attr(sim_name, list_attr)
    cut_offs_to_plot = np.arange(0, len_cut_off, n_th_frame)

    #Change wdir to save tmp png files in local folder (quick hack)
    save_tmp_path = 'save/{}/figs/cut_of_animation/tmp/'.format(sim_name)
    if not os.path.exists(save_tmp_path):
        os.makedirs(save_tmp_path)
    cur_wdir = os.getcwd()
    os.chdir(save_tmp_path)

    animate_cut_off_violin(attrs_list, list_attr, cut_offs_to_plot, sim_name, len_cut_off, fps, cur_wdir)
    animate_cut_off(attrs_list, list_attr, cut_offs_to_plot, sim_name, len_cut_off, fps, cur_wdir)
    os.chdir(cur_wdir)


def animate_cut_off(attrs_list, list_attr, cut_offs_to_plot, sim_name, len_cut_off, fps, cur_wdir, dpi=100):
    plt.rcParams.update({'font.size': 20})
    fig = plt.figure(figsize=(19, 10))
    ani = animation.FuncAnimation(fig, update_plot,
                                  fargs=[attrs_list, list_attr], interval=1,
                                  frames=cut_offs_to_plot)
    Writer = animation.FFMpegFileWriter
    writer = Writer(fps=fps, metadata=dict(artist='Jan Prosi'), bitrate=1800)
    writer.frame_format = 'png'

    save_path = '/{}/save/{}/figs/cut_of_animation/'.format(cur_wdir, sim_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_name = '{}_cut_off_{}_ts.mp4'.format(list_attr, len_cut_off)
    #ani.save(save_path+save_name, writer=writer, dpi=dpi)
    ani.save(save_path+save_name, writer=writer)


def update_plot(cut_num, attrs_list, list_attr):
    plt.cla()

    mean_attrs_list, gen_mean_mean_attrs_list = load_and_process_attrs(list_attr, cut_num, attrs_list)

    x_axis = np.arange(len(gen_mean_mean_attrs_list))
    plt.scatter(x_axis, gen_mean_mean_attrs_list, alpha=0.15, color='blue')
    if list_attr == 'energies':
        ylabel = 'mean energy'
    elif list_attr == 'velocities':
        ylabel = 'mean velocity'
    else:
        ylabel = list_attr

    plt.ylabel(ylabel)
    plt.xlabel('Generation')
    plt.title('Cut off {} time step'.format(cut_num))


def animate_cut_off_violin(attrs_list, list_attr, cut_offs_to_plot, sim_name, len_cut_off, fps, cur_wdir, dpi=100):
    plt.rcParams.update({'font.size': 20})
    fig = plt.figure(figsize=(25, 10))
    ani = animation.FuncAnimation(fig, update_violin_plot,
                                  fargs=[attrs_list, list_attr], interval=1,
                                  frames=cut_offs_to_plot)
    Writer = animation.FFMpegFileWriter
    writer = Writer(fps=fps, metadata=dict(artist='Jan Prosi'), bitrate=1800)
    writer.frame_format = 'png'

    save_path = '/{}/save/{}/figs/cut_of_animation/'.format(cur_wdir, sim_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_name = '{}_cut_off_violin{}_ts.mp4'.format(list_attr, len_cut_off)
    #ani.save(save_path+save_name, writer=writer, dpi=dpi)
    ani.save(save_path+save_name, writer=writer)


def update_violin_plot(cut_num, attrs_list, list_attr, n_th_entry_violin=100):
    mean_attrs_list, gen_mean_mean_attrs_list = load_and_process_attrs(list_attr, cut_num, attrs_list)

    plt.cla()

    short_mean_attrs_list = mean_attrs_list[0::n_th_entry_violin]
    df_labels = np.arange(0, len(mean_attrs_list)-1, n_th_entry_violin)
    df = pd.DataFrame(data=short_mean_attrs_list, index=df_labels)
    df = df.T
    #chart = sns.violinplot(data=df, width=0.8, inner='quartile', scale='width', linewidth=0.05)
    chart = sns.violinplot(data=df, width=0.8, inner='box', scale='width')
    #df.mean().plot(style='_', c='black')  # , ms=30
    #legend_elements = [Line2D([0], [0], marker='_', color='black', label='mean', markerfacecolor='g', markersize=10)]
    #plt.legend(handles=legend_elements)
    plt.title('Cut off {} time step'.format(cut_num))

    if list_attr == 'energies':
        ylabel = 'mean energy'
    elif list_attr == 'velocities':
        ylabel = 'mean velocity'
    else:
        ylabel = list_attr

    plt.ylabel(ylabel)
    plt.xlabel('Generation')



def load_and_process_attrs(list_attr, cut_num, attrs_list):
    # isings is a list of generations including again isings, therefore iterating through the gernerations with list
    # comprehensions, then again iterating through different individuals of one generation within that
    #attrs_list = [attribute_from_isings(ising, list_attr) for ising in isings]

    #Cutting first off generations
    attrs_list = [cut_attrs(cut_num, attrs) for attrs in attrs_list]

    # !!! Taking mean of list_attr NOT MEDIAN !!!
    mean_attrs_list = [[np.mean(list_attr) for list_attr in attrs] for attrs in attrs_list]
    # Now we have the attributes of all inidividuals of all generations in a nice list of lists availablöe for plotting

    # Taking mean over every generation, so we have one data point for each generation
    gen_mean_mean_attrs_list = [np.mean(attrs_one_gen) for attrs_one_gen in mean_attrs_list]

    return mean_attrs_list, gen_mean_mean_attrs_list



def cut_attrs(cut_num, attrs):
    '''
    Cuts away first -cut_num- entries from list attribute
    '''
    new_attrs = []
    for list_attr in attrs:
        list_attr = list_attr[cut_num:]
        new_attrs.append(list_attr)
    return new_attrs


if __name__ == '__main__':
    sim_name = sys.argv[1]
    #sim_name = 'sim-20200604-235417-g_2000_-t_2000_-b_0.1_-dream_c_0_-nat_c_0_-ref_0_-rec_c_0_-n_energies_velocities_saved'
    # sim-20200604-235417-g_2000_-t_2000_-b_0.1_-dream_c_0_-nat_c_0_-ref_0_-rec_c_0_-n_energies_velocities_saved_SMALL_TEST_COPY
    len_cut_off = 20 #500
    n_th_frame = 10
    list_attrs = ['energies', 'velocities']
    fps = 1
    #list_attrs = ['energies']
    for list_attr in list_attrs:
        main(sim_name, list_attr, len_cut_off, n_th_frame, fps)