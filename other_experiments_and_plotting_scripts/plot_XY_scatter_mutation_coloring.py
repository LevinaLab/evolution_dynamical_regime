from helper_functions.automatic_plot_helper import load_multiple_isings_attrs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pickle
from matplotlib.lines import Line2D

def mutation_coloring_main(sim_name, plot_settings):
    plt.rc('text', usetex=True)
    font = {'family': 'serif', 'size': 18, 'serif': ['computer modern roman']}
    plt.rc('font', **font)

    if not plot_settings['only_plot']:
        multiple_isings_attrs = load_multiple_isings_attrs(sim_name, [plot_settings['x_attr'], plot_settings['y_attr'], 'prev_mutation'])
        save_plot_data(sim_name, multiple_isings_attrs, plot_settings)
    else:
        multiple_isings_attrs = load_plot_data(sim_name, plot_settings)
    plot(multiple_isings_attrs, sim_name, plot_settings)


def save_plot_data(sim_name, multiple_isings_attrs, plot_settings):
    save_dir = 'save/{}/figs/mutations_colored/'.format(sim_name)
    save_name = 'multiple_isings_attrs.pickle'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    pickle_out = open(save_dir + save_name, 'wb')
    pickle.dump(multiple_isings_attrs, pickle_out)
    pickle_out.close()


def load_plot_data(sim_name, plot_settings):
    save_dir = 'save/{}/figs/mutations_colored/'.format(sim_name)
    save_name = 'multiple_isings_attrs.pickle'
    print('Load plot data from: {}{}'.format(save_dir, save_name))
    file = open(save_dir+save_name, 'rb')
    multiple_isings_attrs = pickle.load(file)
    file.close()
    return multiple_isings_attrs


def plot(multiple_isings_attrs, sim_name, plot_settings):
    plt.figure(figsize=(10,7))
    for gen, attr_dict in enumerate(multiple_isings_attrs):
        attr_dict['mutation_colors'] = list(map(plot_settings['prev_mutation_colors'].get, attr_dict['prev_mutation']))
        attr_dict['mutation_sizes'] = list(map(plot_settings['prev_mutation_sizes'].get, attr_dict['prev_mutation']))
        attr_dict['mutation_alphas'] = list(map(plot_settings['prev_mutation_alphas'].get, attr_dict['prev_mutation']))

        attr_dict['mutation_sizes_only_point'] = list(map(plot_settings['prev_mutation_sizes_only_point'].get, attr_dict['prev_mutation']))

    # for i in range(len(attr_dict[plot_settings['x_attr']])):
        #     plt.scatter(attr_dict[plot_settings['x_attr']][i], attr_dict[plot_settings['y_attr']][i], c=attr_dict['mutation_colors'][i], s=attr_dict['mutation_sizes'][i], alpha=attr_dict['mutation_alphas'][i])
        plt.scatter(attr_dict[plot_settings['x_attr']], attr_dict[plot_settings['y_attr']], c=attr_dict['mutation_colors'], s=attr_dict['mutation_sizes'], alpha=0.2)
        plt.scatter(attr_dict[plot_settings['x_attr']], attr_dict[plot_settings['y_attr']], c=attr_dict['mutation_colors'], s=attr_dict['mutation_sizes_only_point'], alpha=0.8)

    plt.ylabel(plot_settings['y_label'])
    plt.xlabel(plot_settings['x_label'])


    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=15, alpha=0.75, label=r'One Organism'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=plot_settings['prev_mutation_colors']['copy'], markersize=15, alpha=0.75, label=r'Previously selected and copied'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=plot_settings['prev_mutation_colors']['mate'], markersize=15, alpha=0.75, label=r'Child of mating algorithm'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=plot_settings['prev_mutation_colors']['point'], markersize=15, alpha=1, label=r'Mutated'),
    ]
    if plot_settings['plot_legend']:
        plt.legend(handles=legend_elements, fontsize=17)
    plt.title(plot_settings['title'])
    savefolder = 'save/{}/figs/mutations_colored/'.format(sim_name)
    savefilename = 'mutations_colored.png'
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)
    plt.savefig(savefolder+savefilename, dpi=300, bbox_inches='tight')



if __name__ == '__main__':
    # sim_name = 'sim-20201026-224639_parallel_b1_fixed_4000ts/sim-20201026-224642-b_1_-g_8000_-t_4000_-rec_c_2000_-c_props_10_10_-2_2_100_40_-c_1_-subfolder_sim-20201026-224639_parallel_b1_fixed_4000ts_-n_Run_1' #
    # sim_name = 'sim-20201026-224709_parallel_b10_fixed_4000ts/sim-20201026-224711-b_10_-g_8000_-t_4000_-rec_c_2000_-c_props_10_10_-2_2_100_40_-c_1_-subfolder_sim-20201026-224709_parallel_b10_fixed_4000ts_-n_Run_1'
    # sim_names = ['sim-20201022-190553_parallel_b1_normal_seas_g4000_t2000/sim-20201022-190555-b_1_-g_4000_-t_2000_-noplt_-subfolder_sim-20201022-190553_parallel_b1_normal_seas_g4000_t2000_-n_Run_1', 'sim-20201022-190615_parallel_b10_normal_seas_g4000_t2000/sim-20201022-190618-b_10_-g_4000_-t_2000_-noplt_-subfolder_sim-20201022-190615_parallel_b10_normal_seas_g4000_t2000_-n_Run_1']
    # sim_names = ['sim-20201022-190553_parallel_b1_normal_seas_g4000_t2000_HEL_ONLY_PLOT_MUTATION/sim-20201022-190555-b_1_-g_4000_-t_2000_-noplt_-subfolder_sim-20201022-190553_parallel_b1_normal_seas_g4000_t2000_-n_Run_1', 'sim-20201022-190615_parallel_b10_normal_seas_g4000_t2000_HEL_ONLY_PLOT_MUTATION/sim-20201022-190618-b_10_-g_4000_-t_2000_-noplt_-subfolder_sim-20201022-190615_parallel_b10_normal_seas_g4000_t2000_-n_Run_1']
    sim_names = ['sim-20201022-190615_parallel_b10_normal_seas_g4000_t2000/sim-20201022-190618-b_10_-g_4000_-t_2000_-noplt_-subfolder_sim-20201022-190615_parallel_b10_normal_seas_g4000_t2000_-n_Run_{}'.format(i) for i in range(6,11)]
    plot_legend = [True, False]
    # titles = [r'$\beta_\mathrm{init} = 1$', r'$\beta_\mathrm{init} = 10$']
    # for sim_name, legend, title in zip(sim_names, plot_legend, titles):
    for sim_name in sim_names:
        legend = False
        title = ''
        plot_settings = {}
        plot_settings['only_plot'] = False

        plot_settings['title'] = title
        plot_settings['plot_legend'] = legend

        plot_settings['x_attr'] = 'generation'
        plot_settings['y_attr'] = 'avg_energy'

        plot_settings['x_label'] = 'Generation'
        plot_settings['y_label'] = r'$\langle E_\mathrm{org} \rangle$'

        normal_size = 0.4
        normal_alpha = 0.3

        mutated_size = 0.4
        mutated_alpha = 1.0
        # 'slateblue'

        plot_settings['prev_mutation_colors'] = {'init': 'xkcd:mid green', 'copy': 'xkcd:mid green', 'point': 'navy', 'mate': 'xkcd:deep rose'}
        # plot_settings['prev_mutation_colors'] = {'init': 'olive', 'copy': 'olive', 'point': 'navy', 'mate': 'maroon'}
        plot_settings['prev_mutation_sizes'] = {'init': normal_size, 'copy': normal_size, 'point': mutated_size, 'mate': normal_size}
        plot_settings['prev_mutation_alphas'] = {'init': normal_alpha, 'copy': normal_alpha, 'point': mutated_alpha, 'mate': normal_alpha}

        plot_settings['prev_mutation_sizes_only_point'] = {'init': 0, 'copy': 0, 'point': mutated_size, 'mate': 0}
        mutation_coloring_main(sim_name, plot_settings)

    # Explanation for 'init' being also marked as green and thus count as copied
    # init are not really the initialized organisms, but in reality those, the fittest 10, that have been copied 15 times, but not point mutated

    '''
    Explanation of EA
    Fittest 20 individuals are copied into next generation --> FIRST 20 POSITION OF NEW GENERATION
    Fittest 10 are again copied 15 times, those copies are mutated by a probability of 10 % --> NEXT 15 POSITION OF NEW GENERATION
    For mutation see self.mutate. This includes edge weight mutations, adding/removing of edges, beta mutations (again for certain probabilities)
    The 25 individuals that were created this way will be parents to the last 15 individuals

    '''