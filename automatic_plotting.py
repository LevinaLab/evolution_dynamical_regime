import os
import matplotlib as mpl
mpl.use('Agg') #For server use
from helper_functions.automatic_plot_helper import load_settings
from helper_functions.automatic_plot_helper import load_isings
from helper_functions.automatic_plot_helper import  load_top_isings
from embodied_ising_helper import plot_anythingXY_scatter_food_velocity_optimized, plot_anythingXY_scatter, \
    plot_anything_combined
import sys
import operator

'''!!!!!!!!ONLY CHANGE, WHEN SIMULATION IS NOT RUNNING!!!!!!!   gets called via os to prevent memory leak'''


def main(sim_name, only_top_isings=None, load_isings_list=True, final=False, subfolder=None):
    '''
    final defines whether this is the final/ last generation of simulation is plotted
    '''
    # if subfolder != None:
    #     sim_name = sim_name + subfolder
    settings = load_settings(sim_name)
    if load_isings_list:
        if only_top_isings is not None:
            isings_list = load_top_isings(sim_name, only_top_isings)
            settings['pop_size'] = only_top_isings
            save_txt_path = 'save/{}/figs/'.format(sim_name)
            if not os.path.exists(save_txt_path):
                os.makedirs(save_txt_path)
            f = open(save_txt_path + "Only_first_{}_fittest_individuals_have_been_plotted.txt".format(only_top_isings), "w+")
            f.close()
        else:
            isings_list = load_isings(sim_name, wait_for_memory=True)
            isings_list_fittest = [sorted(isings, key=operator.attrgetter('avg_energy'), reverse=True)[:20] for isings in isings_list]
            #isings_list = load_isings_from_list(sim_name, [0])
    plot_vars = ['Beta', 'avg_velocity', 'food']
    plot_var_tuples = [('generation', 'avg_energy'), ('generation', 'avg_velocity'), ('generation', 'food'),
                       ('generation', 'Beta'), ('Beta', 'avg_energy'), ('Beta', 'avg_velocity'),
                       ('avg_energy', 'avg_velocity'), ('avg_energy', 'food')]

    try:
        if settings['speciation']:
            append_species_stuff = [('species', 'shared_fitness'), ('species', 'avg_energy'), ('generation', 'species')]
            for tup in append_species_stuff:
                plot_var_tuples.append(tup)
    except KeyError:
        print('Older version, when speciation was not implemented')


    # Try plotting norm_avg_energy in case dataset already has I.time_steps
    try:
        for isings in isings_list:
            for I in isings:
                I.norm_avg_energy = I.avg_energy / I.time_steps
        plot_vars.append('norm_avg_energy')
        plot_var_tuples.append(('generation', 'norm_avg_energy'))

        if settings['random_food_seasons']:
            for isings in isings_list:
                for I in isings:
                    I.norm_food_and_ts_avg_energy = I.norm_avg_energy / I.food_in_env

            plot_vars.append('norm_food_and_ts_avg_energy')
            plot_var_tuples.append(('generation', 'norm_food_and_ts_avg_energy'))

    except Exception:
        print('Could not calculate norm_avg_energy (Do isings lack attribute I.time_steps?)')




    # try:
    plot_anything_auto(sim_name, plot_vars, settings, isings_list=isings_list_fittest, autoLoad=False)
    # except Exception:
    #    print('Could not create generational plots')


    try:
        plot_scatter_auto(sim_name, settings, plot_var_tuples, isings_list, autoLoad=False)
    except Exception:
        print('Could not create scatter plot')
    try:
        plot_anythingXY_scatter_food_velocity_optimized.main(sim_name, settings, isings_list, 'avg_velocity', 'food', s=0.8,
                                                             alpha=0.05, autoLoad=False)
    except Exception:
        print('Could not create food velocity scatter plot')


    if final:
        pass
        # try:
        #     if settings['cores'] != 0:
        #         compute_and_plot_heat_capacity_automatic.main(sim_name, settings)
        #
        # except Exception:
        #     print('Could not compute and plot heat capacity')

    #  Trying to fix memory leak:
    del isings_list
    del settings

    #plot_anythingXY_scatter_animation.main(sim_name, settings, isings_list, autoLoad=False, x_lim=None, y_lim=None)
    #  TODO: Animation dies not work for some reasone when called from here but does work when it is called itself... WHY???

def plot_scatter_auto(sim_name, settings, plot_var_tuples, isings_list, autoLoad = True):
    for plot_var_x, plot_var_y in plot_var_tuples:
        plot_anythingXY_scatter.main(sim_name, settings, isings_list, plot_var_x, plot_var_y, s=0.8, alpha=0.05,
                                     autoLoad=autoLoad, name_extension='all_inds')

def plot_anything_auto(sim_name, plot_vars, settings, isings_list = None, autoLoad = True):
    '''
    :param plot_vars: List of string of which each represents an attribute of the isings class
    :param isings_list: List of all isings generations in case it has been loaded previously
    '''

    if settings['energy_model']:
        #os.system("python plot__anything_combined {} avg_energy".format(sim_name))
        plot_anything_combined.main([sim_name], 'avg_energy', settings=settings, isings_lists=[isings_list],
                                    autoLoad=autoLoad, name_extension='fittest_inds')
    else:
        #os.system("python plot__anything_combined {} fitness".format(sim_name))
        plot_anything_combined.main([sim_name], 'fitness', settings=settings, isings_lists=[isings_list],
                                    autoLoad=autoLoad, name_extension='fittest_inds')

    for plot_var in plot_vars:
        plot_anything_combined.main([sim_name], plot_var, settings=settings, isings_lists=[isings_list],
                                    autoLoad=autoLoad, scatter=True, name_extension='fittest_inds')





def plot_all_in_folder(folder_name):
    '''
    :param folder_name: Has to be a sub_folder of save/
    :return:
    '''
    directory_list = [f.path for f in os.scandir('save/{}'.format(folder_name)) if f.is_dir()]
    for sim_name in directory_list:
        if 'sim-' in sim_name:
            sim_name = sim_name.replace('save/', '')
            main(sim_name)


if __name__ == '__main__':
    '''
    first argument sim_name
    second argument 'final_true' in case it is final run 'final_false' otherwise
    ! DON'T CHANGE THIS ! this is called by embodied_ising.py via os to prevent memory leak
    '''

    #sim_name ='3rd_4th_run_figures_training_runs_examples/sim-20200209-124814-ser_-b_10_-f_100_-n_1' #'sim-20200123-210723-g_20_-t_20_-ypi_0.05_-mf_0.1_-n_test' # 'sim-20191229-191241-ser_-s_-b_10_-ie_2_-a_0_500_1000_2000' #'sim-20200103-170603-ser_-s_-b_0.1_-ie_2_-a_0_200_500_1000_1500_1999'#'sim-20200103-170556-ser_-s_-b_1_-ie_2_-a_0_500_1000_1500_1999'
    final = False
    if sys.argv[2] == 'final_true':
        final = True
    # if sys.argv[3] == 'no_subfolder':
    #     subfolder = None
    # else:
    #     subfolder = sys.argv[3]
    main(sys.argv[1], final=final) # , subfolder=subfolder
    # sim_names = ['sim-20200209-124814-ser_-b_10_-f_100_-n_1',
    #             'sim-20200209-124814-ser_-b_10_-f_10_-n_1',
    #             'sim-20200209-124814-ser_-b_1_-f_100_-n_1',
    #             'sim-20200209-124814-ser_-b_1_-f_10_-n_1']
    # for sim_name in sim_names:
    #     main(sim_name)
    #plot_all_in_folder('seasons_training_one_season')

