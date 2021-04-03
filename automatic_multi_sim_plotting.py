from helper_functions.automatic_plot_helper import load_settings
from helper_functions.automatic_plot_helper import load_isings
from embodied_ising_helper import plot_anything_combined


def main(sim_names, load_isings_list=True):
    settings = load_settings(sim_names[0])
    if load_isings_list:
        isings_lists = [load_isings(sim_name) for sim_name in sim_names]
    plot_anything_auto(sim_names, ['Beta', 'avg_velocity'], settings, isings_lists=isings_lists, autoLoad=False)





def plot_anything_auto(sim_name, plot_vars, settings, isings_lists = None, autoLoad = True):
    '''
    :param plot_vars: List of string of which each represents an attribute of the isings class
    :param isings_list: List of all isings generations in case it has been loaded previously
    '''

    if settings['energy_model']:
        #os.system("python plot__anything_combined {} avg_energy".format(sim_name))
        plot_anything_combined.main(sim_name, 'avg_energy', isings_lists=isings_lists, autoLoad=autoLoad)
    else:
        #os.system("python plot__anything_combined {} fitness".format(sim_name))
        plot_anything_combined.main(sim_name, 'fitness', isings_lists=isings_lists, autoLoad=autoLoad)

    for plot_var in plot_vars:
        plot_anything_combined.main(sim_name, plot_var, isings_lists=isings_lists, autoLoad=autoLoad)



if __name__ == '__main__':
    #  Order for beta 0.1 1 1 10
    sim_names = [
                'sim-20200103-170603-ser_-s_-b_0.1_-ie_2_-a_0_200_500_1000_1500_1999',
                'sim-20200103-170556-ser_-s_-b_1_-ie_2_-a_0_500_1000_1500_1999',
                'sim-20191229-191241-ser_-s_-b_10_-ie_2_-a_0_500_1000_2000'
                ]
    main(sim_names)
