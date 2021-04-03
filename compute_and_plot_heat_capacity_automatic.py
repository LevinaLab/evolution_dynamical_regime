from helper_functions.automatic_plot_helper import detect_all_isings
from helper_functions.automatic_plot_helper import list_to_blank_seperated_str
from helper_functions.automatic_plot_helper import load_settings
import visualize_heat_capacity_generational_automatic
import os

def main(sim_name, settings, generations = None, recorded = False, add_subfolder_to_sim_name=True):
    if add_subfolder_to_sim_name:
        sim_name = '{}/{}'.format(settings['save_subfolder'], sim_name)
    if generations is None:
        gen_nums = detect_all_isings(sim_name)
        generations = [gen_nums[-1]]
    cores = settings['cores']
    compute_plot_heat_capacity(sim_name, generations, cores, settings, recorded)

def compute_plot_heat_capacity(sim_name, generation_list, cores, settings, recorded):
    gens_str = list_to_blank_seperated_str(generation_list)
    R, thermal_time, beta_low, beta_high, beta_num, y_lim_high = settings['heat_capacity_props']
    if recorded:
        os.system('bash bash-heat-capacity-generational-automatic-recorded.sh {} "{}" {} {}'
                  .format(sim_name, gens_str, cores, beta_num-1))
    else:
        os.system('bash bash-heat-capacity-generational-automatic.sh {} "{}" {}'.format(sim_name, gens_str, cores))
    if settings['plot_heat_cap']:
        visualize_heat_capacity_generational_automatic.main(sim_name, settings, None, recorded)
    #visualize_heat_capacity_generational_automatic_recorded.main(sim_name, settings, None, recorded)

if __name__ == '__main__':
    # sim_name = 'sim-20200514-013839-g_5_-t_2000_-ref_0_-nat_c_0_-dream_c_0_-rec_c_2_-c_15_-n_long_test'#'sim-20200327-220128-g_8000_-b_1_-ref_2000_-a_500_1000_2000_4000_6000_8000_-n_3_sensors'
    # settings = load_settings(sim_name)
    # main(sim_name, settings, recorded=True)
    # sim_names = ['sim-20200604-235424-g_2000_-t_2000_-b_1_-dream_c_0_-nat_c_0_-ref_0_-rec_c_0_-n_energies_velocities_saved',
    #              'sim-20200604-235433-g_2000_-t_2000_-b_10_-dream_c_0_-nat_c_0_-ref_0_-rec_c_0_-n_energies_velocities_saved',
    #              'sim-20200606-014815-g_2000_-t_4000_-b_1_-dream_c_0_-nat_c_0_-ref_0_-rec_c_0_-noplt_-n_energies_velocities_saved_more_time_steps',
    #              'sim-20200606-014837-g_2000_-t_4000_-b_10_-dream_c_0_-nat_c_0_-ref_0_-rec_c_0_-noplt_-n_energies_velocities_saved_more_time_steps'
    #              ]
    sim_names = ['sim-20200721-184149-g_4000_-t_2000_-iso_-ref_500_-rec_c_250_-a_50_100_500_1000_-no_trace_-n_different_nbetas_from_scratch_isolated_no_beta_jump_GOOD']
    for sim_name in sim_names:
        cores = 20
        generation_list = [3750]
        settings = load_settings(sim_name)
        compute_plot_heat_capacity(sim_name, generation_list, cores, settings, recorded=True)

