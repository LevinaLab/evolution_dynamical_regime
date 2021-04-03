#!/usr/bin/env python

from embodied_ising import ising
from embodied_ising import food
from embodied_ising import EvolutionLearning
import numpy as np
import argparse
import automatic_plotting
import pickle
import time
from helper_functions.automatic_plot_helper import load_settings
from helper_functions.automatic_plot_helper import decompress_pickle



def create_settings():

    args = parse()
    # --- CONSTANTS ----------------------------------------------------------------+
    settings = {}
    
    # ENVIRONMENT SETTINGS
    settings['pop_size'] = args.pop_size #50  # number of organisms #150
    settings['numKill'] = int(settings['pop_size'] / 1.66)
    settings['food_num'] = args.food_num #100  # number of food particles
    settings['food_radius'] = 0.03
    settings['food_energy'] = args.food_energy
    settings['org_radius'] = 0.05
    settings['ANN'] = False  # Use ANN or Ising?
    settings['BoidOn'] = False  # Only use Boid model? #True

    settings['server_mode'] = args.server_mode
    settings['laptop_mode'] = args.laptop_mode
    
    # SIMULATION SETTINGS
    settings['TimeSteps'] = args.time_steps # number of timesteps per iteration #2000

    # number of system-wide spin updates per unit of time  (multiplies computation time)
    settings['thermalTime'] = args.thermal_time


    #  -- settings only used by switch_season_repeat_pipeline --
    settings['switch_off_evolution'] = args.switch_off_evolution

    # In case settings['switch_off_evolution'] == True and settings['loadfile'] != '' isings objects are saved in folder of
    # load simulation. In that case settings['switch_off_evolution'] has to be either positive or negative. According
    # to whether the season has bee switched or not. This is then indicited by the ising folders name. This feature is important for
    # switch_season_repeat_pipeline

    settings['repeat_pipeline_switched_boo'] = None
    settings['save_subfolder'] = args.save_subfolder
    settings['switch_seasons_repeat_pipeline'] = False  #  This has to be activated For the repeat runs of switch_season_repeat_pipeline
    settings['commands_in_folder_name'] = args.commands_in_folder_name



    settings['evolution_toggle'] = False  # only toggles for CriticalLearning
    settings['evolution_rate'] = 1  # only with critical learning number of iterations to skip to kill/mate (gives more time to eat before evolution)
    
    settings['dt'] = 0.2  # kinetic time step      (dt)
    settings['r_max'] = 720
    settings['dr_max'] = 90  # max rotational speed      (degrees per second)
    settings['v_max'] = args.v_max #999  # 0.5 max velocity              (units per second)
    settings['dv_max'] = args.a_max #  0.05  # max acceleration (+/-)    (units per second^2)

    settings['motor_neuron_acceleration'] = args.acc_motor
    
    settings['x_min'] = 0.0  # arena eastern border
    settings['x_max'] = 8.0  # arena western border
    settings['y_min'] = 0.0  # arena southern border
    settings['y_max'] = 8.0  # arena northern border
    
    settings['save_data'] = args.save_data
    #settings['plot'] = args.plot # make plots? #replaced by plot_generations
    # iterations. Also begins saving figures after this many iterations if 'plot' setting is 'False'
    settings['plot_generations'] = args.plot_gens #List of generations that animation should be created for
    #Might not work for two generations in a row in current implmentation
    settings['plotLive'] = False  # live updates of figures
    settings['frameRate'] = 1
    settings['animation_fps'] = args.fps
    settings['fading_traces_animation'] = args.fading_traces_animation
    settings['animation_dpi'] = args.animation_dpi
    
    settings['size'] = args.num_neurons #Total number of neurons in NN
    settings['nSensors'] = 4
    settings['nMotors'] = 4
    settings['learningrate'] = 0.01  # 0.01
    # how many hidden neurons are not connected to each other
    settings['numDisconnectedNeurons'] = 0 #  int((settings['size'] - settings['nSensors'] - settings['nMotors']) / 1.2)
    # how should organisms repopulate, duplicate or mate?
    settings['mateDupRatio'] = 0.5
    settings['mutationRateDup'] = args.mutationRateDup #0.1  # DUPLICATION mutation rate

    settings['init_beta'] = args.init_beta
    settings['mutateB'] = not args.no_mut_beta  # toggle to allow Beta (temperature) mutations (toggle off if critical learning is on)
    settings['sigB'] = args.sig_beta #0.02  # std for Beta mutation
    settings['diff_init_betas'] = args.diff_init_betas
    settings['beta_jump_mutations'] = args.beta_jump_mutations
    settings['beta_linspace'] = args.beta_linspace
    settings['change_beta_loaded_simulation'] = args.change_beta_loaded_simulation
    settings['beta_linspace_within_sim'] = args.beta_linspace_within_sim

    #settings['loadfile'] = sim-20191114-000009_server
    settings['loadfile'] = args.loadfile
    settings['iter'] = args.loaditer
    if settings['loadfile'] is '':
        settings['LoadIsings'] = False
    else:
        settings['LoadIsings'] = True

    # If set_isings is an ising object array, the simulation starts with those isings in generation 0
    settings['set_isings'] = None
    #Seasons
    settings['seasons'] = args.seasons #BOO; Activates seasons
    settings['years_per_iteration'] = args.years_per_iteration #Float amount of seasonal changes per iteration
    settings['min_food_winter'] = args.min_food_winter #  0.5 #FLOAT [0,1]; relative decimation of food in winter
    settings['chg_food_gen'] = args.chg_food_gen

    settings['abrupt_seasons_len'] = args.abrupt_seasons_len

    # Random food seasons
    settings['random_food_seasons'] = args.random_food_seasons
    settings['rand_food_season_limits'] = args.rand_food_season_limits
    
    settings['parallel_computing'] = False #BOO
    
    settings['energy_model'] = not args.no_ener_mod #BOO
    settings['v_min'] = args.v_min  # FLOAT [0,1)
    settings['cost_speed'] = args.cost_speed  # FLOAT [0,1] energy cost of speed as a factor of speed #default 0.05
    settings['initial_energy'] = args.init_energy  # Energy that each organism starts with in each simulation

    settings['plot_pipeline'] = args.plot_pipeline
    settings['refresh_plot'] = args.refresh_plot
    settings['dream_heat_capacity'] = args.dream_heat_capacity
    settings['recorded_heat_capacity'] = args.recorded_heat_capacity
    settings['minimal_energy_initializatin_heat_cap'] = args.minimal_energy_initializatin_heat_cap

    settings['heat_capacity_props'] = args.heat_capacity_props
    settings['plot_heat_cap'] = args.plot_heat_cap

    # natural heat capacity is calculated for every nth generation, if 0 no heat capacity is calculated
    settings['natural_heat_capacity_Nth_gen'] = args.natural_heat_capacity_Nth_gen
    # natural heat capacity beta factor distribution properties, last three linspace arguments
    #10 ** np.linspace(low_lim, high_lim, num_betas)
    settings['natural_heat_capacity_beta_fac_props'] = args.natural_heat_capacity_beta_fac_props  # [-1, 1, 102]
    settings['cores'] = args.cores

    settings['random_time_steps'] = args.random_time_steps
    settings['random_time_step_limits'] = args.random_time_step_limits

    settings['random_time_steps_power_law'] = args.random_time_steps_power_law
    settings['random_time_steps_power_law_limits'] = args.random_time_steps_power_law_limits


    settings['mutation_robustness_path'] = ''  # Special setting for mutation_robustness.py otherwise always =''

    # name that is added to folder that simulation is saved in
    settings['add_save_name'] = ''

    # Speciation stuff
    settings['speciation'] = args.speciation
    #Constants for delta formula c_top, c_weight, c_beta
    settings['shared_fitness_constants'] = tuple(args.shared_fitness_constants) #(1, 1, 1)
    settings['delta_threshold_speciation'] = args.delta_threshold_speciation

    # name that is added to folder that simulation is saved in
    settings['add_save_name'] = ''

    settings['isolated_populations'] = args.isolated_populations

    settings['save_energies_velocities_gens'] = args.save_energies_velocities_gens
    settings['save_energies_velocities_last_gen'] = args.save_energies_velocities_last_gen

    settings['compress_save_isings'] = args.compress_save_isings
    settings['max_speed_eat'] = args.max_speed_eat

    Iterations = args.iterations





    return settings, Iterations




def parse():
    parser = argparse.ArgumentParser(description=
                                     '''Agent-based evolutionary simulation of artificial organisms 
                                     controlled by a statistical neural net (ising model)
                                     #GENERATE ANIMATION FROM LOADED SIMULATION#
                                     Animating existing simulation for a certain generation:
                                     python3 train.py -l SIMULATION_NAME -li NUMBER_GENERATION -a 0 -g 1
                                     Animation will be saved in previous folder of simulation
                                     #SWITCH OFF ALL PLOTS#
                                     Use the following argument: -noplt 
                                     #DEFAULT VALUES#
                                     save_data=True, plot=False, iterations=2000, time_steps=2000, plot_gens=[], fps=20,
                        loadfile='', loaditer=1999, pop_size=50, food_num=100, init_beta=1.0, seasons=False,
                        server_mode=False, cost_speed=0.05, v_max=999.0, v_min=0.05, sig_beta=0.02, no_mut_beta=False,
                        init_energy=2, food_energy=1, no_ener_mod=False, plot_pipeline=True, chg_food_gen=None,
                        years_per_iteration=1, min_food_winter=0.1, thermal_time=10, diff_init_betas=None, acc_motor=True,
                        a_max=0.05, refresh_plot=0, dream_heat_capacity=0, laptop_mode=False, natural_heat_capacity_Nth_gen=0,
                        natural_heat_capacity_beta_fac_props=[-1, 1, 100], recorded_heat_capacity=0, abrupt_seasons_len=0, cores=3,
                        switch_off_evolution=False, fading_traces_animation=True, random_time_steps=False,
                        random_time_step_limits=[100, 8000], heat_capacity_props=[100, 10, -2, 2, 100, 40], speciation=False,
                        delta_threshold_speciation=1, shared_fitness_constants=[1, 1, 1], mutationRateDup=0.1,
                        isolated_populations=False, beta_jump_mutations=False, animation_dpi=150,
                        random_food_seasons=False, rand_food_season_limits=[1, 199], save_subfolder='',
                        save_energies_velocities_gens=None, save_energies_velocities_last_gen=True, random_time_steps_power_law=False,
                        random_time_steps_power_law_limits=[100, 1000000, 700], num_neurons=12, compress_save_isings= False, max_speed_eat=None,
                        beta_linspace=None
                                     ''')
    parser.add_argument('-p', '--pop', dest='pop_size', type=int, help='Number of individuals in each generation')
    parser.add_argument('-f','--food', dest='food_num', type=int,
                        help='''Number of food particles. Serves as largest number of food particles when seasons are 
                        activated''')
    parser.add_argument('-s', '--save', dest='save_data', action='store_false', help="Don't save data of simulation")
    parser.add_argument('-noplt', '--noplot', dest='plot_pipeline', action='store_false',
                        help='Run Plotting pipeline at end of simulation')
    parser.add_argument('-g', '--gen', type=int, dest='iterations', help='Number of generations in simulation')
    parser.add_argument('-t', '--ts', type=int, dest='time_steps',
                        help='Number of time steps in simulation')
    parser.add_argument('-num_neurons', dest='num_neurons', type=int,
                        help='Total number of neurons. There are 4 Sensor neurons and 4 Motor neurons, which are included,'
                             'therefore at least 9')
    parser.add_argument('-noevo', '--no_evolution', dest='switch_off_evolution', action='store_true',
                        help='''This boolean argument deactivates evolution. If command -l loadfile is active, isings 
                        are saved in folder of loaded simulation''')
    parser.add_argument('-b', '--beta', dest='init_beta', type=float, help='Initial beta of first generation')
    parser.add_argument('-bs', '--betas', nargs='+', required=False, dest='diff_init_betas', type=float
                        , help='''In case you want different initial beta values in the first generations define those 
                        in a blank separated list after this argument. Probability of each entry occurring
                         is uniformly distributed''')
    parser.add_argument('-b_linspace', dest='beta_linspace', nargs='+', required=False, type=float,
                        help='blank seperated list: index loglow loghigh number ... In case you want to parallely run multiple simulations with different beta values this'
                             'command will be helpful. It expects a blank seperated list: index_of_current_simulation beta_loglow beta_loghigh number_of_simualtions)'
                             'Those commands will be read by np.linspace, where index gives gives the desired position of the beta value'
                             'of the array created by linspace. loglow and loghigh give the log thresholds of the beta value.'
                             '(probably you want -1 and 1). Number has to be equal to the amount of simulations you are parallely running'
                             ', it gives the number of entries of the linspace array.'
                             'So just leave all values except for index constant and then iterate through index with'
                             'your parallelization tool. Only works when used in train.py')
    parser.add_argument('-b_linspace_within_sim', dest='beta_linspace_within_sim', action='store_true',
                        help='Initializes the first simulation of a simulation log linspace between 0.1 and 10')
    parser.add_argument('-std_b', dest='sig_beta', type=float,
                        help='Std of normal distribution for beta mutation')
    parser.add_argument('-b_jump', dest='beta_jump_mutations', action='store_true',
                        help='Activate jump mutations for beta. Those are random mutations of beta, where old beta'
                             'value is replaced by an exponentially distributed value of beta. This makes it easier'
                             'for a population to "hop out" of its current regime during evolution')
    parser.add_argument('-nmb', '--nomutb', dest='no_mut_beta', action='store_true', help='Switch off beta mutation')
    parser.add_argument('-b_load', dest='change_beta_loaded_simulation', type=float, help='Can only be used, when -l argument is used.'
                                                                              ' The beta of the loaded simulation is changed.')
    parser.add_argument('-a', '--ani', nargs='+', required=False, dest='plot_gens', type=int
                        , help='''Generations of which animation shall be created. 
                        Expects blank separated list of ints.''')
    parser.add_argument('-a_dpi', dest='animation_dpi', type=int,
                        help='Quality of animation in dpi. 150 is default medium quality, 300 is high quality')
    parser.add_argument('-no_trace', '--animation_no_trace', dest='fading_traces_animation', action='store_false',
                        help='Deactivate fading traces in animation. This significantly speeds up computational time'
                             ', This calls completely differnt animation module, so other things might vary as well.')
    parser.add_argument('-a_fps', type=int, dest='fps', help='FPS in animation')
    parser.add_argument('-l','--load', type=str, dest = 'loadfile',
                        help='Filename of previously saved simulation in save folder. Specify iteration using -li')
    parser.add_argument('-li', '--loadi', type=str, dest='loaditer',
                        help='Iteration of previously saved simulation that is loaded. Only use in combination with -l')
    parser.add_argument('-seas', '--seas', action='store_true', dest='seasons', help='activates seasons')
    parser.add_argument('-mf', '--min_food', dest='min_food_winter', type=float,
                        help='[0,1] Minimal amount of food in winter relative to max_food (food_num)')
    parser.add_argument('-aseas_len', dest='abrupt_seasons_len', type=int,
                        help='''Abrupt seasons - changes between summer and winter. Argument defines length of each 
                        season in generations (int). When 0 abrupt seasons are switched off''')
    parser.add_argument('-rand_seas', dest='random_food_seasons', action='store_true', help='''Activates random food 
                        seasons, meaning that every generation a random amount of food is spawned''')
    parser.add_argument('-rand_seas_lim', dest='rand_food_season_limits', nargs='+', type=int, help='''Expacts blank
                        seperated list, which gives limits for random food seasons: [lower_limit upper_limit]''')
    parser.add_argument('-ypi', dest='years_per_iteration', type=float,
                        help='''Number of years per generation when seasons is activated. When <0 one year is longer 
                        than an iteration''')
    parser.add_argument('-ser', '--ser', action='store_true', dest='server_mode',
                        help='''Activates server mode. Certain animation settings are adjusted for linux server''')
    parser.add_argument('-lap', '--lap', action='store_true', dest='laptop_mode',
                        help='''Activates laptop mode. Certain animation settings are adjusted for laptop''')
    parser.add_argument('-cs', '--cospeed', dest='cost_speed', type=float,
                        help='FLOAT [0,1] energy cost of speed as a factor of speed (linear function)')
    parser.add_argument('-ie', '--init_en', dest='init_energy', type=float,
                        help='initial energy at beginning of each generation in energy model')
    parser.add_argument('-fe', '--food_energy', type=float, dest ='food_energy',
                        help='Amount of energy, that individual gets from eating food particle')
    parser.add_argument('-ne', '--no_energy', dest='no_ener_mod', action='store_true',
                        help='Switch off energy model and instead optimize for maximal number of foods eaten')
    parser.add_argument('-vma', '--v_max', dest='v_max', type=float, help='Max speed of agends')
    parser.add_argument('-vmi', '--v_min', dest='v_min', type=float,
                        help='''Min speed of agents. Up until this speed agents do not use energy for movement when 
                        energy model is switched on''')
    parser.add_argument('-ama', '--a_max', dest='a_max', type=float, help='Maximal acceleration of an agent')
    parser.add_argument('-cfg', '--chg_food_gen', dest='chg_food_gen', nargs='+', type=int,
                        help='''Expects a blank separated list of len 2 X Y. At generation X change num_food to Y
                        num_food: maximal number of food when seasons active''')
    parser.add_argument('-tt', '--thermal', dest='thermal_time', type=int,
                        help='Number of thermal steps in each ising network')
    parser.add_argument('-n', '--name', dest='savename', help='Optional name for the folder')
    parser.add_argument('-vmo', '--v_motor', action='store_false', dest='acc_motor',
                        help='Activates speed as motor neuron output instead of acceleration')
    parser.add_argument('-ref', '--ref_plt', dest='refresh_plot', type=int, help='''Refreshes plot every nth generation
                        , so results can be analysed during run. If 0 no refresh.''')
    parser.add_argument('-dream_c', '--dream_heat_cap_gen', dest='dream_heat_capacity', type=int,
                        help='''Every nth generation that dream heat capacity is calculated and plotted. 
                        If 0 dream heat capacity is never calculated and plotted''')
    parser.add_argument('-rec_c', '--recorded_heat_cap_gen', dest='recorded_heat_capacity', type=int, help='''
                        Every nth generation that recorded heat capacity is calculated and plotted. 
                        If 0 dream heat capacity is never calculated and plotted. In the recorded heat capacity sensor 
                        input values are recorded during the simulation and subsequently used to calculate heat cap.''')
    parser.add_argument('-no_min_init', dest='minimal_energy_initializatin_heat_cap', action='store_false',
                        help='Deactivates brute force minimal energy initialization in recorded heat capacity. Deactivating'
                             'this leads to outliers in the heat capacity plots')
    parser.add_argument('-c_props', dest='heat_capacity_props', type=int, nargs='+', help='''Properties of dream and recorded heat 
                        capacity calculation. As blank spaced integer list: R, thermal_time, beta_low, beta_high, num_betas, y_lim_high 
                        R = number of repititions (with newly initialized sensor values) thermal_time = number of thermal iterations used
                        for energy measurements  beta_low = log lower bound for x axis  beta_num = number of betas, 
                        corresponds on resolution of plot  beta_high = log upper bound  y_lim_high = upper bound for plot/100''')
    parser.add_argument('-nat_c', '--natural_heat_cap_gen', dest='natural_heat_capacity_Nth_gen', help='''natural heat capacity 
                        is calculated for every nth generation, if 0 no heat capacity is calculated''', type=int)
    parser.add_argument('-nat_c_props', '--nat_heat_cap_beta_props', dest='natural_heat_capacity_beta_fac_props', nargs='+',
                        type=int, help='''natural heat capacity beta factor distribution properties, last three linspace
                        arguments 10 ** np.linspace(low_lim, high_lim, num_betas) expects blank seperated list len 3''')
    parser.add_argument('-c', '--cores', dest='cores', type=int,
                        help='Amount of cores available for heat capacity calculations. If 0 no heat cap calculations are done')
    parser.add_argument('-plot_c', dest='plot_heat_cap', action='store_true', help='''Plots heat capacity data right after
                        it was calulacted, only works with combination with rec_c''')
    parser.add_argument('-rand_ts', dest='random_time_steps', action='store_true', help='Activate random time steps every generation')
    parser.add_argument('-rand_ts_lim', dest='random_time_step_limits', nargs='+', type=int,
                        help='Expects blank seperated list X Y, where X is the lower and Y the upper limit of the '
                             'uniform random distribution, which is used for random time steps')
    parser.add_argument('-rand_ts_power', dest='random_time_steps_power_law', action='store_true',
                        help='Activate random time steps sampled from power law distribution')
    parser.add_argument('-rand_ts_power_lim', dest='random_time_steps_power_law_limits', nargs='+', type=int,
                        help='Limits of power law distribution, expects blank seperated list low_lim high_lim a'
                             'a is a parameter of the probability distribution. The larger a, the more likely are'
                             'small values')
    parser.add_argument('-spec', '--speciation', dest='speciation', action='store_true', help='Activates speciation in evolutionary algorithm')
    parser.add_argument('-delta', dest='delta_threshold_speciation', type=float, help='Threshold at which genetic distance '
                                                                                      'species are seperated')
    parser.add_argument('-ccc', dest='shared_fitness_constants', nargs='+', type=float,
                        help='Constants c_top, c_weight, c_beta in genetic distance function, that determine which '
                             '"weight" topology differences, weight differences and beta differences have'
                        )
    parser.add_argument('-mutdup', dest='mutationRateDup', type=float,
                        help='Mutation rate for the fittest individuals that have been duplicated (fittest individuals'
                             'are first copied into new generation, then again duplicated. The duplicated are mutated with'
                             'this mutation rate')
    parser.add_argument('-iso', dest='isolated_populations', action='store_true',
                        help='This command only works for "train_isolated_populations_different_betas.py" and "evolve_two_simulations_together.py'
                             'but not with "train.py". If this is active all different initial populations are evolved'
                             'isolated from each other, meaning that they live in the same 2D environment but do not compete'
                             'in the Evolutionary Algorithm')
    parser.add_argument('-subfolder', dest='save_subfolder', help='Subfolder, that the simulation is saved in')
    parser.add_argument('-energies', dest='save_energies_velocities_gens', nargs='+', type=int, help='''Expects blank seperated list of 
                        generation numbers, that energies and velocities dhall be saved for. This means, that the velocity
                        and energy of each organism is saved for each time_step''')
    parser.add_argument('-no_energies_last_gen', dest='save_energies_velocities_last_gen', action='store_false',
                        help='By default energies and verlocities during the lifetime of each organism are'
                             'saved for the last generation of the simulation. This argument can switch that off.')
    parser.add_argument('-compress', dest='compress_save_isings', action='store_true',
                        help='Compress pickle files when saving ising objects, reduces occupied diskspace by 2000 percent '
                             'but increases loading times')
    parser.add_argument('-v_eat_max', dest='max_speed_eat', type=float, help='Max speed that organisms can go when they eat.'
                                                                         'If not used, this feature is not active')
    parser.add_argument('-no_commands', dest='commands_in_folder_name', action='store_false', help='Commands except for '
                                                                                                   '-n (name of simualtion) '
                                                                                                   'are not saved in folder name')


    #-n does not do anything in the code as input arguments already define name of folder. Practical nonetheless.

    parser.set_defaults(save_data=True, plot=False, iterations=2000, time_steps=2000, plot_gens=[], fps=20,
                        loadfile='', loaditer=1999, pop_size=50, food_num=100, init_beta=1.0, seasons=False,
                        server_mode=False, cost_speed=0.05, v_max=999.0, v_min=0.05, sig_beta=0.02, no_mut_beta=False,
                        init_energy=2, food_energy=1, no_ener_mod=False, plot_pipeline=True, chg_food_gen=None,
                        years_per_iteration=1, min_food_winter=0.1, thermal_time=10, diff_init_betas=None, acc_motor=True,
                        a_max=0.05, refresh_plot=0, dream_heat_capacity=0, laptop_mode=False, natural_heat_capacity_Nth_gen=0,
                        natural_heat_capacity_beta_fac_props=[-1, 1, 100], recorded_heat_capacity=0, abrupt_seasons_len=0, cores=3,
                        switch_off_evolution=False, fading_traces_animation=True, random_time_steps=False,
                        random_time_step_limits=[100, 8000], heat_capacity_props=[100, 10, -2, 2, 100, 40], speciation=False,
                        delta_threshold_speciation=1, shared_fitness_constants=[1, 1, 1], mutationRateDup=0.1,
                        isolated_populations=False, beta_jump_mutations=False, animation_dpi=150,
                        random_food_seasons=False, rand_food_season_limits=[1, 199], save_subfolder='',
                        save_energies_velocities_gens=None, save_energies_velocities_last_gen=True, random_time_steps_power_law=False,
                        random_time_steps_power_law_limits=[100, 1000000, 700], num_neurons=12, compress_save_isings= False, max_speed_eat=None,
                        beta_linspace=None, change_beta_loaded_simulation=None, commands_in_folder_name=True, plot_heat_cap=False,
                        minimal_energy_initializatin_heat_cap=True, beta_linspace_within_sim=False)
    args = parser.parse_args()
    return args




# --- MAIN ---------------------------------------------------------------------+

def run(settings, Iterations):
    size = settings['size']
    nSensors = settings['nSensors']
    nMotors = settings['nMotors']
    # LOAD ISING CORRELATIONS
    # filename = 'correlations-ising2D-size400.npy'
    filename2 = 'correlations-ising-generalized-size83.npy'
    # This adds 13 MB to settings.pickle and we don't need it
    # settings['Cdist'] = np.load(filename2)

    # --- POPULATE THE ENVIRONMENT WITH FOOD ---------------+
    foods = []
    for i in range(0, settings['food_num']):
        foods.append(food(settings))

    # Food is only created uniformly distributed at the very beginning.
    # For a new iteration the placement of the food is kept.

    if settings['beta_linspace'] is not None:
        settings['init_beta'] = beta_linspace(settings)

    # --- POPULATE THE ENVIRONMENT WITH ORGANISMS ----------+
    try:
        set_isings = settings['set_isings']
    except KeyError:
        set_isings = None

    try:
        change_beta_loaded_simulation = settings['change_beta_loaded_simulation']
    except KeyError:
        change_beta_loaded_simulation = None

    if set_isings is not None:
        isings = settings['set_isings']
    elif settings['LoadIsings']:
        if not settings['switch_seasons_repeat_pipeline']:
            loadfile = 'save/' + settings['loadfile'] + '/isings/gen[' + str(settings['iter']) + ']-isings.pickle'
        else:
            loadfile = 'save/' + settings['loadfile'] + '/isings/gen[' + str(settings['iter']) + ']-isings.pickle'

        startstr = 'Loading simulation:' + loadfile + ' (' + str(settings['TimeSteps']) + \
                   ' timesteps) x (' + str(Iterations) + ' iterations)'

        if not settings['switch_seasons_repeat_pipeline']:
            prev_settings = load_settings(settings['loadfile'])

            #pop size of current simulation is taken from loaded simulation
            settings['pop_size'] = prev_settings['pop_size']
        print(startstr)

        try:
            file = open(loadfile, 'rb')
            isings = pickle.load(file)
            file.close()
        except FileNotFoundError:
            # Looking for compressed ising file in case normal pickle file is not found
            isings = decompress_pickle(loadfile)

        if settings['speciation']:
            for I in isings:
                I.species = 0
                I.shared_fitness = 0
        if change_beta_loaded_simulation is not None:
            for I in isings:
                I.Beta = change_beta_loaded_simulation


    else:
        startstr = 'Starting simulation: (' + str(settings['TimeSteps']) + \
                   ' timesteps) x (' + str(Iterations) + ' iterations)'
        print(startstr)
        isings = []
        for i in range(0, settings['pop_size']):
            isings.append(ising(settings, size, nSensors, nMotors, name='gen[0]-org[' + str(i) + ']'))

        if settings['beta_linspace_within_sim']:
            beta_linspace_within_sim(isings, settings)
        

    # --- CYCLE THROUGH EACH GENERATION --------------------+
    # Choose between CriticalLearning (which has both inverse-ising and GA with toggle)
    # or EvolutionLearning which is only GA. The functions are fairly similar, should find a
    # better way to call them than this.
    # ------------------------------------------------------+
    
    #No critical learning:
    # CriticalLearning(isings, foods, settings, Iterations)

    sim_name, not_used_isings = EvolutionLearning(isings, foods, settings, Iterations)

    return sim_name


def beta_linspace(settings):
    '''
    settings['beta_linspace'] : index, loglow, loghigh, number
    Those commands will be read by np.linspace, where index gives gives the desired position of the beta value
    of the array created by linspace. loglow and loghigh give the log thresholds of the beta value.
    (probably you want -1 and 1). Number gives the number of entries of the linspace array.
    '''
    index, loglow, loghigh, number = settings['beta_linspace']
    index = int(index) - 1
    number = int(number)
    beta_exp = np.linspace(loglow, loghigh, number)[index]
    beta = 10 ** beta_exp
    return beta


def beta_linspace_within_sim(isings, settings):
    linspace_exponents = np.linspace(-1, 1, settings['pop_size'])
    linspace_betas = [10**expo for expo in linspace_exponents]
    for I, new_beta in zip(isings, linspace_betas):
        I.Beta = new_beta



# --- RUN ----------------------------------------------------------------------+

if __name__ == '__main__':
    settings, Iterations = create_settings()
    t1 = time.time()
    sim_name = run(settings, Iterations)
    t2 = time.time()
    print('total time:', t2-t1)
    if settings['save_data'] and settings['plot_pipeline']:
        automatic_plotting.main(settings['save_subfolder'] + '/' + sim_name)




# --- END ----------------------------------------------------------------------+
