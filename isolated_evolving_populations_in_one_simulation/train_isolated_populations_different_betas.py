from train import create_settings
from embodied_ising import ising
from embodied_ising import EvolutionLearning
from embodied_ising import food
import numpy as np
import time
import automatic_plotting

def run(settings, Iterations):



    size = settings['size']
    nSensors = settings['nSensors']
    nMotors = settings['nMotors']
    # LOAD ISING CORRELATIONS
    # filename = 'correlations-ising2D-size400.npy'
    filename2 = 'correlations-ising-generalized-size83.npy'
    settings['Cdist'] = np.load(filename2)

    # --- POPULATE THE ENVIRONMENT WITH FOOD ---------------+
    foods = []
    for i in range(0, settings['food_num']):
        foods.append(food(settings))

    # Food is only created uniformly distributed at the very beginning.
    # For a new iteration the placement of the food is kept.


    # --- POPULATE THE ENVIRONMENT WITH ORGANISMS ----------+

    startstr = 'Starting simulation: (' + str(settings['TimeSteps']) + \
               ' timesteps) x (' + str(Iterations) + ' iterations)'
    print(startstr)
    isings = []
    for i in range(0, settings['pop_size']):
        isings.append(ising(settings, size, nSensors, nMotors, name='gen[0]-org[' + str(i) + ']'))

    # ------- Reinitialize isings --------

    for i, I in enumerate(isings):
        if i < 25:
            I.species = 0
            I.Beta = 1
            if settings['isolated_populations']:
                I.isolated_population = 0
        else:
            I.species = 1
            I.Beta = 10
            if settings['isolated_populations']:
                I.isolated_population = 1



    #No critical learning:
    # CriticalLearning(isings, foods, settings, Iterations)

    sim_name, not_used_isings = EvolutionLearning(isings, foods, settings, Iterations)

    return sim_name

# --- RUN ----------------------------------------------------------------------+

if __name__ == '__main__':
    settings, Iterations = create_settings()
    t1 = time.time()
    sim_name = run(settings, Iterations)
    t2 = time.time()
    print('total time:', t2-t1)
    if settings['save_data'] and settings['plot_pipeline']:
        automatic_plotting.main(sim_name)