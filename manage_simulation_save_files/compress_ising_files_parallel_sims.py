
from helper_functions.automatic_plot_helper import compress_isings_in_parallel_simulations

if __name__ == '__main__':
    folder_names = ['sim-20201102-220107_parallel_b1_rand_seas_g4000_t2000_fixed_250_foods_compressed', 'sim-20201102-220135_parallel_b10_rand_seas_g4000_t2000_fixed_250_foods_compressed']
    for folder_name in folder_names:
        compress_isings_in_parallel_simulations(folder_name)