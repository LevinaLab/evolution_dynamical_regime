from shutil import copyfile
import numpy as np
from helper_functions.automatic_plot_helper import detect_all_isings
import sys
from os import listdir
from os.path import isfile, join


def merge(first_sim ,second_sim):
    '''Copies the ising files from second_sim into first_sim.
    The names of the ising files of the second_sim are renamed
     counting upwards from the highest generation of first_sim'''
    first_folder = 'save/{}/isings/'.format(first_sim)
    second_folder = 'save/{}/isings/'.format(second_sim)
    max_gen_first = np.max(detect_all_isings(first_sim))
    all_isings_second = [f for f in listdir(second_folder)
                         if isfile(join(second_folder, f))
                         and f.endswith('isings.pickle')]
    all_isings_second_new = []

    for name in all_isings_second:
        i_begin = name.find('[') + 1
        i_end = name.find(']')
        curr_gen = int(name[i_begin:i_end])
        new_gen = curr_gen + max_gen_first +1
        new_name = name[:i_begin] + str(new_gen) + name[i_end:]
        all_isings_second_new.append(new_name)

    for old_second, new_second in zip(all_isings_second, all_isings_second_new):
        copyfile(second_folder + old_second, first_folder + new_second)

if __name__ == '__main__':
    first_sim = sys.argv[1]
    second_sim = sys.argv[2]
    merge(first_sim, second_sim)



