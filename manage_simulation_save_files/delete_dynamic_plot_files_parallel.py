from helper_functions.automatic_plot_helper import subfolders_in_folder
import os
import shutil


def delete_main(delete_settings):
    parallel_dirs = subfolders_in_folder(delete_settings['delete_in_dir'], ['sim', '_parallel'])
    for parallel_dir in parallel_dirs:
        sim_dirs = subfolders_in_folder(parallel_dir, ['sim'])
        for sim_dir in sim_dirs:
            repeated_dir = '{}/{}'.format(sim_dir, 'repeated_generations')
            try:
                delete(repeated_dir)
            except FileNotFoundError:
                pass


            # files_to_delete = [f for f in listdir(repeated_dir) if isfile(join(repeated_dir, f)) and f.endswith('2000')]


            # dynamic_range_dirs = all_folders_in_dir_with(repeated_dir, '')
            # for dynamic_range_dir in dynamic_range_dirs:
            #     delete(dynamic_range_dir)


def delete(dir):
    delete_file_dirs = []
    for folder_name in os.listdir(dir):
        folder_dir = os.path.join(dir, folder_name)
        # !!! THIS DEFINES THE DELETION CRITERIUM!!!"
        # if folder_name.endswith('gen_2000') or (folder_name.endswith('2000') and '3rd_trydynami' in folder_name):
        if 'load_gen_3999_with_energies_saved_' in folder_name:
            print('Deleting {}'.format(folder_dir))
            # This is the deletion command:
            shutil.rmtree(folder_dir)



if __name__ == '__main__':
    delete_settings = {}
    delete_settings['delete_in_dir'] = 'save/'
    # delete_settings['delete_all_dynamic_range_folders_that_include']
    delete_main(delete_settings)