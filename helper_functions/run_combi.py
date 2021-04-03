import copy

'''
This class is used for switch_season_repeat_pipeline and switch_season_repeat_plotting
The class has to be in a separate file in order for the code to work properly
(otherwise the pipeline has to import the plotting function and the plotting function the pipeline, which obviously does
 not work)
'''


class RunCombi:
    def __init__(self, settings, food, beta, same_repeat, tot_same_repeats, food_summer, food_winter):
        '''
        This Class includes the properties of a certain simulation run
        '''
        settings = copy.deepcopy(settings)
        settings['food_num'] = food
        settings['init_beta'] = beta
        self.settings = settings

        if food == food_winter:
            season_name = 'winter'
        elif food == food_summer:
            season_name = 'summer'
        else:
            raise Exception('''In the current implementation of pipeline food_num has to be either 10 or 100 
            (winter and summer)''')

        # This defines the same of the folder, that the run is saved in
        subfolder = 'b{}_{}_{}'.format(beta, season_name, same_repeat)
        self.subfolder = subfolder
        self.food = food
        self.beta = beta
        self.season = season_name
        self.same_repeat = same_repeat
        self.tot_same_repeats = tot_same_repeats
