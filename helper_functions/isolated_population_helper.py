import operator
def fittest_in_isolated_populations(isings_list_dict, fittest_percent_of_isings = 0.4):
    '''

    @param fittest_percent_of_isings: Percent of fittest isings that are chosen
    @return: isings_list_dict with only fittest individuals of each population in each generation
    '''
    fittest_isings_list_dict = {}
    for pop_name in isings_list_dict:
        isings_list = isings_list_dict[pop_name]
        fittest_isings_list = []
        for isings in isings_list:
            choose_fittest = int(len(isings) * fittest_percent_of_isings)
            fittest_isings = sorted(isings, key=operator.attrgetter('avg_energy'), reverse=True)[:choose_fittest]
            fittest_isings_list.append(fittest_isings)
        fittest_isings_list_dict[pop_name] = fittest_isings_list

    return fittest_isings_list_dict



def seperate_isolated_populations(isings_list):
    '''
    Sorts all isngs objects in ising lists to a dict of ising_lists that only contain one isolated_population

    @return: A dict of isings_lists, with one isings_list for each isolated_population
    '''
    iso_pop_names = set()
    for isings in isings_list:
        for I in isings:
            iso_pop_names.add(I.isolated_population)

    iso_pops_dict_isings_list = {}
    for i, isings in enumerate(isings_list):
        iso_pops_dict_isings = {}
        for iso_pop_name in iso_pop_names:
            curr_isolated_isings = []
            for I in isings:
                if I.isolated_population == iso_pop_name:
                    curr_isolated_isings.append(I)
            iso_pops_dict_isings[iso_pop_name] = curr_isolated_isings

        if i == 0:
            for iso_pop_name in iso_pop_names:
                iso_pops_dict_isings_list[iso_pop_name] = [iso_pops_dict_isings[iso_pop_name]]
        else:
            for iso_pop_name in iso_pop_names:
                iso_pops_dict_isings_list[iso_pop_name].append(iso_pops_dict_isings[iso_pop_name])

    return iso_pops_dict_isings_list


# Accidentally did this twice...
# def seperate_isings_isolated_populations(isings):
#     '''
#     Seperates list of ising objects into several lists, where each lsit includes all ising objects
#     with other ising objects that they share the isolated population with.
#     isolated population is introduced when using command -iso in train_isolated_populations_different_betas.py or evolve_two_simulations
#     together.py
#     @param isings:
#     @return [isings_population1, isings_population2, ...]:
#     '''
#     # different_iso_populations = {}
#     different_iso_populations = {I.isolated_population for I in isings}
#     seperated_isings = []
#     for isolated_population_name in different_iso_populations:
#         curr_isings_pop = []
#         for I in isings:
#             if I.isolated_population == isolated_population_name:
#                 curr_isings_pop.append(I)
#         seperated_isings.append(curr_isings_pop)
#     return seperated_isings