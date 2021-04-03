import numpy as np
import copy

def speciation(isings_old, isings_new, max_species_num_ever, settings):
    repres_inds_all_species = pick_rand_inds_from_prev_gen(isings_old)
    for I_new in isings_new:
        I_new.species = None

    # Merging those species, whose representatives are compatible:
    combination_tuples_repres_inds = combination_tuple(repres_inds_all_species, repres_inds_all_species)
    for I_old_species_tuple in combination_tuples_repres_inds:
        dist = genetic_distance(I_old_species_tuple[0], I_old_species_tuple[1], settings)
        if dist < settings['delta_threshold_speciation']:
            I_old_species1, I_old_species2 = I_old_species_tuple
            #merged_species_name = '({}_{})'.format(I_old_species1.species, I_old_species2.species)
            merged_species_name = max_species_num_ever + 1
            max_species_num_ever = merged_species_name
            # Make one of the individuals representative for new species and delete the other one out of repres_inds_all_species
            # (If we wanted to be accurate we would have to draw a new representative out of merged species, but for
            # simplicity I'm not going to do that)
            I_old_species1.species = merged_species_name
            for I_old in repres_inds_all_species:
                if I_old == I_old_species2:
                    del I_old

    for I_old_species in repres_inds_all_species:
        for I_new in isings_new:
            # Assign species of representative individual from old generation to current individual if genetic distance
            # is below threshold and individual has not already been assigned a species to
            if (genetic_distance(I_new, I_old_species, settings) < settings['delta_threshold_speciation']) and (I_new.species is None):
                I_new.species = I_old_species.species

    # Creating new species for those individuals, that were not compatible to any of the existing species
    # TODO: When new species are created what about two species, that
    all_species_names = get_all_species_names(isings_new)

    for I_new in isings_new:
        # Those species that could not be assigned to existing species have NONE assigned and get new species
        # What about two individuals that have low genetic distance to each other, but both are above delta away
        # any other old representative? They both would eb assigned a new species to, which however in next generation
        # be merged...
        if I_new.species is None:
            # Get maximal number that is in current species names and add 1
            # In case none of the new isings could be assigned to old species (all_species_names is empty in that case)
            # caount again from 0
            if all_species_names == []:
                max_species_number = str(0)
            else:
                #max_species_number = str(max([max([int(s) for s in species_name.split() if s.isdigit()]) for species_name in all_species_names])+1)
                #max_species_number = str(max([max(re.findall(r'\d+', species_name)) for species_name in all_species_names])+1)
                all_species_names_string = ''
                for species_name in all_species_names:
                    all_species_names_string += str(species_name)
                #max_species_number = max(re.findall(r'\d+', all_species_names_string))

            #new_species_name = str(int(max_species_number) + 1)
            new_species_name = max_species_num_ever + 1
            I_new.species = new_species_name
            max_species_num_ever = new_species_name
    return max_species_num_ever


def calculate_shared_fitness_continuous_species(isings_new, settings):
    for i, I_new in enumerate(isings_new):
        other_isings_below_threshold = 1
        for j, I_new_comp in enumerate(isings_new):
            dist = genetic_distance(I_new, I_new_comp, settings)
            if (dist < settings['delta_threshold_speciation']) and (i != j):
                other_isings_below_threshold +=1
        I_new.shared_fitness = I_new.avg_energy / other_isings_below_threshold


def calculate_shared_fitness(isings_new):
    '''
    Calculate fitness according to fitness sharing principle
    Fitness is divided by number of individuals in own species
    This has to be executed after speciation assigned species to isings_new
    '''
    # divide previous fitness by number of members in own species

    for i, I_new in enumerate(isings_new):
        members_own_species = 1
        for j, I_new_compare in enumerate(isings_new):
            if I_new.species == I_new_compare.species and j != i:
                members_own_species += 1
        I_new.shared_fitness = I_new.avg_energy / members_own_species


# --------Helper functions-----------

def genetic_distance(I1, I2, settings):
    '''
    Calculates genetic distance delta from slightly modified formula of delta formula from the NEAT Algorithm
    '''
    #TODO: Consider biases here, in case we use them!!
    c_top, c_weight, c_beta = settings['shared_fitness_constants']
    # TODO: Largest genome size should be all possible number of edges??
    if I1.size > I2.size:
        largest_genome_size = I1.size
    else:
        largest_genome_size = I2.size
    #beta_diff = np.abs(I1.Beta - I2.Beta)
    #Beta differences on log scale:
    beta_diff = np.abs(np.log(I1.Beta) - np.log(I2.Beta))
    beta_diff = np.abs(np.log(I1.Beta) - np.log(I2.Beta))

    delta = (c_top*topology_difference(I1.maskJ, I2.maskJ)/largest_genome_size) + \
            c_weight*weight_difference(I1, I2) + c_beta*beta_diff

    return delta


# Helper functions for genetic_distance():
def topology_difference(maskJ1, maskJ2):
    '''
    Calculates amount of different edges in both networks
    '''
    mask_xor = np.logical_xor(maskJ1, maskJ2)
    num_different_edges = mask_xor[mask_xor == True].size

    return num_different_edges


def weight_difference(I1, I2):
    '''
    Calculates average weight difference of shared edges
    '''
    maskJ_merged = np.logical_and(I1.maskJ, I2.maskJ)
    J1_fitted = set_J_0_where_mask_false(I1.J, maskJ_merged)
    J2_fitted = set_J_0_where_mask_false(I2.J, maskJ_merged)
    dist_matrix = np.abs(J1_fitted - J2_fitted)
    #num_values_bigger_0 = dist_matrix[dist_matrix != 0].size
    num_shared_edges = np.sum(maskJ_merged)
    if num_shared_edges == 0:
        average_dist = 1
    else:
        average_dist = np.sum(dist_matrix)/num_shared_edges

    return average_dist


def set_J_0_where_mask_false(J, maskJ):
    J_out = copy.deepcopy(J)
    for i in range(np.shape(J)[0]):
        for j in range(np.shape(J)[1]):
            if maskJ[i, j] == False:
                J_out[i, j] = 0
    return J_out


def get_all_species_names(isings):
    all_species_names = set()
    for I in isings:
        if not I.species is None:
            all_species_names.add(I.species)
    all_species_names = list(all_species_names)
    return all_species_names


def combination_tuple(arr1, arr2):
    combination_tuples = []
    for i, en1 in enumerate(arr1):
        for j, en2 in enumerate(arr2):
            if j < i:
                combination_tuples.append((en1, en2))
    return combination_tuples


# Helper functions vor speciation():
def pick_rand_inds_from_prev_gen(isings_old):
    '''
    Picks one random indivudual for every species from the last generation
    Each one represents one species in speciation(). Adapted from NEAT
    According to NEAT paper it would be more accurate to calculate average distance to all individuals in previous species,
    then assign it to species, that had lowest average distance. But this approach is sufficient according to paper and less time intensive
    I also think, that it adds some more randomness, which also ocurs in natural evolution
    However for our algorithm it might be necessary to prevent random species creation and merging?
    '''

    all_species_names = get_all_species_names(isings_old)

    rand_inds_all_species = []
    for species_num in all_species_names:
        all_inds_curr_species = []
        for I_old in isings_old:
            if I_old.species == species_num:
                all_inds_curr_species.append(I_old)
        random_ind_curr_species = np.random.choice(all_inds_curr_species, size=1)[0]
        rand_inds_all_species.append(random_ind_curr_species)

    return rand_inds_all_species
