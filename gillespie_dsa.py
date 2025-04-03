'''
Gillespie DSA for a stochastic compartmental SCIR epidemic, Individual-Based and Explicitly Spatial - rewrite
Multiple species compatible, and with radial control, with varying species-specific probabilities of detection.

A. Vargas Richards, Dec. 2024, Jan., Feb. 2025
'''

import numpy as np
import pandas as pd
import random, math
from scipy.spatial.distance import pdist, squareform
from collections import Counter 
import multiprocessing as mp 
import os, shutil
from numpy.random import seed
import landscape_generators
from pathlib import Path # for writing out epi_arrays 

[HOST_ID, SPS_ID, INFEC_STATUS, T_RATE, T_SC, T_CI, T_IR] = [0, 1, 2, 3, 4, 5, 6] # these are our indexing constants to prevent ambiguity and ensure rapid model modification if needed.

def epi_dir(results_dir):
    '''
    Checks if an appropriate directory to write out epi_array (all available information about the epidemic) and if not creates one along with guide.txt file for
    reproducibility

    '''
    dpath = Path(results_dir)
    try:
        dpath.mkdir()
        print(f"Directory '{dpath}' created successfully.")
    except FileExistsError:
        print(f"Directory '{dpath}' already exists.")
    return 

def write_results(epi_array, results_dir, replicate_number): 
    '''
    Writes out the epi_array to a desired `results_dir` in its own individual file
    ''' 
    np.savetxt(fname=f'{results_dir}/epi{replicate_number}.txt', X=epi_array, delimiter='\t') # should probably make this path crossplatform in future using pathlib
    print(f'Saved epi successfully as epi{replicate_number}.txt in directory {results_dir}')
    return

def read_results(results_dir):
    '''
    Read all results in a particular results_dir
    '''



def make_pandas(epi_array, col_names): # comparatively slow function, perhaps...
    '''
    Converts a numpy array to a pandas dataframe for better readability / printing
    a wrapper to pd.DataFrame
    '''
    return pd.DataFrame(data=epi_array, columns=col_names)

def located_df(epi_frame,locations ):
    '''
    Adds the location of hosts to the epi_frame.
    '''
    epiframe_located = pd.concat([epi_frame, locations], axis=1)
    return epiframe_located

def landscape_array(filename): 
    '''
    construct an array with three columns: x, y coordinate and species identity. 
    The host_id is implicit .
    '''
    lscape = pd.read_csv(filename, header=None)
    try:
        lscape.columns = ["pos_X", "pos_Y","sps_id"]
    except: # if there are no species column then add one
        lscape["sps_id"] = 0
        lscape.columns = ["pos_X", "pos_Y","sps_id"]
    return lscape

def cauchy_thick (distance_matrix, alpha):
    '''
    thick-tailed cauchy dispersal kernel parameterised by (alpha = scale parameter)
    '''
    return 1/(1 + np.power(np.divide(distance_matrix, alpha), 2))

def exponential_kernel(distance_matrix, alpha): 
    '''
    Thin-tailed exponential dispersal kernel parameterised by (alpha= scale parameter)
    '''
    return np.exp(np.divide(-distance_matrix, alpha))

def flat_kernel(distance_matrix, beta): 
    '''
    The flat kernel (all ones), multiplied by beta for the symmetric model.
    Useful for debugging and comparing to the non-spatial model
    '''
    ones_matrix = np.ones(np.shape(distance_matrix))
    # return np.multiply(ones_matrix, beta)
    return ones_matrix

def make_kernel(system_state, landscape, kernel_function, scale_parameter, species_parameters): 
    '''
    Function which constructs a multispecies transmission kernel.
    nu and phi multiply to give beta for the single species case (nu*phi = beta)
    for multispecies transmission, the 
 
    Arguments
    ---------

    `landscape` gives the species identity of each host and its position

    n.b.
    --------
    In the transmission kernel values can be dependent 
    on the composition and arrangement of host species (since transmission is asymmetric)
    '''
    [[nu0, phi0, sigma0, gamma0],[nu1, phi1, sigma1, gamma1]] = species_parameters

    host_positions = landscape[["pos_X", "pos_Y"]]
    system_state[:, SPS_ID] = landscape["sps_id"] # 
    dmat = squareform(pdist(host_positions)) # dmat is the inter-host distance matrix
    N = len(dmat) # this will be the number of hosts, since all hosts have a position
    kernel = kernel_function(dmat, scale_parameter) # preliminary kernel, w/o species factors.
    for host in range(N): # for each host
        if system_state[host, SPS_ID] == 0: # for the first type of host
            nu = nu0; phi = phi0 # define the infectivity to others and susceptibility to infection from others respectively as the species-specific value
        elif system_state[host, SPS_ID] == 1: # for the second type of host do the same
            nu = nu1; phi = phi1
        else: # throw an error because something is wrong
            print(f'{system_state[host, SPS_ID]} is unrecognised species identity.')
            return ValueError # this will ensure the program halts 
        kernel[host, :] *= nu # the first index is the row
        kernel[:, host] *= phi # now the columns, any one of which represents the
    #fig, ax = plt.subplots() 
    #sla = ax.imshow(kernel) 
    #bar = plt.colorbar(sla) 
    #plt.xlabel('Individual Number') 
    #plt.ylabel('Individual Number') 
    #bar.set_label('K(d_ij)') 12
    #plt.title(f"Transmission kernel: function {kernel_function}\nSpecies Parameters{species_parameters}\n")
    #plt.savefig("graphic_kernel.svg")
    #plt.show() 

    #plt.imshow(kernel)
   
    return [kernel, system_state, dmat] # system state now has the updated 

def neighbour_radius(dmat, control_radius): 
    '''
    Produce an unweighted symmetric graph for use in constant radial control: a(ij) = 1 if d(ij) < d_cut.
    Operates on squareform(pdist(distances))
    '''
    adj = lambda dij: 1 if dij <= control_radius else 0
    vect_adj = np.vectorize(adj)
    cutoff_matrix = vect_adj(dmat)
    return cutoff_matrix

def setup_system(initial_condition, landscape, **kwargs): 
    ''' 
    called to set up the epi_array; does not calulate rates but sets the states to the correct values
    also manages the landscape set up in multispecies cases.

    Arguments
    ---------

    `initial condition`: vector of length 4.

    `landscape`: either  a file with comma / tab delimited values or a function for generating and returning a relevant landscape

    Returns
    ------
    '''
    assert (len(initial_condition) == 4),"Initial condition of incorrect length!\nSpecify as [S0,C0,I0,R0]"

    if type(landscape) == str:
        print(f'assuming landscape {landscape} refers to a file containing host species and locations')
        host_landscape = landscape_array(landscape)
    elif isinstance(landscape, pd.DataFrame): # if the landscape has been passed as a pandas dataframe we don't need to do anything to it 
        #print(f'data frame passed as landscape')
        host_landscape = landscape
    else:
        print(f'assuming landscape {landscape} refers to a generative model.') # the landscape presumably points to a generation method
        host_landscape = landscape(**kwargs)
        host_landscape.columns = ["pos_X","pos_Y", "sps_id"]

    cols = [HOST_ID, SPS_ID, INFEC_STATUS, T_RATE, T_SC, T_CI, T_IR] = [0, 1, 2, 3, 4, 5, 6]
    [STATUS_S, STATUS_C, STATUS_I, STATUS_R] = [0,1,2,3]
    [S0, C0, I0, R0] = initial_condition
    #print(f'Initial condition set as \n--------------\nS0 = {S0}\nC0 = {C0}\nI0={I0}\nRem0 = {R0}\n-----------------')
    N = S0 + C0 + I0 + R0 # total number of hosts held constant
    epi_array = np.empty([N, len(cols)]) # this array will contain all the available data about the sample epidemic path from which S(t), ..., R(t) can be reconstructed, 0 <= t < t_end
    epi_array[:, HOST_ID] = np.arange(N)
    epi_array[:, SPS_ID] = pd.Series(host_landscape["sps_id"]).to_numpy() # altered for a multi-species case
    #print(f'species total is {Counter(epi_array[:, SPS_ID])}')
    epi_array[:, INFEC_STATUS] = STATUS_S # initialise all hosts to susceptible by default
    epi_array[:, T_SC: (T_IR + 1)] = -1 # -1 to indicate that an event has not occurred
    cryptics = np.array(random.sample(epi_array[:,HOST_ID].tolist(), C0)).astype(int) # randomly sample w/o repetition the host IDs for the cryptics.
    infecteds = np.array(random.sample(epi_array[:,HOST_ID].tolist(), I0)).astype(int)
    # print(f'cryptics are {cryptics} infecteds are {infecteds} at time = 0')
    epi_array[cryptics, INFEC_STATUS] = STATUS_C # change some S -> C 
    epi_array[cryptics, T_SC] = 0
    epi_array[infecteds, INFEC_STATUS] = STATUS_I # change some S -> I if applicable
    epi_array[infecteds, T_CI] = 0
    return [epi_array, host_landscape]

def bound_floats(kernel_table): 
    '''
    compute the lower bound for the floating point numbers, this can be easily adjusted by the user
    '''
    min_kernelval = np.min(kernel_table)
    min_float = min_kernelval # heuristic can change later if desired
    # print(f'min_float is {min_float}')
    min_float = 1e-15
    #print(f'min float is {min_float}')
    return min_float

def correct_floats(matrix, min_bound): 
    '''
    Set floats smaller than the minimum calculated bound to zero, else do nothing, for each element of a matrix.
    '''
    correct_flt = lambda number: float(0) if np.abs(number) < min_bound else number
    vect = np.vectorize(correct_flt)
    corrected_mat = vect(matrix)
    #print(f'corrected matrix is \n{corrected_mat}')
    return corrected_mat
    
def _row_rate(row, system_state,transmission_kernel, sigmas, gammas): 
    '''
    calculate the transition rate for the particular host_id.
    This function is a helper function, called in a vectorised manner by `compute_rates`
    '''
    [HOST_ID, SPS_ID, INFEC_STATUS, T_RATE, T_SC, T_CI, T_IR] = [0, 1, 2, 3, 4, 5, 6]
    [STATUS_S, STATUS_C, STATUS_I, STATUS_R] = [0,1,2,3]

    infectious_hosts = np.where((system_state[:, INFEC_STATUS] == STATUS_C) | (system_state[:, INFEC_STATUS] == STATUS_I))[0]    
    #print(f'infectious hosts are {infectious_hosts}')
    species_identity = int(row[SPS_ID])
    host_identity = int(row[HOST_ID])

    if row[INFEC_STATUS] == STATUS_S:
        potential_pressures = transmission_kernel[infectious_hosts, host_identity] # we first identify the column containing the potential infection pressures towards our host of interest
        rate = math.fsum(potential_pressures) 

    elif row[INFEC_STATUS] == STATUS_C:
        #print(f'sigma rate for {species_identity} is {sigmas[species_identity] } ')
        rate = sigmas[species_identity]

    elif row[INFEC_STATUS] == STATUS_I:
        rate = gammas[species_identity]

    elif row[INFEC_STATUS] == STATUS_R:
        rate = 0
    else:
        print(f'***{row[INFEC_STATUS]} is an unrecognised infection status, exiting***')
        return ValueError
    
    row[T_RATE] = rate
    return row

def compute_rates(epi_state, transmission_kernel, sigmas, gammas, min_float): 
    '''
    compute rates from scratch: used as a checker /debug function and is called at the start of the simulation.
    '''
    new_state = np.apply_along_axis(func1d=_row_rate, axis=1,
                                    arr= epi_state, system_state = epi_state ,
                                    transmission_kernel=transmission_kernel,
                                    sigmas= sigmas, gammas= gammas)
    corrected_rates = correct_floats(new_state, min_float)
    return corrected_rates

def radial_control(epi_array, time, control_parameters, adjacency_matrices, transmission_kernel, sigmas, gammas, min_float, debug=False):
    """
    Implementation of constant radius of removal control. Optimized for better performance and clarity.


    Explanation of control parameter
    """
    STATUS_S, STATUS_C, STATUS_I, STATUS_R = 0, 1, 2, 3
    HOST_ID, SPS_ID, INFEC_STATUS, T_RATE, T_SC, T_CI, T_IR = 0, 1, 2, 3, 4, 5, 6
    SPECIES_0, SPECIES_1 = 0, 1

    adj_matA, adj_matB = adjacency_matrices
    survey_freq, frac_A, frac_B, prob_detec_A, prob_detec_B, radius_A, radius_B = control_parameters

    symptomatic_A = np.where((epi_array[:, INFEC_STATUS] == STATUS_I) & (epi_array[:, SPS_ID] == SPECIES_0))[0]
    symptomatic_B = np.where((epi_array[:, INFEC_STATUS] == STATUS_I) & (epi_array[:, SPS_ID] == SPECIES_1))[0]

    num_detected_A = np.random.binomial(int(len(symptomatic_A) * frac_A), prob_detec_A) # implementation of the frac_A and frac_B added here.
    num_detected_B = np.random.binomial(int(len(symptomatic_B) * frac_B), prob_detec_B) # detection implemented as binomial distribution random variables (ie Bernoulli trials)

    detected_A = np.random.choice(symptomatic_A, num_detected_A, replace=False) if num_detected_A > 0 else np.array([])
    detected_B = np.random.choice(symptomatic_B, num_detected_B, replace=False) if num_detected_B > 0 else np.array([])

    adjacents_A = np.where(adj_matA[:, detected_A].sum(axis=1) > 0)[0] if len(detected_A) > 0 else np.array([])
    adjacents_B = np.where(adj_matB[:, detected_B].sum(axis=1) > 0)[0] if len(detected_B) > 0 else np.array([])
    for_removal = np.unique(np.concatenate((adjacents_A, adjacents_B))).astype(int) 
    statuses = epi_array[for_removal, INFEC_STATUS]
    valid_mask = statuses != STATUS_R 
    for_removal = for_removal[valid_mask]  
    statuses = statuses[valid_mask]  
    if for_removal.size > 0:
        removable_CI = for_removal[(statuses == STATUS_I) | (statuses == STATUS_C)]
        removable_suscept = for_removal[statuses == STATUS_S]

        epi_array[removable_suscept, INFEC_STATUS] = STATUS_R
        epi_array[removable_suscept, T_IR] = time
        epi_array[removable_suscept, T_RATE] = 0

        if debug:
            print(f"For removal (S): {removable_suscept}")
            print(f"Removable hosts (C|I): {removable_CI}")

        epi_array[removable_CI, INFEC_STATUS] = STATUS_R
        epi_array[removable_CI, T_IR] = time
        epi_array[removable_CI, T_RATE] = 0

        susceptibles = np.where(epi_array[:, INFEC_STATUS] == STATUS_S)[0]
        if susceptibles.size > 0 and removable_CI.size > 0:
            transmission_sums = np.sum(transmission_kernel[removable_CI[:, None], susceptibles], axis=0)
            epi_array[susceptibles, T_RATE] -= transmission_sums

        epi_array = correct_floats(epi_array, min_float)
        return epi_array
    else:
        return epi_array


def compute_fsize(epi_array):
    '''
    Wrapper for '_compute_fsize' function which only takes a single array; this can return a list of final sizes when necessary
    '''
    fss = []
    if type(epi_array) == list:
        for epi in epi_array:
            fss.append(_compute_fsize(epi))
    else:
        return _compute_fsize(epi_array)
    return fss

def _compute_fsize(epi_array):
    '''
    Computes the final size of the epidemic as `N - num_susceptibles`
    '''

    [HOST_ID, SPS_ID, INFEC_STATUS, T_RATE, T_SC, T_CI, T_IR] = [0, 1, 2, 3, 4, 5, 6]
    [STATUS_S, STATUS_C, STATUS_I, STATUS_R] = [0,1,2,3]
        
    N = len(epi_array[:, INFEC_STATUS])
    #print(f'computed N as {N}')
    fdict = Counter(epi_array[:,INFEC_STATUS])
    fsize = N - fdict.get(STATUS_S, 0)
    return fsize

def gillespie(initial_condition, landscape, kernel_function, scale_parameter 
               , species_parameters, control_parameters, control_on, control_start, 
               return_fsize, return_audpc, return_simple_audpc,debug, write_files, reused_kernel, return_t50, write_out, **kwargs):
    '''
    ## Gillespie Direct Simulation of Spatially Explicit Individual-Based Stochastic Epidemic

    The main stochastic epidemic function with control, spatially explicit and tracks each individual. Usually called by one of the parallelisation wrapper functions. 

    Arguments
    ----
    `landscape`: 
    The landscape on which the epidemic is simulated. May refer to a generative model, but this is not advised for faster performance.

    `kernel_function`:
    either `cauchy_thick` or `exponential` dispersal kernel functions. 

    `reused_kernel`:
    For identical landscapes and transmission parameters, the transmission kernel lookup table need not be recalculated. Instead it is passed as `reused_kernel`
    but care must be taken to ensure this is a valid use. MUST be set = `None` if not desired.

    `return_audpc`: 
    if True, then the Area Under Disease Progress Curve (AUDPC) will be returned. `gillespie` also assumes that the T_end of the epi. is desired, so will return that too.

    The return value of the function will therefore be:
        [final_size, T_end, AUDPC]
    otherwise the function will simply return the final_size or the epi_array at the end of the epidemic....

    
    `return_t50`:
    if True, then the time at which half of all hosts are infected will be returned by the simulation.

    assert type(control_on) == bool, "specify true/false arg for control_on"
    assert type(write_files) == bool, "specify true/false arg for write_files"      
    assert type(return_fsize) == bool, "specify a true/false arg for return type"
    assert type(debug) == bool, "specify a true/false arg for debug mode"
    '''

    if write_files == True: # writing out files if desired for e.g., debugging
        dirname = './control_rep'
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        else:
            shutil.rmtree(dirname)          
            os.makedirs(dirname)

    rslts = [] # this array will hold the relevant results for the epi, incl. T_end, final size/ epidemic impact and the AUDPC

    [HOST_ID, SPS_ID, INFEC_STATUS, T_RATE, T_SC, T_CI, T_IR] = [0, 1, 2, 3, 4, 5, 6]
    [STATUS_S, STATUS_C, STATUS_I, STATUS_R] = [0,1,2,3]
    [SPECIES_0, SPECIES_1] = [0,1]
    [survey_freq, frac_A, frac_B, prob_detec_A, prob_detec_B, radius_A, radius_B] = control_parameters

    sigmas = species_parameters[:, 2]
    gammas = species_parameters[:, 3]

    if control_start == 'random':
        delta_0 = np.random.uniform() * survey_freq # draw the time of first control from a uniform distribution on [0, Delta]
    elif control_start == 'zero':
        delta_0 = 0
    else:
        print(f'provide a valid argument for the time of first control! Either "random" or "zero"')
        return ValueError
    
    control_times = np.arange(delta_0, survey_freq**3, step=survey_freq) # compute the times at which surveys are to be performed
    controls_performed, time = 0, 0
    control_at = [] # this records the times at which control was actually performed for debugging purposes

    if reused_kernel == None:
        # in this case, since initial infectives are being generated inside this process there is no risk of accidental copying of the initial cryptics across processes
        [epi_array, host_landscape] = setup_system(initial_condition, landscape=landscape) # assign the hosts to their initial states
        [transmission_kernel, epi_array, distance_matrix] = make_kernel(system_state=epi_array,
        landscape=host_landscape, kernel_function=kernel_function, scale_parameter=scale_parameter, 
        species_parameters=species_parameters) # we construct our transmission kernel with all transmission constants taken into calc.

    else: # (i.e.) kernel is supplied for this particular run and has been previously calculated
        [transmission_kernel, epi_array, distance_matrix] = reused_kernel # user has called kernel function outside of gillespieDSA
        epi_array[:,T_SC] = -1 # if events haven't happened they are assigned a negative time '-1' to prevent any ambiguity ...
        epi_array[:, INFEC_STATUS] = 0 # initially all hosts are susceptible
        [S0, C0, I0, R0] = initial_condition # unpack the initial condition
        cryptics = np.array(random.sample(epi_array[:,HOST_ID].tolist(), C0)).astype(int) # randomly sample w/o repetition the host IDs for the cryptics.
        infecteds = np.array(random.sample(epi_array[:,HOST_ID].tolist(), I0)).astype(int)
        # print(f'cryptics are {cryptics} infecteds are {infecteds} at time = 0')
        epi_array[cryptics, INFEC_STATUS] = STATUS_C # change some S -> C 
        epi_array[cryptics, T_SC] = 0 # by definition, these events happen at time = 0
        epi_array[infecteds, INFEC_STATUS] = STATUS_I # change some S -> I if applicable
        epi_array[infecteds, T_CI] = 0
        
    control_matA = neighbour_radius(dmat = distance_matrix, control_radius = radius_A) # construct the control matrix for each sps.
    control_matB = neighbour_radius(dmat = distance_matrix, control_radius = radius_B)
    # note there is potential to construct these matrices as sparse matrices which could speed up execution.
    # however, i didn't implement this here.

    min_float = bound_floats(transmission_kernel) # compute the min float
    epi_array = compute_rates(epi_array, transmission_kernel, sigmas, gammas, min_float) # compute the initial rates
    #print(f'minimum float is {min_float} for this epidemic')

    Nhosts = len(transmission_kernel)
    half_hosts = int(Nhosts/2) # will be used to calculate T(1/2)
    fdict = Counter(epi_array[:, INFEC_STATUS]) # basically, count up the number of different individuals in each state.
    num_cryptics = fdict.get(STATUS_C, 0) # importantly, assigns zero if there are no cryptics otherwise we'll get a None instead of zero which is bad!
    num_symptomatics = fdict.get(STATUS_I, 0)  
    num_susceptibles = fdict.get(STATUS_S, 0)
    epi_ongoing = True

    while epi_ongoing == True: # which incorporates a state-based termination condition rather than a rate-based one which would be more error-prone
        fdict = Counter(epi_array[:, INFEC_STATUS]) # basically, count up the number of different individuals in each state.
        num_cryptics = fdict.get(STATUS_C, 0) # importantly, assigns zero if there are no cryptics otherwise we'll get a None instead of zero which is bad!
        num_symptomatics = fdict.get(STATUS_I, 0)  
        num_susceptibles = fdict.get(STATUS_S, 0)

        if int(num_cryptics + num_symptomatics) == 0: # no active infections left so we should break out of the main simulation loop as nothing more can happen
            epi_ongoing = False
            break

        if control_on == False: # e.g., for a normalisation plot
            if int(num_susceptibles) == 0:
                epi_ongoing = False
        elif control_on != True:
            print(f'set control_on to boolean argument (NPr)')
            return TypeError
        
        epi_array = correct_floats(epi_array, min_float) # first make sure the floats are corrected
        rand1, rand2 = np.random.random(2) # generate the relevant rands for selecting the event and computing timestep
        if debug: # if debug mode is on we can compute the transition rates from scratch, but this takes a long time so not normally done for better performance. 
           epi_array = compute_rates(epi_state=epi_array, transmission_kernel=transmission_kernel, sigmas=sigmas, gammas=gammas, min_float=min_float) # compute from scratch each time if needed
        total_rate = np.sum(epi_array[:,T_RATE]) # total rate will in part determine the t_step
        fdict = Counter(epi_array[:, INFEC_STATUS]) # basically, count up the number of different individuals in each state.

        if total_rate < 0: 
            print(f'***epi failed, {Counter(epi_array[:, INFEC_STATUS])}, total_rate = {total_rate} [NPr]')
            return ValueError #
        
        elif total_rate == 0: # the epi has finished 
            continue

        if debug:
            old = epi_array
            epi_array = compute_rates(epi_array, transmission_kernel, sigmas, gammas, min_float)
            assert epi_array.all() == old.all(), "recalc changed "
        
        delta_t = (-1/total_rate) * math.log(rand1) # compute the timestep stochastically
        c_time = control_times[controls_performed]
        
        if time + delta_t > c_time and control_on == True:
            if int(num_cryptics + num_symptomatics) == 0:
                break
            # print(f'Performed control at  time {c_time}: {Counter(epi_array[:, INFEC_STATUS])}')
            epi_array = radial_control(epi_array, 
                                       time, control_parameters,
                                         [control_matA, control_matB], transmission_kernel, 
                                       sigmas, gammas, min_float, debug)
            controls_performed += 1 
            control_at.append(c_time) # note when we performed control
            time = c_time
            continue

        P = rand2 * total_rate # draw a second random float
        #print(f'total rate at time {time} = {total_rate}, P = {P}')
        cumulative_rates = np.cumsum(epi_array[:,T_RATE]) # ccalculates the sum of rates up to x event for each x
        selected_host = np.searchsorted(cumulative_rates, P) # select the relevant host based on P and cumulative rates
        host_species = int(epi_array[selected_host, SPS_ID]) # note the species identity of the selected host
        
        sigma = sigmas[host_species] # the sigma/ gamma may be species-specific so we need to calc. it here
        gamma = gammas[host_species] 
        #print(f'sigmas are {sigmas}')
        if epi_array[selected_host, INFEC_STATUS] == STATUS_S:
            epi_array[selected_host, INFEC_STATUS] = STATUS_C # make the host cryptically infected
            epi_array[selected_host, T_SC] = time + delta_t # the event occurs at current time PLUS TIMESTEP as time hasn't yet been incremented
            epi_array[selected_host, T_RATE] = sigma
            susceptibles = np.where(epi_array[:, INFEC_STATUS] == STATUS_S)[0]
            for S_host in susceptibles:
                epi_array[S_host, T_RATE] += transmission_kernel[selected_host, S_host] # add the infectious pressure from the new cryptic to the susceptibles
        
        elif epi_array[selected_host, INFEC_STATUS] == STATUS_C: # no nonlocal effects in this case under our current parameterisation
            epi_array[selected_host, INFEC_STATUS] = STATUS_I # make the host symptomatically infected 
            epi_array[selected_host, T_CI] = time + delta_t 
            epi_array[selected_host, T_RATE] = gamma
        
        elif epi_array[selected_host, INFEC_STATUS] == STATUS_I: # if we've drawn a currently symptomatically infected host it must be removed (this would imply that gamma > 0)
            epi_array[selected_host, INFEC_STATUS] = STATUS_R
            epi_array[selected_host, T_IR] = time + delta_t
            epi_array[selected_host, T_RATE] = 0
            susceptibles = np.where(epi_array[:, INFEC_STATUS] == STATUS_S)[0]
            for S_host in susceptibles: # now we subtract the infection pressure from this host on other susceptibles
                epi_array[S_host, T_RATE] -= transmission_kernel[selected_host, S_host]
        
        elif epi_array[selected_host, INFEC_STATUS]  == STATUS_R: # note that this shouldn't be triggered ordinarily            
            print(f'Removed event should never be possible: bug.')
            return ValueError # hence the error is returned to alert the user
        
        else:
            print(f'Unrecognised infection status {epi_array[selected_host, INFEC_STATUS]}')
            return ValueError
        
        if debug: # need to revise this code
            rates_fast = epi_array[:, T_RATE]
            rates_slow = compute_rates(epi_array, transmission_kernel, sigmas, gammas, min_float)[:, T_RATE]
            print(np.sum(rates_fast - rates_slow))
            assert rates_fast.all() == rates_slow.all(), "Incorrect rate calculation detected"
            epi_array = compute_rates(epi_array, transmission_kernel, sigmas, gammas, min_float) # RECALCULATE RATES FROM SCRATCH        
        # print(f'trate = {total_rate}, timestep = {delta_t}')
        if return_t50 == True: # if we want to return the time at which half the hosts become infected   
            fdict = Counter(epi_array[:, INFEC_STATUS]) # basically, count up the number of different individuals in each state.
            if int(fdict.get(STATUS_I, 0) + fdict.get(STATUS_C, 0)) == half_hosts:
                t50 = time + delta_t
        time += delta_t # move time forwards by the timestep

    # now t50 may never have been assigned if the epidemic was eradicated prior to half pop being infected at once
    try:
        assert t50 > 0
    except:
        t50 = -1 # to indicate that the 

    #print(f'EPI TERMINATED AT TIME {time} days\n{Counter(epi_array[:, INFEC_STATUS])}')
    if write_out == True:
        return epi_array


       
    if return_audpc == True: # begin building the results array
        rslts.append(compute_audpc(epi_array, species_parameters))
        rslts.append(compute_fsize(epi_array))
        #rslts.append(typed_audpc(epi_array,species_parameters )) # returns the type-separated AUDPC so we can see which type contributes
        # print(f'results array is {rslts}')

    elif return_audpc == False: # we return either final epidemic size or the epi array itself in this case.
        if return_fsize:
            return compute_fsize(epi_array)
        else:
            return epi_array
    else: 
        print(f'*** NPr: WARNING: PLEASE SET "return_audpc" to a valid boolean argument instead of {return_audpc}')
        return ValueError #
    
    if return_t50:
        rslts.append(t50)

    return rslts

def infective_time_notype(row):
    '''
    Similar to  `infective_time`, except does not take the host type into consideration. Called by `agnostic_audpc`
    '''
    [HOST_ID, SPS_ID, INFEC_STATUS, T_RATE, T_SC, T_CI, T_IR] = [0, 1, 2, 3, 4, 5, 6]
    [STATUS_S, STATUS_C, STATUS_I, STATUS_R] = [0,1,2,3]    

    end_status = row[INFEC_STATUS] # the the status of the host at the end of the epidemic
    hostid  = row[HOST_ID]
    typeid = row[SPS_ID]
    assert (end_status == STATUS_S) or (end_status == STATUS_R), f"Improper array passed to AUDPC function: epi. not fully finished sensu stricto\nHost in state {end_status}, host_id {hostid}, type {typeid}"
    
    if end_status == STATUS_S:
        time_infectious = 0

    elif end_status == STATUS_R:
        infected_at = row[T_SC]
        removed_at = row[T_IR]
        time_infectious = removed_at - infected_at
    else:
        return ValueError # there is a serious problem if this case appears, so it is here for debugging purposes
    return time_infectious 

def infective_time(row, species_parameters):
    '''
    ## Computes the time which the host represented in the row data was infective (ie member of C or I) for.
    Multiplies this by the infectivity of the host type, nu_i. Since different host types can emit different amounts of infectious pressure per unit time.

    Hence the return value has only relative meaning compared to other ones computed for different epidemics, since the `nu` and `phi` lack proper units.

    This can then be used for calculations by the AUDPC related functions
    '''

    [HOST_ID, SPS_ID, INFEC_STATUS, T_RATE, T_SC, T_CI, T_IR] = [0, 1, 2, 3, 4, 5, 6]
    [STATUS_S, STATUS_C, STATUS_I, STATUS_R] = [0,1,2,3]    

    [nu_A, nu_B] = species_parameters[:, 0] 
    end_status = row[INFEC_STATUS] # the the status of the host at the end of the epidemic
    assert (end_status == STATUS_S) or (end_status == STATUS_R), f"Improper array passed to AUDPC function: epi. not fully finished sensu stricto\nHost in state {end_status}"
    
    if row[SPS_ID] == 0: # the primary species/ host type 
        specific_nu = nu_A
    elif row[SPS_ID] == 1: # the alternate host, 'host B'
        specific_nu = nu_B
    else:
        print(f'Unrecognised species identity {row[SPS_ID]}')

    if end_status == STATUS_S:
        time_infectious = 0

    elif end_status == STATUS_R:
        infected_at = row[T_SC]
        removed_at = row[T_IR]
        time_infectious = removed_at - infected_at
    else:
        return ValueError # there is a serious problem if this case appears, so it is here for debugging purposes
    return time_infectious*specific_nu # weighted by the species-specific infectivity.

def agnostic_audpc(finished_epi):
    '''
    Computes area under disease progress curve in the sense of Cunniffe et al., 2015 Plos Comp Bio.
    Does NOT take into account the type of each host (for that see `compute_audpc`)

    '''
    audpc =  np.sum(np.apply_along_axis(infective_time_notype, 1, finished_epi)) 
    return audpc

def compute_audpc(finished_epi, species_parameters):
    '''
    ## Computes the Area Under Disease Progress Curve (AUDPC) for a terminated multitype epidemic.
    
    Since the epidemic is multitype, and infectivity is multitype, this must be done on the whole epi array 

    Importantly, the function assumes that the epidemic has fully ended: all hosts are either R or S. 
    '''

    audpc =  np.sum(np.apply_along_axis(infective_time, 1, finished_epi, species_parameters)) 
    return audpc

def typed_audpc(finished_epi, species_parameters):
    '''
    Computes the AUDPC returning it as individual type-specific components
    '''
    [HOST_ID, SPS_ID, INFEC_STATUS, T_RATE, T_SC, T_CI, T_IR] = [0, 1, 2, 3, 4, 5, 6]
    [STATUS_S, STATUS_C, STATUS_I, STATUS_R] = [0,1,2,3]    

    [nu_A, nu_B] = species_parameters[:, 0] 
    sps_A_epi = finished_epi[finished_epi[:, SPS_ID] == 0]
    sps_B_epi = finished_epi[finished_epi[:, SPS_ID] == 1]

    unsc_audpc_A = agnostic_audpc(sps_A_epi)
    unsc_audpc_B = agnostic_audpc(sps_B_epi)
    
    sc_audpc_A = unsc_audpc_A * nu_A # scale the AUDPC by the relevant type-specifici infectivity
    sc_audpc_B  = unsc_audpc_B * nu_B

    return [sc_audpc_A, sc_audpc_B]

def ensure_stoch():
    '''
    Ensures that the parallel child processes have process-specific randomness to avoid psuedoreplication
    '''
    return seed() 

def gillespie_prll(initial_condition, landscape, kernel_function, scale_parameter 
               , species_parameters, control_parameters, control_on, control_start, return_fsize, return_audpc,return_simple_audpc,debug, write_files, reused_kernel, return_t50, nsim,
               write_out):
    """
    ## Multiprocessing Wrapper for `gillespie` stochastic epidemic simulation
    
    This uses multiprocessing to distribute the computation across the available cores.
    Enforces process-specific random number generation to prevent stochasticity errors (since if naively done the worker processes will 
    share randomness, resulting in a partially determinstic model)
    
    - Also enforces random generation of initial cryptic location 
    
    ### Returns
    ------

    - Final epidemic size or the epi_array, or results; see the `gillespie` function itself for more detail on this

    - also returns the times at which controls were performed...: important for calculation of export risk to surrounding areas. 

    I.e., 
    """

    assert(type(write_out) == bool), "Pleease specify a boolean value for `write_out` argument"
    epi_dir(results_dir='epi_results')

    [epi_array, host_landscape] = setup_system(initial_condition=initial_condition, landscape=landscape) # we only need to calculate the kernel once
    reused_kernel = make_kernel(system_state=epi_array,
        landscape=host_landscape, kernel_function=kernel_function, scale_parameter=scale_parameter, 
        species_parameters=species_parameters)

    if write_out == True:
        epi_dir(results_dir='epi_results') # check for whether there is already a results directory so we don't overwrite anything by mistake
    elif write_out == False:
        pass
    else:
        print(f'please specify a boolean value for "write_out" rather than {write_out}')
        return ValueError
    with mp.Pool(initializer=ensure_stoch) as pl: # initialises the pool for the relevant worker processes and ensures that they will have different instantiations of initial condiitions
        epi = pl.starmap(gillespie, [(initial_condition, landscape, kernel_function, scale_parameter 
            , species_parameters, control_parameters, control_on, control_start,
            return_fsize, return_audpc, return_simple_audpc, debug, write_files, reused_kernel, True,  write_out) for i in range(nsim)])

    if write_out == True:
        for count, value in enumerate(epi):
            write_results(epi_array=value, results_dir='epi_results', replicate_number=count)
    return epi

if __name__ == "__main__":
    initial_condition = [1190,10,0,0]
    scale_parameter = 119
    kernel_function = exponential_kernel
    species_parameters = np.array([[0.015, 0.14, (1/350), 0], [0.015, 0.14, (1/350), 0]]) 

    landscapes = landscape_generators.CSR.gen_csr(2, 4000, nhosts_A=600, nhosts_B=600, nhosts_tot=1200,num_landscapes=1)
    control_parameters = [90,1,1,1,0.2,30,30] # 
    #  NB control parameters as [survey_freq, frac_A, frac_B, prob_detec_A, prob_detec_B, radius_A, radius_B] = control_parameters

    control_on = True
    control_start = 'random'
    return_fsize = True
    return_audpc = True
    return_simple_audpc = True
    debug = False
    write_files = False
    reused_kernel = None
    num_simulations = 10
    for landscape in landscapes:
        sims = gillespie_prll(initial_condition, landscape, kernel_function, scale_parameter 
               , species_parameters, control_parameters, control_on, control_start, return_fsize, return_audpc, 
               return_simple_audpc,debug, write_files, reused_kernel, return_t50=True, nsim=num_simulations, write_out=True)


'''
if __name__ == "__main__":
    initial_condition = [1100,10,0,0]

    scale_parameter = 84.5
    sps_parms = np.array([[0.0275, 0.14, (1/100), 0],
                          [0.0275, 0.14, (1/100), 0]])
    lss = landscape_generators.CSR.gen_csr(2, 3500, nhosts_A=600, nhosts_B=600, nhosts_tot=1200,num_landscapes=1)
    kernel_function = cauchy_thick

    
    radii = []
    fsizes = []
    for landscape in lss:
        for radius in np.arange(0,180, step=5):
            control_parameters = [90,1,1,1,radius,radius]
            fs = gillespie_prll(initial_condition, landscape, kernel_function, scale_parameter 
                , sps_parms, control_parameters, False, 'zero',False, False, False, None, 1)
            mean_fs = np.mean(fs)
            radii.append(radius)
            fsizes.append(mean_fs)
    print(f'radii are {radii}')
    print(f'mean_fs are {fsizes}')
    plt.scatter(x=radii, y=fsizes)
    plt.savefig('radiusscan.svg')
    plt.show()

################################################################################################################################

if __name__ == "__main__":
    initial_condition = [1101,10,0,0]
    fitted_beta = 0.00036
    scale_parameter = 36.1
    sps_parms = np.array([[math.sqrt(fitted_beta), math.sqrt(fitted_beta), (1/107), 0],
                          [math.sqrt(fitted_beta), math.sqrt(fitted_beta), (1/107), 0]])
    control_parameters = [90,1,1,1,31,31]
    epi = gillespie(initial_condition, "smallLandscape.txt", cauchy_thick, 36.1 
               , sps_parms, control_parameters, False, False, False, False, None)

if __name__ == "__main__":
    cols = ["HOST_ID", "SPS_ID", "INFEC_STATUS", "T_RATE", "T_SC", "T_CI", "T_IR"]
    exp_rad = 36
    cauchy_rad = 31
    exp_beta = 0.0009
    cauchy_beta = 0.00036
    fitted_beta = 0.00000845
    citrus_pos = pd.read_csv("./smallLandscape.txt", header=None)
    sps_params = np.array([[math.sqrt(exp_beta),math.sqrt(exp_beta) , (1/107), 0],[0, 0, (0) , 0]])
    flat_parms = np.array([[math.sqrt(fitted_beta),math.sqrt(fitted_beta) , (1/107), 0],[0, 0, (0) , 0]])
    ctrl_params = [90, 1, 1, 1, exp_rad, exp_rad]
    initial_cond = [1101,10,0,0]
    num_simulations = 2000
    with mp.Pool() as pool:
        sim_paths = pool.starmap(gillespie, [(initial_cond, exponential_kernel, 36.1,
                                                citrus_pos, sps_params, ctrl_params, True, True, False, False, None) for i in range(num_simulations)])
    
    print(f'--------------------------------')
    print(f'--------------------------------')
    print(f'For {num_simulations} epidemic:')
    print(f'mean value =  {np.mean(sim_paths)}')
    print(f'2.5% = {np.percentile(sim_paths, 2.5)}')
    print(f'50%  value = {np.median(sim_paths)}')
    print(f'97.5% = {np.percentile(sim_paths, 97.5)}')
    print(f'--------------------------------')
    print(f'--------------------------------')

    #sns.histplot(sim_paths, kde=True)
    sim_paths2 = []
    alt_ctrl = [90, 1, 1, 1, 30,30]
    with mp.Pool() as pool:
        sim_paths2 = pool.starmap(gillespie, [(initial_cond, exponential_kernel, 36.1,
                                                citrus_pos, sps_params, ctrl_params, True, True, False, False) for i in range(num_simulations)])
    #sns.histplot(sim_paths2, kde=True)

    sim_paths3 = []
    alt_ctrl = [90, 1, 1, 1, 50, 50]
    with mp.Pool() as pool:
        sim_paths3 = pool.starmap(gillespie, [(initial_cond, exponential_kernel, 36.1,
                                                citrus_pos, sps_params, ctrl_params, True, True, False, False) for i in range(num_simulations)])
    #sns.histplot(sim_paths3, kde=True, legend=f'control radius = {radius_A - 2}')

    # now make the dataframe
    dframe = pd.DataFrame( {str('f') + 'm': sim_paths, str(30) + 'm': sim_paths2, str(50)+'m':sim_paths3})
    sns.kdeplot(data=dframe)#
    plt.title(f"Final Epidemic Size Distributions for Three Radii: Miami B2-Exponential\n{num_simulations} replicates / radius")
    plt.xlabel("Final Epidemic Size")
    
    plt.show()

if __name__ == "__main__":
    cols = ["HOST_ID", "SPS_ID", "INFEC_STATUS", "T_RATE", "T_SC", "T_CI", "T_IR"]
    exp_beta = 0.0009
    cauchy_beta = 0.00036
    fitted_beta = 0.00000845
    citrus_pos = pd.read_csv("./smallLandscape.txt", header=None)
    sps_params = np.array([[math.sqrt(exp_beta),math.sqrt(exp_beta) , (1/107), 0],[0, 0, (0) , 0]])
    radius_A, radius_B = 36, 36
    ctrl_params = [90, 1, 1, 1, radius_A, radius_B]
    initial_cond = [1101,10,0,0]
    num_simulations = 2000
    with mp.Pool() as pool:
        sim_paths = pool.starmap(gillespie, [(initial_cond, exponential_kernel, 36.1,
                                                citrus_pos, sps_params, ctrl_params, True, True, False, False) for i in range(num_simulations)])
    
    print(f'--------------------------------')
    print(f'--------------------------------')
    print(f'For {num_simulations} epidemic:')
    print(f'mean value =  {np.mean(sim_paths)}')
    print(f'2.5% = {np.percentile(sim_paths, 2.5)}')
    print(f'50%  value = {np.median(sim_paths)}')
    print(f'97.5% = {np.percentile(sim_paths, 97.5)}')
    print(f'--------------------------------')
    print(f'--------------------------------')

    sns.histplot(sim_paths)
    plt.title(f"Final Size Distribution, {num_simulations} replicates\nConstant radius of removal, r* = {radius_A}")
    plt.show()

######################################################

# REFERENCES #

# Cunniffe et al 2015

# = Optimising and Communicating Options for the Control of Invasive Plant Disease When There Is Epidemiological Uncertainty
# Cunniffe NJ, Stutt ROJH, DeSimone RE, Gottwald TR, Gilligan CA (2015) Optimising and Communicating Options for the Control of Invasive Plant Disease When There Is Epidemiological Uncertainty. PLOS Computational Biology 11(4): e1004211.
# https://doi.org/10.1371/journal.pcbi.1004211 

#####################################################
'''