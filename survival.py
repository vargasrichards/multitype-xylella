'''
Analyse survival functions
-------------------------

Tracking the number or fraction of hosts which remain infectious at time t /days after having been infected at a given time t
Can be used to analyse individual epidemics as well as ensembles of epidemics. 

Time(survived ) = Time(removed, via death or control event) - Time(infected)

integrating this gives the AUDPC

the idea of survival analysis here is to separate effects of normalisation on the epidemic outcome and instead look at the time taken for hosts broadcasting infectious pressure to be removed 

A. Vargas Richards, 2025
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gillespie_dsa
import landscape_generators
from collections import Counter
from pathlib import Path
from numpy import loadtxt

def _compute_survival(epi_array, temporal_res, normalise):
    '''
    Plots survival curves by host type: 
    (survivald being time infectious before removal or death) 

    `temporal_res` : gives the 
    
    `normalise` : if True, then the number surviving is divided by the initial number of hosts of that type

    '''

    [HOST_ID, SPS_ID, INFEC_STATUS, T_RATE, T_SC, T_CI, T_IR] = [0, 1, 2, 3, 4, 5, 6]
    SPECIES_0, SPECIES_1 = 0,1
    # first we should separate the array into its types...
    A_array = np.where(epi_array[:, SPS_ID] == SPECIES_0)[0]
    B_array = np.where(epi_array[:, SPS_ID] == SPECIES_1)[0]
    As = epi_array[A_array]
    Bs = epi_array[B_array]
    survival_A = np.apply_along_axis(gillespie_dsa.infective_time_notype, 1, As)
    survival_B = np.apply_along_axis(gillespie_dsa.infective_time_notype, 1, Bs)
    active_A = []
    active_B = []
    times=[]
    max_tA = np.max(survival_A)
    max_tB = np.max(survival_B)
    for plot_time in np.arange(0, np.max([max_tA, max_tB]) + 1, step = temporal_res):
        times.append(plot_time)
        nactive_A= Counter((survival_A > plot_time)).get(True,0)
        nactive_B= Counter((survival_B > plot_time)).get(True,0)
        active_A.append(nactive_A)
        active_B.append(nactive_B)

    if normalise == True: # then we need to divide by the total number of type i infected
        active_A /= np.max(active_A)
        active_B /= np.max(active_B)
        print(f'active A  = {active_A}\nactive B = {active_B}')
    return [active_A, active_B, times]

def find_nearest(array, value):
    '''
    this is used in the approximate calculation of host half-life
    source : https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array 
    '''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def compute_survival(epi_array, temporal_res, normalise, halflives):
    
    '''
    ## Calls the helper function above in the case of an ensemble of epidemics passed to the function.

    ## Arguments
    -------------

    `halflives` : if set to `True`, then the half life of each host type will be annotated on the plot
    
    '''
    runningA = np.zeros(30000)
    runningB = np.zeros(30000)

    if type(epi_array) == list:
        nreps = len(epi_array) 
        print(f'computing mean survival curve for an ensemble of epidemics\nwith {nreps} replicates')
        for epi in epi_array:
            print(f'_compute_survival(epi_array=epi, temporal_res=temporal_res, normalise=True) = {_compute_survival(epi_array=epi, temporal_res=temporal_res, normalise=True)}')
            [A_data, B_data, times] = _compute_survival(epi_array=epi, temporal_res=temporal_res, normalise=True)
            times = np.array(times)
            to_pad_A = 30000 - len(A_data)
            to_pad_B = 30000 - len(B_data)
            A_data = np.pad(A_data,(0, to_pad_A), 'constant')
            B_data = np.pad(B_data,(0, to_pad_B), 'constant')
            runningA = np.add(A_data, runningA)
            runningB = np.add(B_data, runningB)
    
        # now we divide through
        runningA = np.array(runningA).astype(float)
        runningB = np.array(runningB).astype(float)

        runningA /= float(nreps)
        runningB /= float(nreps)
        print(f'A: {runningA}\nB: {runningB}\ntimes: {times}')
        print(f'A: {len(runningA)}\nB: {len(runningB)}\ntimes: {len(times)}')
        to_pad_times = 30000 - len(times)
        times = np.pad(times, (0, to_pad_times), 'constant')
        sns.set_theme()
        # now for the half-life analysis 
        if halflives:
            print(f'find_nearest(runningA, 0.5 ) = {find_nearest(runningA, 0.5 )}')
            A_hlife = times[find_nearest(runningA, 0.5 )]
            B_hlife = times[find_nearest(runningB, 0.5 )]
            delta_hlife = B_hlife - A_hlife 
            # assert delta_hlife > 0, ""
            # need to plot these as vertical lines
            plt.axvline(x = A_hlife, ymax=0.5, color = 'tab:blue', label = r'$T_{\frac{1}{2}}$' + f' type A host = {A_hlife} days')
            plt.axvline(x = B_hlife, ymax=0.5,color= 'tab:orange', linestyle="dashed",label = r'$T_{\frac{1}{2}}$' + f' type B host = {B_hlife} days')

        plt.scatter(times, runningA,label = f"Mean survival, Type A",s=2)
        plt.scatter( times, runningB,label = f"Mean survival, Type B",s=2)
        plt.title(r"$T_{\frac{1}{2}}(B) - T_{\frac{1}{2}}(A) = $" + f"{delta_hlife} days")
        plt.xlabel(f"Time post infection /days")
        plt.ylabel(f'Fraction of hosts infectious')
        plt.legend(frameon=False)
        plt.savefig("ensemble_survival.svg")
        plt.show()
    else:
        print(f'computing survival curve for a single stochastic epidemic')
        _compute_survival(epi_array=epi_array, temporal_res=temporal_res, normalise=normalise)
    return plt

def half_sensitivity():
    '''
    Computes delta (t1/2) for a range of values for either the landscape parameters or the host parameters
    Finds the crossing-overpoint
    
    '''
    return

def panel_plot(epi_array1, epi_array2, temporal_res, normalise, halflives, title1, title2):
    '''
    Compare two different scenarios and the survival of different host types...

    - can compare two different landscape models, for instance. 

    Essentially we 
    '''
    subplot1 = compute_survival(epi_array1, temporal_res, normalise, halflives)
    subplot2 = compute_survival(epi_array2, temporal_res, normalise, halflives)
    plt.subplots = subplot1, subplot2 # not sure if this is the correct syntax
    plt.show()
    return 

def half_life(epi_array):
    '''
    Computes the time taken for half of the hosts to become removed or dead following infection

    note that we can't extract this information from AUDPC alone, but AUDPC  = infectivity of type i *  area under survival curve (unnormalised) for that type.

    '''
    # first we need to check which types are in the epi_array and divide it into two.
    [HOST_ID, SPS_ID, INFEC_STATUS, T_RATE, T_SC, T_CI, T_IR] = [0, 1, 2, 3, 4, 5, 6]
    SPECIES_0, SPECIES_1 = 0,1

    A_array = np.where(epi_array[:, SPS_ID] == SPECIES_0)[0]
    B_array = np.where(epi_array[:, SPS_ID] == SPECIES_1)[0]
    As = epi_array[A_array]
    Bs = epi_array[B_array]

    number_As = len(As) # for use when normalising later
    number_Bs = len(Bs)

    survival_A = np.apply_along_axis(gillespie_dsa.infective_time_notype, 1, As)
    survival_B = np.apply_along_axis(gillespie_dsa.infective_time_notype, 1, Bs)

    print(f'survival for type A is \n{survival_A}')
    print(f'survival for type B is \n{survival_B}')
    return

def ensemble_survival(results_dir):
    '''
    Computes survival for all epis written out in their individual subdirectories in the parent directory
    '''
    
    [HOST_ID, SPS_ID, INFEC_STATUS, T_RATE, T_SC, T_CI, T_IR] = [0, 1, 2, 3, 4, 5, 6]
    SPECIES_0, SPECIES_1 = 0,1

    currdir = Path('.')
    for epi_file in list(currdir.glob('*epi*.txt')): # catchall exprs. for the files which could be relevant
        with epi_file as e:
            finished_epi = np.genfromtxt(e, delimiter="\t", missing_values=" +") # the finished epidemic is read in
            # we can also plot the half-life.... 
            compute_survival(epi_array=finished_epi, temporal_res=1, normalise=True)
    return

def halflives():
    '''
    Plot the halflives of infected hosts under different parameterisations
    '''


    return

if __name__ == "__main__":
    initial_cond = [1100,10,0,0]
    Nhosts = int(np.sum(initial_cond))
    frac_B = 0.8
    nb = int(frac_B* Nhosts)
    na = Nhosts - nb
    lscape = landscape_generators.CSR.gen_csr(2, 4000, na, nb,1110, 1) # unclustered
    # we can compare survival functions across landscape models.... 
    kernel_function = gillespie_dsa.exponential_kernel
    scale_parameter = 119
    #sps_parms = np.array([[0.01  ,  0.09      , (1/350) ,0.00053   ], [0.01 ,   0.09     ,  (1/350),0.00053  ]])
    
    sps_parms = np.array([[0.015, 0.14, (1/350), 0], [0.015, 0.14, (1/350), 0]]) 
    survey_interval = 90
    radius = 1
    radius = 1
    p_d_A = 1
    p_d_B = 0.2  # to implement crypticity
    control_parameters = [survey_interval, 1, 1, p_d_A, p_d_B, radius, radius]

    n_paths=250
    control_applied = True
    control_start = "random"
    for l in lscape:
        epis = gillespie_dsa.gillespie_prll(initial_cond, l, kernel_function, scale_parameter 
        , sps_parms, control_parameters, nsim=n_paths, 
        control_on=control_applied, control_start=control_start, return_audpc=False,
        return_simple_audpc=False,return_fsize=False, debug=False, write_files=False, 
        reused_kernel=None, return_t50=False, write_out=True)
        compute_survival(epis, temporal_res=1, normalise=True, halflives=True)


# we also need to consider the case of the clustered landscape
# for a clustered landscape

    lscape = landscape_generators.neymanscott.get_ns(2, fgg)