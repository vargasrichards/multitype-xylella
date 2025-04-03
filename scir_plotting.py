'''
Plotting functions for the gillespie_dsa file.
Mainly utility functions, particularly called by model_verif.py

A. Vargas Richards, 03.12.2024, 17.12.2024. 2025
'''

from gillespie_dsa import *
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from collections import Counter
import experiment1

citrus_pos = pd.read_csv("./smallLandscape.txt", header=None) # we first read in the citrus positions
citrus_pos.columns = ["posX","posY"]

def reconstruct_row(row, desired_time): # function called by reconstruct array
    [HOST_ID, SPS_ID, INFEC_STATUS, T_RATE, T_SC, T_CI, T_IR] = [0, 1, 2, 3, 4, 5, 6]
    STATUS_S, STATUS_C, STATUS_I, STATUS_R = 0,1,2,3
    t_sc, t_ci, t_ir = row[T_SC:(T_IR+1)] 
    sc_occurred, ci_occurred, ir_occurred = 0,0,0
    if t_sc != -1 and t_sc  < desired_time and t_sc >= 0:
        sc_occurred = 1
    if t_ci != -1 and t_ci < desired_time and t_ci >= 0:
        ci_occurred = 1
    if t_ir != -1 and t_ir < desired_time and t_ir >= 0:
        ir_occurred = 1

    if ir_occurred == True:
        state = STATUS_R
    elif (sc_occurred == True) and (ci_occurred == True) and (ir_occurred == False): 
        state = STATUS_I
    elif (sc_occurred == True) and ((ci_occurred + ir_occurred ) == False):
        state = STATUS_C
    elif (sc_occurred == False) and (ir_occurred  == False):
        state = STATUS_S
    else:
        print(f'error')
        return ValueError
    row[INFEC_STATUS] = state
    return row

def reconstruct_array(epi_array, chosen_time):
    print(f'epi array passed to reconstruct_array as {epi_array}')
    '''
    Reconstructs the infection status of each individual in the population at the 'chosen_time'
    '''
    return np.apply_along_axis(reconstruct_row, 1, epi_array, chosen_time)


def make_path(epi_array):
    '''
    Construct and return the states of the population at all timesteps where transitions occur
    '''
    [HOST_ID, SPS_ID, INFEC_STATUS, T_RATE, T_SC, T_CI, T_IR] = [0, 1, 2, 3, 4, 5, 6]
    STATUS_S,STATUS_C,STATUS_I,STATUS_R = 0,1,2,3

    times = np.arange(0,1500,step=2).tolist()

    S_path = []
    C_path = []
    I_path = []
    R_path = []
    
    for transition_time in times:
        print(f'transition time {transition_time}')
        pop_state = np.apply_along_axis(reconstruct_row, 1, epi_array, transition_time)
        #print(f'pop state is \n{pop_state}')
        states = pop_state[:, INFEC_STATUS]
        fdict = Counter(states)
        s = fdict.get(STATUS_S)
        c = fdict.get(STATUS_C)
        i = fdict.get(STATUS_I)
        r = fdict.get(STATUS_R)
        S_path.append(s)
        C_path.append(c)
        I_path.append(i)
        R_path.append(r)
    print(S_path, C_path, I_path, R_path, times)
    return [S_path, C_path, I_path, R_path, times]

def make_sps_path(epi_array): 
    '''
    # reconstructs the path of an epidemic - 2 species case

    Note that this returns a slightly less spiky sample path than the reality unless the timestep is set sufficently low
    '''
    [HOST_ID, SPS_ID, INFEC_STATUS, T_RATE, T_SC, T_CI, T_IR] = [0, 1, 2, 3, 4, 5, 6]
    STATUS_S,STATUS_C,STATUS_I,STATUS_R = 0,1,2,3

    times = np.arange(0,1400,step=10).tolist() # can tune this resolution to taste

    S1_path = []; S2_path = []
    C1_path = []; C2_path = []
    I1_path = []; I2_path = []
    R1_path = []; R2_path = []
    
    for transition_time in times:
        pop_state = np.apply_along_axis(reconstruct_row, 1, epi_array, transition_time)
        sps_1_arr =  np.where(pop_state[:, SPS_ID] == 0)[0]
        fdict_1 = Counter(pop_state[sps_1_arr, INFEC_STATUS])
        s_1 = fdict_1.get(STATUS_S)
        c_1 = fdict_1.get(STATUS_C)
        i_1 = fdict_1.get(STATUS_I)
        r_1 = fdict_1.get(STATUS_R)
        S1_path.append(s_1)
        C1_path.append(c_1)
        I1_path.append(i_1)
        R1_path.append(r_1)

        sps_2_arr =  np.where(pop_state[:, SPS_ID] == 1)[0]    
        fdict_2 = Counter(pop_state[sps_2_arr, INFEC_STATUS])
        s_2 = fdict_2.get(STATUS_S)
        c_2 = fdict_2.get(STATUS_C)
        i_2 = fdict_2.get(STATUS_I)
        r_2 = fdict_2.get(STATUS_R)
        S2_path.append(s_2)
        C2_path.append(c_2)
        I2_path.append(i_2)
        R2_path.append(r_2)

    return [[S1_path, C1_path, I1_path, R1_path],[S2_path, C2_path, I2_path, R2_path] , times]


def visualise_system(epi_array, host_locations, time, show): # provides a plot of the system at specified time-points
    '''
    Plot system in space, at a given time. 
    Have legend with 
    '''
    subplot_num = 0
    print(f'about to set epi_array within visualise system as \n{epi_array}')
    unlocated_epi = make_pandas(epi_array, ["HOST_ID", "SPS_ID", "INFEC_STATUS", "T_RATE", "T_SC", "T_CI", "T_IR"])
    fig, ax = plt.subplots()
    fig.suptitle('Epidemic Sampled at Particular Times')
    subplot_num += 1
    state_frame = make_pandas(reconstruct_array(unlocated_epi, time)) 
    
    #state_frame = pd.concat([state_frame, host_locations], axis=1)
    #viridis = mpl.colormaps['viridis'].resampled(4)
    print(f'state frame is \n {state_frame}')
    
    subplot = sns.scatterplot(data=state_frame, x="posX", y="posY", legend='auto', hue="infection_status")
    plt.title("Time = " + str(time))
    plt.savefig("epicourse.svg")
    if show:
        plt.show()
    else:
        return subplot

def facet_plots (finished_epi, host_locations,plot_times):
    '''
    Make a set of plots at specified time, in faceted style. 

    Label with times, and numbers in each compartment.
    depends upon the visualise_system function
    '''


    fig, axs = plt.subplots(2, 2, layout='constrained')
    ax = axs[0][0]
    ax.plot(visualise_system(finished_epi, host_locations, plot_times[0], False))
    ax.set_title('Title0 0')
    ax.set_ylabel('YLabel0 0')

    ax = axs[0][1]
    ax.plot(visualise_system(finished_epi, host_locations, plot_times[1], False))
    ax.set_title('Title0 1')
    ax.xaxis.tick_top()
    ax.tick_params(axis='x', rotation=55)


    for i in range(2):
        ax = axs[1][i]
        ax.plot(visualise_system(finished_epi, host_locations, plot_times[i+2], False))
        ax.set_ylabel('YLabel1 %d' % i)
        ax.set_xlabel('XLabel1 %d' % i)
        if i == 0:
            ax.tick_params(axis='x', rotation=55)

    fig.align_labels()  # same as fig.align_xlabels(); fig.align_ylabels()
    return 


if __name__ == "__main__":
    plot_times = [0,200,400,600]
    species_parameters = experiment1.PS1.sps_parms
    finished_epi = gillespie([1101,10,0,0], "smallLandscape.txt", cauchy_thick, 36.1 
               , species_parameters, [1,1,1,1,1,1], False, False, False, False, None)
    facet_plots (finished_epi, "smallLandscape.txt",plot_times)

# [1]  https://stackoverflow.com/questions/4100 
