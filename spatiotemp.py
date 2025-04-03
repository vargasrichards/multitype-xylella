'''
Functions for reporting the epidemic paths at arbitrary spatiotemporal resolution

A. Vargas Richards, Feb 2025
'''
import scir_plotting # this will provide tools for the reconstruction of the epidemic at an arbitrary time
import pandas as pd
import gillespie_dsa
import numpy as np
from collections import Counter # this will provide the tools for constructing the dpc#
import matplotlib.pyplot as plt
import landscape_generators
import seaborn as sns

def spatiotemp(finished_epi, sample_times, landscape, vline_times):
    '''
    ## Primary function which accepts an epi_array at the end of the epidemic

    Uses a single landscape - MUST BE THE LANDSCAPE USED FOR THE EPI as the coords will  be attached 
    ### Arguments:
    --------------
    '''
    FONTSIZE = 20
    MARKERSIZE = 20
    [HOST_ID, SPS_ID, INFEC_STATUS, T_RATE, T_SC, T_CI, T_IR] = [0, 1, 2, 3, 4, 5, 6] # constants for indexing arrays
    [STATUS_S, STATUS_C, STATUS_I, STATUS_R] = [0,1,2,3]

    x_pos = landscape['pos_X']
    y_pos = landscape['pos_Y']
    locations_array = np.array([x_pos, y_pos]).T
    end_dpc = np.max(sample_times)
    dpc_times=np.arange(0,stop=end_dpc, step=1)

    Cs = []
    Ss = []
    Is = []
    dead = []
    dpc_series = []

    for sample_t in sample_times:
        reconstructed = scir_plotting.reconstruct_array(finished_epi, sample_t)
        reconstructed_located = np.concatenate([reconstructed, locations_array], axis=1)
        np.savetxt(f'epi_T={sample_t}.csv', reconstructed_located, delimiter=",")

    for dpc_t in dpc_times:
        reconstructed = scir_plotting.reconstruct_array(finished_epi, dpc_t)
        state_dict = Counter(reconstructed[:, INFEC_STATUS])
        num_cryptics = state_dict.get(STATUS_C, 0) 
        num_symptomatics = state_dict.get(STATUS_I,0)
        num_susceptibles = state_dict.get(STATUS_S, 0)
        num_dead = state_dict.get(STATUS_R, 0)
        Cs.append(num_cryptics)
        Ss.append(num_susceptibles)
        Is.append(num_symptomatics)
        dead.append(num_dead)
        dpc_series.append(dpc_t)

    fig, ax = plt.subplots()
    ax.tick_params(axis='both', labelsize=FONTSIZE)

    ax.plot(dpc_series, Ss, color=  'green', label='Susceptible', markersize=MARKERSIZE)
    ax.plot(dpc_series, Cs , color='orange',label='Cryptically Infected', markersize=MARKERSIZE)
    ax.plot(dpc_series, Is , color='red'  , label='Symptomatially Infected', markersize=MARKERSIZE)
    ax.plot(dpc_series, dead , color='black'  , label='Dead', markersize=MARKERSIZE)
    #ax.vlines(x=vline_times,ymin=0, ymax=1000)
    plt.xlabel("Time /days", fontsize=FONTSIZE)
    plt.ylabel("Number of hosts", fontsize=FONTSIZE)
    plt.legend(prop = { "size": FONTSIZE })
    plt.savefig('dpc_subplot.svg')
    plt.show()
    return

def slice_images(times):
    STATUS_S = 0
    STATUS_C = 1
    STATUS_I = 2
    STATUS_R = 3 # defining the constants used for our epi ARRAY
    sns.set_style("whitegrid")
    colors = {0 : '#00C44F', 1 : '#ffa500', 2 : '#FF2500', 3:'#000000'}

    # this should probably be parallelised as it can take a v long time to run....
    for time in times:
        epi = pd.read_csv(f'epi_T={time}.csv', delimiter= ',')
        epi.columns = ["HOST_ID", "SPS_ID", "INFEC_STATUS", "T_RATE", "T_SC", "T_CI", "T_IR","pos_X", "pos_Y"] 
        pop_state = Counter(epi["INFEC_STATUS"])
        num_S= pop_state.get(STATUS_S, 0)
        num_C= pop_state.get(STATUS_C, 0)
        num_I= pop_state.get(STATUS_I, 0)
        num_R= pop_state.get(STATUS_R, 0)
        popstate_str = f'S(t) = {int(num_S)}\nC(t) = {int(num_C)}\nI(t) = {int(num_I)}\nR(t) = {int(num_R)}'
        snp = sns.relplot(x="pos_X", y="pos_Y", hue="INFEC_STATUS", size="SPS_ID",
                    sizes=(20, 40), alpha=.7, palette=colors,hue_order=range(0,4),
                    height=6, data=epi).set(title=f'Time = {time} days', xlabel="X position /m", ylabel= "Y position /m")
        plt.annotate(popstate_str,  
            xy=(1.2, 0.8), 
            xycoords=('axes fraction', 'figure fraction'),
            xytext=(12, 12),  
            textcoords='offset points',
            size=10, ha='right', va='top')      

        snp.savefig(f'{time}_nocontrol.svg')
        snp.savefig(f'{time}_nocontrol.png')
    return

def summary_landscapes(landscapes):
    '''
    This function plots some landscapes in a grid. Usually 9 landscapes are supported
    '''
    #colors = {0: '#00C44F'} # we'll only need S hosts in this case as opposed to `grid_epi` below which 
    markers = {0: 'o', 1: 's'}  
    colours = {}
    marker_colours = {0: 'blue', 1:'orange'}
    ncols = 2
    nrows = 1
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True)
    axes = axes.flatten()
    i = 0
    for ax in axes:
        ax.set_aspect('equal')
        subset = landscapes[i] # landscape of interest
        if i == 0:
            initial_label = ""
        else:
            initial_label = "_"
        i += 1
        
        for sps_id in subset["sps_id"].unique():
            sps_subset = subset[subset["sps_id"] == sps_id]
            ax.scatter(x=sps_subset["pos_X"],y= sps_subset["pos_Y"],
                       marker=markers.get(sps_id, 'o'),s=4,  alpha=1, label=str(initial_label) + f"Host Type {sps_id}")
        # ax.set_title(f"") # we need to put in the landscape details here...
        lscape_str = f""
        ax.text(1, 0.5, lscape_str, transform=ax.transAxes, fontsize=10,
                verticalalignment='center', bbox=dict(facecolor='white', alpha=0.5))
    
    marker_handles = [plt.Line2D([0], [0], marker=markers[m], linestyle='None', markersize=10, markeredgecolor=marker_colours[m],markerfacecolor=marker_colours[m]) for m in markers]
    typedict = {0:'A', 1: 'B'}
    marker_labels = [f"Host Type {typedict.get(m)}" for m in markers]
    
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), handles=marker_handles, labels = marker_labels , frameon= False)
    fig.supxlabel("X position /m")
    fig.supylabel("Y position /m")
    plt.savefig("landscape_subplots.svg")
    plt.show()
    return 

def grid_epi(selected_times):
    """
    Function to plot a 2D grid of epidemic snapshots at selected time points,
    displaying infection counts to the side of each plot and a single legend outside.
    """
    [STATUS_S, STATUS_C, STATUS_I, STATUS_R] = [0,1,2,3]

    colors = {0: '#00C44F', 1: '#ffa500', 2: '#FF2500', 3: '#000000'}
    markers = {0: 'o', 1: 's'}  
    ncols = 3
    nrows = 3
    FONTSIZE=20

    all_data = []
    pop_states = {}
    for time in selected_times:
        epi = pd.read_csv(f'epi_T={time}.csv', delimiter=',')
        epi.columns = ["HOST_ID", "SPS_ID", "INFEC_STATUS", "T_RATE", "T_SC", "T_CI", "T_IR", "pos_X", "pos_Y"]
        epi["Time"] = time  
        all_data.append(epi)
        
        pop_state = Counter(epi["INFEC_STATUS"])
        pop_states[time] = pop_state
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 12), sharex=True, sharey=True)
    #fig.subplots_adjust(wspace=0.4, hspace=0.1)

    axes = axes.flatten()
    splt = 0
    for ax, time in zip(axes, selected_times):
        ax.tick_params(axis='both', labelsize=FONTSIZE)

        subset = all_data[selected_times.index(time)]
        
        for sps_id in subset["SPS_ID"].unique():
            speciesdf = subset[subset["SPS_ID"] == sps_id]
            if splt == 0:
                relevants = speciesdf
            else:
                relevants = speciesdf[speciesdf["INFEC_STATUS"] != STATUS_S]
            ax.scatter(relevants["pos_X"], relevants["pos_Y"], c=relevants["INFEC_STATUS"].map(colors),
                    marker=markers.get(sps_id, 'o'), s=14, alpha=0.5, label=f"Host Type {sps_id}")
        splt += 1
        ax.set_title(f"{time} days", fontsize= FONTSIZE)
        ax.set_aspect('equal')
        if ax in axes[-ncols:]:
            ax.set_xlabel("X position /m", fontsize=FONTSIZE)
            fig.suptitle('test title', fontsize=20)

        if ax in axes[::ncols]:
            ax.set_ylabel("Y position /m", fontsize=FONTSIZE)      
        pop_state = pop_states[time]

    for ax in axes[len(selected_times):]:
        fig.delaxes(ax)
    
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in colors.values()]
    labels = ["Susceptible", "Cryptically Infected", "Symptomatically Infected", "Removed"]
    marker_handles = [plt.Line2D([0], [0], marker=markers[m], color='black', linestyle='None', markersize=10) for m in markers]
    typedict = {0:'A', 1: 'B'}
    marker_labels = [f"Host Type {typedict.get(m)}" for m in markers]
    
    fig.legend(handles + marker_handles, labels + marker_labels, title="", loc="upper right", bbox_to_anchor=(1, 1), prop = { "size": FONTSIZE })
    plt.suptitle("Epidemic Progress: No Control Implemented\nComplete Spatial Randomness, Fraction(B) = 0.5, Exponential Kernel", fontsize=FONTSIZE )
    plt.savefig("epidemics.svg")
    plt.show()

def examine_epi(times):
    slice_images(times)
    return
# plotting the landscapes

#landscapes = [
#*CSR.gen_csr(2,4000,nhosts_A=555, nhosts_B=555, nhosts_tot=1110, num_landscapes=1), *CSR.gen_csr(2,4000,nhosts_A=1110, nhosts_B=0, nhosts_tot=1110, num_landscapes=1)]
##summary_landscapes(landscapes)

    
if __name__ == "__main__":
    initial_condition = [1100,10,0,0]
    lscape_size = 3000 # since the normalisation below
    Nhosts = int(np.sum(initial_condition))
    n_B = int(Nhosts/2)
    num_landscapes = 1
    scale_parameter = 119
    kernel_function = gillespie_dsa.exponential_kernel
    species_parameters = np.array([[0.00833, 0.14 ,(1/350), 0.00053 ] ,[0.00833 ,0.14, (1/350), 0.00053 ]])
    radius_A = 500
    radius_B = 500
    control_parameters = [90, 1,1,1,0.2,radius_A, radius_B]    
    # nb. survey_freq, frac_A, frac_B, prob_detec_A, prob_detec_B, radius_A, radius_B = control_parameters

    landscapes = landscape_generators.CSR.gen_csr(num_species=2, scaler=lscape_size,
                                                            nhosts_A=(Nhosts - n_B), nhosts_B=n_B,
                                                            nhosts_tot=Nhosts, num_landscapes=num_landscapes)

    example_epi = gillespie_dsa.gillespie_prll(initial_condition, *landscapes, kernel_function, scale_parameter, species_parameters, control_parameters,
                                            False, 'random', True, True, True, False, False, None, True, 1, True)
    plot_times = [0, 250,500,750,1000,1250,1500,1750]
    
    for epi in example_epi:
        for landscape in landscapes:
            spatiotemp(epi, np.arange(0,2000,step=10), landscape =landscape, vline_times= plot_times)

    examine_epi(plot_times)
    grid_epi(plot_times)


