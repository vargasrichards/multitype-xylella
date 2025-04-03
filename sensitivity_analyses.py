'''
Toolbox of functions for sensitivity analyses

Includes functions for:

- Number of hosts (preserving overall density, so landscape is scaled up)


# Outcome variables and reporting
-----
 - outcome variables are written out for plotting using other functions


A. Vargas Richards, Cambridge 2025
'''

import numpy as np
import landscape_generators as lg
import gillespie_dsa as gd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def survey_interval (survey_intervals, **kwargs):
    '''
    # Examines the effect of a changing survey interval on the ensuing epidemic dynamics subject to radial control


    ## Arguments
    ------------

    `survey_intervals` : the values (in days) to scan over

    '''


    #for sinterval in survey_interva
    
def vary_Nhosts(N_low, N_high, N_step, radius_low, radius_high, radius_step, 
    frac_B, density, nlandscapes, kernel_function, scale_parameter, sps_parms, 
    control_parameters,nrep , filename):
    '''
    Function to test the response of the epidemic to increasing patch size, but with
    CSR on a fixed-density landscape. Can be adapted to clustered landscape and arbitrary kernels.

    For a constant radius of control, optimised to a small N, the number of hosts is increased and then we examine the response of normalised Ke: Ke/Nhosts 
    '''
    return_fsize = True
    debug = False
    with open(filename, 'w') as f:
    # f.write(f'{frac_B}\t{radius}\t{round(median_fsize)}\t{median_endtimes}\t{median_audpc}\t{median_audpc_sps}\n')
        f.write(f'# Sensitivity Analysis: Varying N Hosts \n')
        f.write(f'# Generative Landscape Model: Complete Spatial Randomness (CSR) \n')
        f.write(f'# Kernel = {kernel_function}, Scale Parameter = {scale_parameter} m\n Sps Parameters = {sps_parms} \n')
        f.write(f'# Control parameters = {control_parameters}\n')
        f.write(f'# control_start = "random"\n')
        f.write(f'# species parameters = {sps_parms}\n')
        f.write(f'N_hosts\tLandscape_Length\tDensity\tRadius\tFrac_B\tMed_KE\tKE_q10\tKE_q20\tKE_q30\tKE_q40\tKE_q50\tKE_q60\tKE_q70\tKE_q80\tKE_q90\tMed_AUDPC_SPS\tAUDPC_q10\tAUDPC_q20\tAUDPC_q30\tAUDPC_q40\tAUDPC_q60\tAUDPC_q70\tAUDPC_q80\tAUDPC_Q90\n')
        for N in np.arange(start=N_low, stop=N_high+N_step, step= N_step):
            print(f'NPr: generating landscapes for Nhosts  = {N}')
            N_b = int(frac_B) * N
            N_a = int(N - N_b)
            side_length = 1000 * float(np.sqrt(N / density) ) # we compute the side length of the square landscape to main
            landscapes = lg.CSR.gen_csr(2, scaler=side_length, nhosts_A= N_a,
            nhosts_B= N_b, nhosts_tot=N, num_landscapes= nlandscapes)
            initial_cond = [N - int(N/100), int(N/100), 0, 0 ] # maybe this can be refined further? 
            fsizes = []
            sps_audpc = []  #holds the species-refined audpc 

            for radius in np.arange(radius_low, radius_high, step=radius_step):
                print(f'*NPr: Radius of control {radius}m')
                # we need to set the control parameters now... 
                control_parameters[-1] = radius
                control_parameters[-2] = radius
                lc = 0
                for lscape in landscapes:
                    lc += 1
                    print(f'NPr: landscape {lc}/{nlandscapes} at Nhosts = {N} hosts')
                    [epi_array, host_landscape] = gd.setup_system(initial_cond, landscape=lscape)  
                    [transmission_kernel, epi_array, distance_matrix] = gd.make_kernel(system_state=epi_array,
                    landscape=lscape, kernel_function=kernel_function, scale_parameter=scale_parameter,
                    species_parameters=sps_parms)  
                    rs = gd.gillespie_prll(initial_cond, lscape, kernel_function, scale_parameter
                            , sps_parms, control_parameters, control_on=True, control_start='random', return_fsize=return_fsize, return_audpc=True, 
                            return_simple_audpc=True, debug=debug, write_files=False, reused_kernel=transmission_kernel,return_t50=True,nsim=nrep, write_out=False)
                    for r in rs:
                        #print(f'r returned as {r}')
                        sps_audpc.append(r[0]) # the speciesspecific AUDPC
                        fsizes.append(r[1])

                med_Ke = np.median(fsizes)
                median_audpc_sps = np.median(sps_audpc)

                p10_audpc_sps = np.percentile(sps_audpc,10)
                p20_audpc_sps = np.percentile(sps_audpc, 20)
                p30_audpc_sps = np.percentile(sps_audpc,30)
                p40_audpc_sps = np.percentile(sps_audpc,40)
                p60_audpc_sps = np.percentile(sps_audpc, 60)
                p70_audpc_sps = np.percentile(sps_audpc,70)
                p80_audpc_sps = np.percentile(sps_audpc,80)
                p90_audpc_sps = np.percentile(sps_audpc,90)

                ke_q10 = np.percentile(fsizes, 10)
                ke_q20 = np.percentile(fsizes,20)
                ke_q30 = np.percentile(fsizes, 30)
                ke_q40 = np.percentile(fsizes,40)
                ke_q60 = np.percentile(fsizes, 60)
                ke_q70 = np.percentile(fsizes, 70)
                ke_q80 = np.percentile(fsizes, 80)
                ke_q90 = np.percentile(fsizes, 90)

                [null_var1,null_var2,null_var3,null_var4,null_var5,radius, radius] = control_parameters # just to get the radius of conntrol being applied.
                dens = round(N /( side_length ** 2 ), ndigits=5)  # density of hosts, rounded to a reasonable degree (should all be tge same throughout anyway )
                # .write(f'N_hosts\tLandscape_Length\tDensity\tRadius\tMed_KE\tKE_q10\tKE_q20\tKE_q10\tKE_q30\tKE_q70\tKE_Q90\tMed_AUDPC_SPS\tAUDPC_q10\tAUDPC_q30\tAUDPC_q70\tAUDPC_Q90\n')
                f.write(f'{N}\t{side_length}\t{dens}\t{radius}\t{frac_B}\t{med_Ke}\t{ke_q10}\t\t{ke_q20}\t{ke_q30}\t{ke_q40}\t{ke_q60}\t{ke_q70}\t{ke_q80}\t{ke_q90}\t{median_audpc_sps}\t{p10_audpc_sps}\t{p20_audpc_sps}\t{p30_audpc_sps}\t{p40_audpc_sps}\t{p60_audpc_sps}\t{p70_audpc_sps}\t{p80_audpc_sps}\t{p90_audpc_sps}\n')
    print(f'Finished sensitivity analysis!')
    return


def scale_analysis(results_file):
    '''
    This function analyses the output of the `vary_Nhosts` function.
    Plots unnormalised outcomes for Ke and AUDPC and then normalises them by dividing through by the number of hosts.


    '''
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=False)
    axes[0,1].set_ylim(0,1)
    axes[1,1].set_ylim(0,1)

    sensitivity_res = pd.read_csv(results_file, sep='\t')
    print(f'read in results of the sensitivity analysis as:\n{sensitivity_res}')
    sensitivity_res.columns = [['N_hosts','Landscape_Length /m','Density','Removal Radius','Med_KE','KE_q10','KE_q30','KE_q70',
                                'KE_Q90','Med_AUDPC_SPS','AUDPC_q10','AUDPC_q30','AUDPC_q70','AUDPC_Q90']]

    # now we should compute the metrics scaled by the number of hosts.
    median_kes=sensitivity_res["Med_KE"].to_numpy()
    median_audpcs = sensitivity_res["Med_AUDPC_SPS"].to_numpy()
    host_array = sensitivity_res['N_hosts'].to_numpy()
    scaled_ke = median_kes/host_array
    scaled_audpcs = median_audpcs/host_array
    print(f"scaled median KE is {median_kes/host_array}")


    axes[0,0].scatter(x=host_array, y=median_kes)
    axes[1,0].scatter(x=host_array,y=median_audpcs)

    axes[0,1].scatter(x=host_array, y=scaled_ke)
    axes[1,1].scatter(x=host_array,y=scaled_audpcs)
    fig.supxlabel("Number of hosts, N")
    fig.suptitle("Sensitivity Analysis: Varying N")
    plt.show()
    return


if __name__ == "__main__":
    sps_parms = np.array([[0.0025  ,  0.09      , (1/350) ,0.00053   ], [0.0025  ,   0.09     ,  (1/500),0.00053   ]]) # ie the delta-sigma case
    control_parameters =  [90, 1, 1, 1, 1, 400, 400]
    nrep = 25
    vary_Nhosts(N_low=1000, N_high=5000, N_step=1000, radius_low=0, radius_high=2000, radius_step=100,
    frac_B=0.5, density=69.3, nlandscapes=2, kernel_function=gd.exponential_kernel, scale_parameter=119, sps_parms=sps_parms, 
    control_parameters=control_parameters,nrep=nrep, filename="test_sensitivity.txt" )


# we need to do this analysis for a cauchy kernel as well. 

###################################
#
# NORMALISATION I: 
#
#
#
# NORMALISATION II: 
#
