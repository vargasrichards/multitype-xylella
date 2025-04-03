'''
Functions for miscellaneous parameter sweeps etc.
Code for sweeping p_d and radius if desired
Also contains code for sweeping proportion of second host type

- some outputs of this file can be visualised using 'result_reader.py' script 

A Vargas Richards. Dec 2024, Jan., Feb. 2025

'''

import numpy as np, multiprocessing as mp
import landscape_generators
from gillespie_dsa import *
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

class dispersal:
    '''
    # Contains dispersal scale parameter for kernel usage.
    '''
    cauchy_alpha = 84.5
    exponential_alpha = 119
    citrus_alpha = 36.1 # for (Hyatt-Twynham et al., 2017)

def pd_plot(initial_cond, species_parameters, ctrl_parameters, num_simulations, landscape, kernel_function, scale_parameter, pd_B
            , radiusA, radiusB):
    '''
    Scan the effect of misspecified p_d on mean epidemic final size in the single species case.
    '''
    sims = []
    probs = []
    for pd_A in np.arange(0,1.2, step=.2):
        ctrl_parameters = [1,1,pd_A, pd_B, radiusA, radiusB]
        sim_paths = gillespie_prll(initial_cond, landscape, kernel_function,
                                   scale_parameter, species_parameters, ctrl_parameters,
                                   True, True, False, False, None, num_simulations )
        sims.append(np.mean(sim_paths)/np.sum(initial_cond))
        probs.append(pd_A)

    sns.set_theme(style="whitegrid")
    sns.despine()
    plt.scatter(x = probs, y=sims)
    plt.xlabel("Probability of Detection: host in class I)")
    plt.ylabel(f"Mean Final Epidemic Size, {num_simulations} epidemics")
    plt.title("Effect of Imperfect Detection of Symptomatic Host\n Fixed Radius of Removal Optimised for Perfect Detection")
    plt.savefig("detection_sweep_singlesps.eps")
    plt.show()

def radius_plot(initial_cond, species_parameters, ctrl_parameters, num_simulations, landscapes, kernel_function,
                scale_parameter, radius_low, radius_high, radius_step):
    '''
    Scan the effect of different radii of removal  on median epidemic impact (K_{E})
    '''
    fig, ax = plt.subplots()
    radii = []
    Kes = []; ke_q10s = [];  ke_q30s = [];  ke_q70s = [];  ke_q90s = [] # these store the epidemicn impacta
    num_landscapes = int(len(landscapes))
    # hence the number of replicates per landscape is just 
    nreplicates = int(num_simulations  / num_landscapes)
    print(f'performing {nreplicates} epidemics per each of {num_landscapes} landscapes\ntotal epidemics per landscape parameterisation {num_simulations} epidemics')
    
    for radius in np.arange(radius_low,radius_high + radius_step, step=radius_step):
        print(f'set radius {radius} m')
        # NEED TO ALTER CONTROL PARAMETERS HERE BUT ONLY TO ALTER RADIUS (I guess it doesn't matter for the all A landscapes but anyways )
        # ctrl_parameters = [90,1,1,1,1,radius, radius]
        ctrl_parameters = [90,1,1,1,1,radius, radius]
        all_rslts = []
        for lscape in landscapes:
            [epi_array, host_landscape] = setup_system(initial_cond, landscape=lscape) # assign the hosts to their initial states
            [transmission_kernel, epi_array, distance_matrix] = make_kernel(system_state=epi_array,
            landscape=lscape, kernel_function=kernel_function, scale_parameter=scale_parameter,
            species_parameters=species_parameters) # we construct our transmission kernel with all transmission constants
            sim_paths = gillespie_prll(initial_condition=initial_cond, landscape=lscape, kernel_function= kernel_function, 
                                    scale_parameter=scale_parameter, species_parameters=species_parameters, control_parameters= ctrl_parameters,
                                    control_on=True, control_start="random", return_fsize=True, return_audpc=False, return_simple_audpc=False, debug=False, 
                                    write_files=False, reused_kernel= transmission_kernel, return_t50=False, nsim=nreplicates, write_out=False)
            #sims.append(np.mean(sim_paths))
            print(f'sim paths {sim_paths}')
            all_rslts.append(sim_paths)

        Kes.append(np.median(all_rslts))
        ke_q10s.append(np.percentile(all_rslts, 10))
        ke_q30s.append(np.percentile(all_rslts, 30))
        ke_q70s.append(np.percentile(all_rslts, 70))
        ke_q90s.append(np.percentile(all_rslts, 90))
        radii.append(radius)

    ax.fill_between(x=radii, y1=ke_q10s, y2=ke_q30s, color = 'blue', alpha=0.1)
    ax.fill_between(x=radii, y1=ke_q30s, y2=ke_q70s, color='blue', alpha=0.3)
    ax.fill_between(x=radii, y1=ke_q70s, y2=ke_q90s, color = 'blue',alpha=0.1)

    ax.plot(radii, Kes, 'o', color='tab:brown')
    # ax.set_title(f"Control on CSR Landscape, Cauchy Kernel: {Nhosts} type A Hosts on Square Length {length/1000} km\nEffect of Removal Radius on Final Size", fontsize=13)
    ax.tick_params(labelsize=11.5)
    ax.set_xlabel('Radius of host removal / m', fontsize=13)
    ax.set_ylabel(r"$Epidemic impact, K_{E}$", fontsize=13)
    ax.spines[['right', 'top']].set_visible(False)

    plt.savefig("radius_singlesps_sweep.svg")
    plt.show()
    return

def radius_responses(initial_cond, species_parameters, ctrl_parameters, num_simulations, landscapes, kernel_function,
                scale_parameter, radius_low, radius_high, radius_step, fname ):
    '''
    Plot responses to the varying radius of control. Produces an svg with two panels, one Ke and one AUDPC
    '''
    FONTSIZE =14
    POINTSIZE = 13
    [LIGHTEST, LIGHTISH, DARKISH, DARKEST] = [.2,.4,.6,.8] # this defines the transparencies for the fill indicating the distribution of the epidemic outcomes
    
    fig, (ax_upper, ax_lower) = plt.subplots(2, sharex=True, sharey=False)
    # fig.suptitle('Vertically stacked subplots')
    radii = []
    Kes = []; ke_q10s = [];  ke_q30s = [];  ke_q70s = [];  ke_q90s = [] # these store the epidemic impacts
    ke_q20s = []; ke_q40s = []; ke_q60s=  []; ke_q80s = []
    audpcs = []; audpcs_q10s = []; audpcs_q30s = []; audpcs_q70s = []; audpcs_q90s = []
    audpcs_q20s = []; audpcs_q40s = []; audpcs_q60s = []; audpcs_q80s = []

    num_landscapes = int(len(landscapes))

    nreplicates = int(num_simulations  / num_landscapes)
    print(f'performing {nreplicates} epidemics per each of {num_landscapes} landscapes\ntotal epidemics per landscape parameterisation {num_simulations} epidemics')
    
    for radius in np.arange(radius_low,radius_high + radius_step, step=radius_step):
        print(f'set radius {radius} m')
        ctrl_parameters = [90,1,1,1,1,radius, radius]
        all_rslts = []

        ke_replicate = []
        audpc_replicate = [] # temporarily stores the resulrs from 


        for lscape in landscapes:
            [epi_array, host_landscape] = setup_system(initial_cond, landscape=lscape) # assign the hosts to their initial states
            [transmission_kernel, epi_array, distance_matrix] = make_kernel(system_state=epi_array,
            landscape=lscape, kernel_function=kernel_function, scale_parameter=scale_parameter,
            species_parameters=species_parameters) # we construct our transmission kernel with all transmission constants
            
            sim_paths = gillespie_prll(initial_condition=initial_cond, landscape=lscape, kernel_function= kernel_function, 
                                    scale_parameter=scale_parameter, species_parameters=species_parameters, control_parameters= ctrl_parameters,
                                    control_on=True, control_start="random", return_fsize=True, return_audpc=True, return_simple_audpc=False, debug=False, 
                                    write_files=False, reused_kernel= transmission_kernel, return_t50=False, nsim=nreplicates, write_out=False)
            
            #sims.append(np.mean(sim_paths))
            print(f'sim paths {sim_paths}')
            np.savetxt('epis.epi', X = sim_paths) # for examination
            # now we need to calculate the 
            all_rslts.append(sim_paths)
            for replicate in sim_paths:
                # we need to scale the audpc....
                scaled_audpc = replicate[0]/1000
                audpc_replicate.append(scaled_audpc)
                ke_replicate.append(replicate[1])

        ke_q10s.append(np.percentile(ke_replicate, 10))
        ke_q20s.append(np.percentile(ke_replicate, 20))
        ke_q30s.append(np.percentile(ke_replicate, 30))
        ke_q40s.append(np.percentile(ke_replicate, 40))
        Kes.append(np.median(ke_replicate))
        ke_q60s.append(np.percentile(ke_replicate, 60))
        ke_q70s.append(np.percentile(ke_replicate, 70))
        ke_q80s.append(np.percentile(ke_replicate, 80))
        ke_q90s.append(np.percentile(ke_replicate, 90))

        # now we need to append the relevants for AUDPC...
        audpcs_q10s.append(np.percentile(audpc_replicate, 10))
        audpcs_q20s.append(np.percentile(audpc_replicate, 20))
        audpcs_q30s.append(np.percentile(audpc_replicate, 30))
        audpcs_q40s.append(np.percentile(audpc_replicate, 40))
        audpcs.append(np.median(audpc_replicate))
        audpcs_q60s.append(np.percentile(audpc_replicate, 60))
        audpcs_q70s.append(np.percentile(audpc_replicate, 70))
        audpcs_q80s.append(np.percentile(audpc_replicate, 80))
        audpcs_q90s.append(np.percentile(audpc_replicate, 90))

        radii.append(radius)

    ax_upper.fill_between(x=radii, y1=ke_q10s, y2=ke_q20s, color = 'blue', alpha=LIGHTEST, lw=0, label = "10th-20th percentile")
    ax_upper.fill_between(x=radii, y1=ke_q20s, y2=ke_q30s, color = 'blue', alpha=LIGHTISH, lw=0, label = "20th-30th percentile")
    ax_upper.fill_between(x=radii, y1=ke_q30s, y2=ke_q40s, color = 'blue', alpha=DARKISH, lw=0, label ="30th-40th percentile" )
    ax_upper.fill_between(x=radii, y1=ke_q40s, y2=Kes, color = 'blue', alpha=DARKEST, lw=0, label = "40th-50th percentile")
    ax_upper.fill_between(x=radii, y1=Kes, y2=ke_q60s, color = 'blue', alpha=DARKEST, lw=0, label = "50th-60th percentile")
    ax_upper.fill_between(x=radii, y1=ke_q60s, y2=ke_q70s, color='blue',   alpha=DARKISH, lw=0, label = "60th-70th percentile")
    ax_upper.fill_between(x=radii, y1=ke_q70s, y2=ke_q80s, color = 'blue', alpha=LIGHTISH, lw=0, label ="70th-80th percentile" )
    ax_upper.fill_between(x=radii, y1=ke_q80s, y2=ke_q90s, color = 'blue', alpha=LIGHTEST, lw=0, label ="80th-90th percentile" )
    ax_upper.scatter(radii, Kes, marker='o', color='tab:brown', s=POINTSIZE, label="Median value")

    # ax.set_title(f"Control on CSR Landscape, Cauchy Kernel: {Nhosts} type A Hosts on Square Length {length/1000} km\nEffect of Removal Radius on Final Size", fontsize=13)
    ax_upper.tick_params(labelsize=FONTSIZE)
    #ax_upper.set_xlabel('Radius of host removal / m', fontsize=FONTSIZE)

    ax_upper.set_ylabel(r"Epidemic impact $K_{E}$", fontsize=FONTSIZE)
    ax_upper.spines[['right', 'top']].set_visible(False)

    fig.supxlabel("Radius of removal /m", fontsize=FONTSIZE)
    # now for the lower axes... with AUDPC as response variable

    ax_lower.fill_between(x=radii, y1=audpcs_q10s, y2=audpcs_q20s, color = 'blue', alpha=LIGHTEST, lw=0)
    ax_lower.fill_between(x=radii, y1=audpcs_q20s, y2=audpcs_q30s, color = 'blue', alpha=LIGHTISH, lw=0)
    ax_lower.fill_between(x=radii, y1=audpcs_q30s, y2=audpcs_q40s, color = 'blue', alpha=DARKISH,lw= 0)
    ax_lower.fill_between(x=radii, y1=audpcs_q40s, y2=audpcs_q60s, color = 'blue', alpha=DARKEST,lw=0)
    ax_lower.fill_between(x=radii, y1=audpcs_q60s, y2=audpcs_q70s, color='blue',   alpha=DARKISH,lw=0)
    ax_lower.fill_between(x=radii, y1=audpcs_q70s, y2=audpcs_q80s, color = 'blue', alpha=LIGHTISH,lw=0)
    ax_lower.fill_between(x=radii, y1=audpcs_q80s, y2=audpcs_q90s, color = 'blue', alpha=LIGHTEST,lw=0)

    ax_lower.scatter(radii, audpcs, marker='o', color='tab:brown',s=POINTSIZE)
    # ax.set_title(f"Control on CSR Landscape, Cauchy Kernel: {Nhosts} type A Hosts on Square Length {length/1000} km\nEffect of Removal Radius on Final Size", fontsize=13)
    ax_lower.tick_params(labelsize=FONTSIZE)
    #ax_lower.set_xlabel('Radius of host removal / m', fontsize=FONTSIZE)
    ax_lower.set_ylabel(r"AUDPC (x 10$^3$)", fontsize=FONTSIZE)
    ax_lower.spines[['right', 'top']].set_visible(False)

    ax_upper.legend(frameon=False,bbox_to_anchor=(1.45, 0.5), loc='center right') # legend doesn't need a border and needs to be positioned outside of either plot
    plt.savefig(fname)

    #plt.show()
    return

"""
def radius_prob(N, r_low, r_high, r_step, pr_low, pr_high, prob_step, num_simulations, landscape, kernel_function
                , scale_parameter,initial_condition,species_A_parms, species_B_parms): # sweep both radius ans
    '''
    # Sweep both radius of removal and range of p_d (probability of detecting I).
    '''
    species_parameters= np.array(species_A_parms,species_B_parms)
    sims = []
    probs = []
    radius_array =  []
    fsizes = []

    [epi_array, host_landscape] = setup_system(initial_condition=initial_condition, landscape=landscape) # we only need to calculate the kernel once
    reused_kern = make_kernel(system_state=epi_array,
    landscape=host_landscape, kernel_function=kernel_function, scale_parameter=scale_parameter, species_parameters=species_parameters)

    for radius in np.arange(start=r_low,stop= r_high, step=r_step):
        print(f'radius = {radius}')
        for prob_A in np.arange(pr_low,pr_high, step=prob_step):
            probs.append(prob_A)
            radius_array.append(radius)

            print(f'prob A = {prob_A}, radius= {radius}')
            ctrl_params = [90, 1, prob_A, prob_B, radius_A, radius_B]
            print(f'beginning prob_A = {prob_A} epis')

            sim_paths = gillespie_prll(initial_condition, landscape, kernel_function, scale_parameter,
                species_parameters, ctrl_params, True, True, False, False, reused_kern, num_simulations)
            Efs = np.mean(sim_paths)
            fsizes.append(Efs)
        poss_optimal = sim_paths.index(np.min(sim_paths))
        radi_optimal =
        sims.append(( np.mean(sim_paths))/N)

    Z = np.array(sims)
    cma = plt.cm.get_cmap('viridis')
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    p = ax.scatter(x=np.array(probs),y=np.array(radius_array),s=200,c=Z, marker = 's', cmap =cma  );
    plt.title(f"Varying Probability of detection and Radius of Host removal\nMean Final Epidemic Size\n{num_simulations} epidemics / point")
    plt.xlabel("Probability of detection, p_d")
    plt.ylabel("Radius of host removal /m")
    cbar = plt.colorbar(p)
    cbar.set_label('Final Epidemic Size')
    plt.savefig('colour_fs.svg')
    plt.show()
    return
"""

def frac_B_radius_2d(p_d_B, frac_B_min, frac_B_max, frac_B_step, initial_condition,radius_low, radius_high,
                     radius_step, reps, scale_parameter, kernel_function
                     ,lscape_size, sps_parameters, num_landscapes,Nhosts):
    '''
    # Perform a 2D parameter scan on  proportion_B and the radius of control

    # will outline the 'best' management, which can then be resimulated in a 1D scan at better resolution

    ## Arguments
    -------

    p_d_B: probability of detection of species B
    '''
    control_start = 'random'
    medians = []
    radii = []
    frac_Bs = []

    for frac_B in np.arange(frac_B_min, frac_B_max+frac_B_step, frac_B_step):
        print(f'frac B is {frac_B}')
        n_B = int(Nhosts * frac_B)
        landscapes = landscape_generators.CSR.gen_csr(num_species=2, scaler=lscape_size,
                                                 nhosts_A=(Nhosts - n_B), nhosts_B = n_B,
                                                 nhosts_tot=Nhosts, num_landscapes=num_landscapes)
        control_on = True
        return_fsize = True
        debug = False
        write_files = False
        reused_kernel = None
        
        for radius in np.arange(radius_low, radius_high+radius_step, step=radius_step):
            # [survey_freq, frac_A, frac_B, prob_detec_A, prob_detec_B, radius_A, radius_B] = control_parameters
            control_parameters = [90, 1, 1, 1, 0.2, radius, radius]
            lc = 0
            fss = []
            for landscape in landscapes:
                lc += 1
                print(f'landscape {lc}/ {num_landscapes} at radius of removal = {radius}m')
                control_parameters = [90, 1, 1, 1, p_d_B, radius, radius]
                [epi_array, host_landscape] = setup_system(initial_condition, landscape=landscape) # assign the hosts to their initial states
                [transmission_kernel, epi_array, distance_matrix] = make_kernel(system_state=epi_array,
                landscape=landscape, kernel_function=kernel_function, scale_parameter=scale_parameter,
                species_parameters=sps_parameters) # we construct our transmission kernel with all transmission constants
                fsizes = gillespie_prll(initial_condition, landscape, kernel_function, scale_parameter
                        , sps_parameters, control_parameters, control_on, control_start, return_fsize, debug, write_files, reused_kernel, reps)
                fss.append(*fsizes) 

            median_fsize = np.median(fss)
            medians.append(median_fsize)
            frac_Bs.append(frac_B)
            radii.append(radius)
        
    Z = np.array(np.divide(medians,Nhosts))
    cma = plt.cm.get_cmap('plasma')
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    p = ax.scatter(x=np.array(radii),y=np.array(frac_Bs),s=200,c=Z, marker = 's', cmap =cma  );
    plt.title(f"Proportion(B) and Removal Radius\nMedian Final Epidemic Size\n{reps*num_landscapes} epidemics / point")
    plt.xlabel("Radius of host removal /m")
    plt.ylabel("Fraction of species B")
    cbar = plt.colorbar(p)
    cbar.set_label('Median Final Size')
    plt.savefig('colour_fs.svg')
    plt.show()
    return
    
def _two_radii(rA_low, rA_high, rB_low, rB_high, radius_step, initial_condition, reps, scale_parameter, kernel_function,
    lscape_size, sps_parameters, num_landscapes, Nhosts, frac_B, p_d_B):
    '''
    *DEPRECATED*
    
    Examine the effect of two-radius control. Note that  low-resolution initial scan may be required to progress to a higher res., second scan
    '''

    radii_A = np.arange(rA_low, rA_high, step=radius_step)
    radii_B = np.arange(rB_low, rB_high, step=radius_step)

    filename = f'two_radii_{scale_parameter}.txt'

    n_B = int(Nhosts * frac_B)

    landscapes = landscape_generators.CSR.gen_csr(num_species=2, scaler=lscape_size,
                                                nhosts_A=(Nhosts - n_B), nhosts_B=n_B,
                                                nhosts_tot=Nhosts, num_landscapes=num_landscapes)

    control_start = 'random'
    medians = []
    md_endt = []
    md_audpc = []
    md_audpc_sps = []
    frac_Bs = []

    try:
        with open(filename, 'w') as f:

            f.write(f'# 2D Parameter Scan for Two-Radius Control (res = {radius_step} m): Landscape Size {lscape_size}\n')
            f.write(f'# Nhosts = {Nhosts}; total host density = {Nhosts/((int(lscape_size)/1000)**2)} hosts/km^-2')
            f.write(f'# Kernel = {kernel_function}; Scale Parameter = {scale_parameter}m ')
            f.write(f'# Sps Parameters = {sps_parameters}\n')
            f.write(f'# control_start = {control_start}\n')
            f.write(f'# ######################################################\n')
            f.write('Frac_B\tN\tN_a\tN_b\tMed_KE_A\tMed_KE_B\tRadius_A\tRadius_B\tMed_KE\tQ1_KE\tQ3_KE\tMed_time\tQ1_endtime\tQ3_endtime\tMed_AUDPC\tQ1_AUDPC\tQ3_AUDPC\tMed_AUDPC_SPS\tQ1_AUDPC_SPS\tQ3_AUDPC_SPS\tQ1_AUDPC_A\tQ3_AUDPC_A\t \
                    Med_AUDPC_A\tQ1_AUDPC_B\tQ3_AUDPC_B\tMed_AUDPC_B\n')
            
            for radius_A in radii_A:
                print(f'*NPr: radius A  = {radius_A} m')

                control_on = True
                return_fsize = True
                debug = False
                write_files = False
                reused_kernel = None
                
                for radius_B in radii_B:
                    print(f'NPr: set radius B {radius_B} m')
                    lc = 0
                    fsizes = []
                    endtimes = []
                    audpc = []
                    sps_audpc = []  #holds the species-refined audpc 
                    audpc_a = [] # the type-specific area under disease progress curve
                    audpc_b = []
                    akes = [] # will hold the type-specific epidemic impact
                    bkes = []

                    for landscape in landscapes:
                        lc += 1
                        print(f'NPr: landscape {lc}/{num_landscapes} at radiusA = {radius_A}, radiusB = {radius_B}')
                        control_parameters = [90, 1, 1, 1, p_d_B, radius_A, radius_B]
                        [epi_array, host_landscape] = setup_system(initial_condition, landscape=landscape)  
                        [transmission_kernel, epi_array, distance_matrix] = make_kernel(system_state=epi_array,
                        landscape=landscape, kernel_function=kernel_function, scale_parameter=scale_parameter,
                        species_parameters=sps_parameters)  

                        rs = gillespie_prll(initial_condition, landscape, kernel_function, scale_parameter
                                , sps_parameters, control_parameters, control_on, control_start, return_fsize, True, True, debug, write_files, reused_kernel, True,reps)
                        for r in rs:
                            sps_audpc.append(r[0]) # the speciesspecific AUDPC
                            audpc.append(r[1])
                            fsizes.append(r[2])
                            endtimes.append(r[3])
                            a = r[4]
                            b = r[5]
                            audpc_a.append(a)
                            audpc_b.append(b)
                            [ake, bke] = r[6]
                            akes.append(ake) # the epidemic impact for the 
                            bkes.append(bke)
                            # now append the range of relevant values for the 

                    median_fsize = np.median(fsizes)
                    median_ake = np.median(akes)
                    median_bke = np.median(bkes)

                    median_audpc = np.median(audpc)
                    median_audpc_sps = np.median(sps_audpc)
                    median_endtimes = np.median(endtimes)

                    median_audpc_a = np.median(audpc_a)
                    median_audpc_b = np.median(audpc_b)

                    q25_audpc_a = np.percentile(audpc_a, 25)
                    q75_audpc_a = np.percentile(audpc_a, 75)

                    q25_audpc_b = np.percentile(audpc_b, 25)
                    q75_audpc_b = np.percentile(audpc_b, 75)                   

                    q25_audpc = np.percentile(audpc, 25)
                    q75_audpc = np.percentile(audpc, 75)

                    q25_audpc_sps = np.percentile(sps_audpc, 25)
                    q75_audpc_sps = np.percentile(sps_audpc, 75)
           
                    d25 = np.percentile(fsizes, 25)
                    d75 = np.percentile(fsizes, 75)

                    q25_endtime = np.percentile(endtimes,25)
                    q75_endtime = np.percentile(endtimes, 75)

                    md_endt.append(median_endtimes)
                    md_audpc.append(median_audpc)
                    md_audpc_sps.append(median_audpc_sps)
                    medians.append(median_fsize)

                    frac_Bs.append(frac_B)

                    f.write(f'{frac_B}\t{Nhosts}\t{Nhosts-n_B}\t{n_B}\t{median_ake}\t{median_bke}\t{radius_A}\t{radius_B} \
                            \t{round(median_fsize,1)}\t'
                            f'{round(d25,1)}\t{round(d75,1)}\t{round(median_endtimes,1)}\t{round(q25_endtime,1)}\t{round(q75_endtime,1)}\t'
                            f'{round(median_audpc,1)}\t{round(q25_audpc,1)}\t'
                            f'{round(q75_audpc,1)}\t{round(median_audpc_sps,1)}\t{round(q25_audpc_sps,1)}\t'
                            f'{round(q75_audpc_sps,1)}\t{round(q25_audpc_a,1)}\t{round(q75_audpc_a,1)}\t{round(median_audpc_a,1)}\t{round(q25_audpc_b,1)}\t{round(q75_audpc_b,1)}\t{round(median_audpc_b,1)}\n')
    except FileExistsError:
        print(f'the file already exists! Please move or delete the existing file before attempting another run')
        return
    return
    
def frac_B_radius_2dmod(p_d_B, frac_B_min, frac_B_max, frac_B_step, initial_condition, radius_low, radius_high,
                     radius_step, reps, scale_parameter, kernel_function, lscape_size, sps_parameters,
                     num_landscapes, Nhosts):
    '''
    # Perform a 2D parameter scan on  proportion_B and the radius of control

    # will outline the 'best' management, which can then be resimulated in a 1D scan at better resolution

    also has been modified to write out results to specific directories for reproducibility
    '''
    control_start = 'random'
    medians = []
    md_endt = []
    md_audpc = []
    md_audpc_sps = []
    radii = []
    frac_Bs = []


    filename = f"{scale_parameter}.txt" # constructing a vaguely specific filename 
    try:
        with open(filename, 'w') as f:
        # f.write(f'{frac_B}\t{radius}\t{round(median_fsize)}\t{median_endtimes}\t{median_audpc}\t{median_audpc_sps}\n')
            f.write(f'# 2D Parameter Scan for FracB and Single Radius Control: Landscape Size {lscape_size}\n')
            f.write(f'# Kernel = {kernel_function}; Scale Parameter = {scale_parameter}m ; Sps Parameters = {sps_parameters}\n')
            f.write(f'# control_start = {control_start}\n')
            f.write('Frac_B\tN\tN_a\tN_b\tMed_KE_A\tMed_KE_B\tRadius\tMed_KE\tQ1_KE\tQ3_KE\tMed_time\tQ1_endtime\tQ3_endtime\tMed_AUDPC\tQ1_AUDPC\tQ3_AUDPC\tMed_AUDPC_SPS\tQ1_AUDPC_SPS\tQ3_AUDPC_SPS\tQ1_AUDPC_A\tQ3_AUDPC_A\tMed_AUDPC_A\tQ1_AUDPC_B\tQ3_AUDPC_B\tMed_AUDPC_B\n')

            control_on = True
            return_fsize = True
            debug = False
            write_files = False
            reused_kernel = None

            for frac_B in np.arange(frac_B_min, frac_B_max+frac_B_step, step=frac_B_step):
                print(f'*NPr: frac B is {frac_B}')
                n_B = int(Nhosts * frac_B)
                landscapes = landscape_generators.CSR.gen_csr(num_species=2, scaler=lscape_size,
                                                            nhosts_A=(Nhosts - n_B), nhosts_B=n_B,
                                                            nhosts_tot=Nhosts, num_landscapes=num_landscapes)

                for radius in np.arange(radius_low, radius_high+radius_step, step=radius_step):
                    print(f'NPr: set radius {radius} m')
                    lc = 0
                    fss = []
                    fsizes = []
                    endtimes = []
                    audpc = []
                    sps_audpc = []  #holds the species-refined audpc 
                    audpc_a = [] # the type-specific area under disease progress curve
                    audpc_b = []
                    akes = [] # will hold the type-specific epidemic impact
                    bkes = []

                    for landscape in landscapes:
                        lc += 1
                        print(f'NPr: landscape {lc}/{num_landscapes} at radius of removal = {radius}m')
                        control_parameters = [90, 1, 1, 1, p_d_B, radius, radius]
                        [epi_array, host_landscape] = setup_system(initial_condition, landscape=landscape)  
                        [transmission_kernel, epi_array, distance_matrix] = make_kernel(system_state=epi_array,
                        landscape=landscape, kernel_function=kernel_function, scale_parameter=scale_parameter,
                        species_parameters=sps_parameters)  
                        rs = gillespie_prll(initial_condition, landscape, kernel_function, scale_parameter
                                , sps_parameters, control_parameters, control_on, control_start, return_fsize, True, True, debug, write_files, reused_kernel, True,reps)
                        for r in rs:
                            sps_audpc.append(r[0]) # the speciesspecific AUDPC
                            audpc.append(r[1])
                            fsizes.append(r[2])
                            endtimes.append(r[3])
                            a = r[4]
                            b = r[5]
                            audpc_a.append(a)
                            audpc_b.append(b)
                            [ake, bke] = r[6]
                            akes.append(ake) # the epidemic impact for the 
                            bkes.append(bke)
                            # now append the range of relevant values for the 

                    median_fsize = np.median(fsizes)
                    median_ake = np.median(akes)
                    median_bke = np.median(bkes)

                    median_audpc = np.median(audpc)
                    median_audpc_sps = np.median(sps_audpc)
                    median_endtimes = np.median(endtimes)

                    median_audpc_a = np.median(audpc_a)
                    median_audpc_b = np.median(audpc_b)

                    q25_audpc_a = np.percentile(audpc_a, 25)
                    q75_audpc_a = np.percentile(audpc_a, 75)

                    q25_audpc_b = np.percentile(audpc_b, 25)
                    q75_audpc_b = np.percentile(audpc_b, 75)                   

                    q25_audpc = np.percentile(audpc, 25)
                    q75_audpc = np.percentile(audpc, 75)

                    #d10_audpc = np.percentile(audpc,10)
                    #d30_audpc = np.percentile(audpc,30)
                    #d70_audpc = np.percentile(audpc,70)
                    #d90_audpc = np.percentile(audpc,90)

                    q25_audpc_sps = np.percentile(sps_audpc, 25)
                    q75_audpc_sps = np.percentile(sps_audpc, 75)
           
                    #d10_audpc_sps = np.percentile(sps_audpc,10)
                    #d30_audpc_sps = np.percentile(sps_audpc,30)
                    #d70_audpc_sps = np.percentile(sps_audpc,70)
                    #d90_audpc_sps = np.percentile(sps_audpc,90)

                    # d10 = np.percentile(fsizes, 10)
                    d25 = np.percentile(fsizes, 25)
                    d75 = np.percentile(fsizes, 75)
                    # d90 = np.percentile(fsizes, 90)

                    q25_endtime = np.percentile(endtimes,25)
                    q75_endtime = np.percentile(endtimes, 75)
                    #d30_endtime = np.percentile(endtimes,30)
                    #d70_endtime = np.percentile(endtimes,70)
                    #d90_endtime = np.percentile(endtimes,90)

                    md_endt.append(median_endtimes)
                    md_audpc.append(median_audpc)
                    md_audpc_sps.append(median_audpc_sps)
                    medians.append(median_fsize)

                    frac_Bs.append(frac_B)
                    radii.append(radius)
                    print(f'Frac_B: {frac_B}, Radius: {radius}, Median Final Size: {round(median_fsize)}')

                    f.write(f'{frac_B}\t{Nhosts}\t{Nhosts-n_B}\t{n_B}\t{median_ake}\t{median_bke}\t{radius}\t{round(median_fsize,1)}\t'
                            f'{round(d25,1)}\t{round(d75,1)}\t{round(median_endtimes,1)}\t{round(q25_endtime,1)}\t{round(q75_endtime,1)}\t'
                            f'{round(median_audpc,1)}\t{round(q25_audpc,1)}\t'
                            f'{round(q75_audpc,1)}\t{round(median_audpc_sps,1)}\t{round(q25_audpc_sps,1)}\t'
                            f'{round(q75_audpc_sps,1)}\t{round(q25_audpc_a,1)}\t{round(q75_audpc_a,1)}\t{round(median_audpc_a,1)}\t{round(q25_audpc_b,1)}\t{round(q75_audpc_b,1)}\t{round(median_audpc_b,1)}\n')
                    # f.write('Frac_B\tRadius\tMed_KE\tQ1_KE\tQ3_KE\tMed_time\tQ1_endtime\tQ3_endtime\tMed_AUDPC\tQ1_AUDPC\tQ3_AUDPC\tMed_AUDPC_SPS\tQ1_AUDPC_SPS\tQ3_AUDPC_SPS\n')

    except FileExistsError:
        print(f'the file already exists! Please move or delete the existing file before attempting another run')
        return
    
    Z = np.array(np.divide(medians, Nhosts))
    cma = plt.cm.get_cmap('plasma')
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    p = ax.scatter(x=np.array(radii), y=np.array(frac_Bs), s=200, c=Z, marker='s', cmap=cma)
    unique_frac_Bs = np.unique(frac_Bs)
    for frac_B in unique_frac_Bs:
        indices = [i for i, x in enumerate(frac_Bs) if x == frac_B]
        min_value_index = indices[np.argmin(np.array(medians)[indices])]
        min_radius = radii[min_value_index]
        ax.scatter(min_radius, frac_B, s=250, edgecolors='green', linewidths=2, facecolors='none') 

    plt.title(f"Proportion(B) and Removal Radius\nMedian Final Epidemic Size\n{reps*num_landscapes} epidemics / point")
    plt.xlabel("Radius of host removal /m")
    plt.ylabel("Fraction of species B")
    cbar = plt.colorbar(p)
    cbar.set_label('Median Final Size')
    plt.savefig('fs_optimality_highlig.svg')
    plt.show()


    # now we can plot T_ends
    Z = np.array(md_endt)
    cma = plt.cm.get_cmap('plasma')
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    p = ax.scatter(x=np.array(radii), y=np.array(frac_Bs), s=200, c=Z, marker='s', cmap=cma)
    unique_frac_Bs = np.unique(frac_Bs)
    for frac_B in unique_frac_Bs:
        indices = [i for i, x in enumerate(frac_Bs) if x == frac_B]
        min_value_index = indices[np.argmin(np.array(md_endt)[indices])]
        min_radius = radii[min_value_index]
        ax.scatter(min_radius, frac_B, s=250, edgecolors='green', linewidths=2, facecolors='none') 

    plt.title(f"Proportion(B) and Removal Radius\nMedian  Epidemic End Time\n{reps*num_landscapes} epidemics / point")
    plt.xlabel("Radius of host removal /m")
    plt.ylabel("Fraction of species B")
    cbar = plt.colorbar(p)
    cbar.set_label('Median Epidemic End Time /days')
    plt.savefig('epi_endtimeplot.svg')
    plt.show()

    Z = np.array(md_audpc)
    cma = plt.cm.get_cmap('plasma')
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    p = ax.scatter(x=np.array(radii), y=np.array(frac_Bs), s=200, c=Z, marker='s', cmap=cma)
    unique_frac_Bs = np.unique(frac_Bs)
    for frac_B in unique_frac_Bs:
        indices = [i for i, x in enumerate(frac_Bs) if x == frac_B]
        min_value_index = indices[np.argmin(np.array(md_audpc)[indices])]
        min_radius = radii[min_value_index]
        ax.scatter(min_radius, frac_B, s=250, edgecolors='green', linewidths=2, facecolors='none') 

    plt.title(f"Proportion(B) and Removal Radius\n AUDPC \n{reps*num_landscapes} epidemics / point")
    plt.xlabel("Radius of host removal /m")
    plt.ylabel("Fraction of species B")
    cbar = plt.colorbar(p)
    cbar.set_label('AUDPC')
    plt.savefig('audpc.svg')
    plt.show()
    # and also do similar plots for other metrics
    # what about compound metrics
    return

def frac_B_radius_ns(p_d_B, frac_B_min, frac_B_max, frac_B_step, initial_condition, radius_low, radius_high,
                     radius_step, reps, scale_parameter, kernel_function, lscape_size, sps_parameters,
                     num_landscapes, Nhosts):
    '''
    # Perform a 2D parameter scan on  proportion_B and the radius of control: Neyman Scott landscape

    # will outline the 'best' management, which can then be resimulated in a 1D scan at better resolution

    also has been modified to write out results to specific directories for reproducibility
    '''
    control_start = 'random'
    control_on = True
    return_fsize = True
    debug = False
    write_files = False
    reused_kernel = None

    medians = []
    md_endt = []
    md_audpc = []
    md_audpc_sps = []
    radii = []
    frac_Bs = []


    filename = f"{scale_parameter}_ns_landscape.txt" # constructing a vaguely specific filename 
    try:
        with open(filename, 'w') as f:
        # f.write(f'{frac_B}\t{radius}\t{round(median_fsize)}\t{median_endtimes}\t{median_audpc}\t{median_audpc_sps}\n')
            f.write(f'# 2D Parameter Scan for FracB and Single Radius Control: Landscape Size {lscape_size}\n')
            f.write(f'# Neyman Scott Landscape \n')
            f.write(f'# Kernel = {kernel_function}; Scale Parameter = {scale_parameter}m ; Sps Parameters = {sps_parameters}\n')
            f.write(f'# control_start = {control_start}\n')
            f.write('Frac_B\tN\tN_a\tN_b\tMed_KE_A\tMed_KE_B\tRadius\tMed_KE\tQ1_KE\tQ3_KE\tMed_time\tQ1_endtime\tQ3_endtime\tMed_AUDPC\tQ1_AUDPC\tQ3_AUDPC\tMed_AUDPC_SPS\tQ1_AUDPC_SPS\tQ3_AUDPC_SPS\tQ1_AUDPC_A\tQ3_AUDPC_A\tMed_AUDPC_A\tQ1_AUDPC_B\tQ3_AUDPC_B\tMed_AUDPC_B\n')

            for frac_B in np.arange(frac_B_min, frac_B_max+frac_B_step, step=frac_B_step):
                print(f'*NPr: frac B is {frac_B}')
                n_B = int(Nhosts * frac_B)
                #landscapes = landscape_generators.CSR.gen_csr(num_species=2, scaler=lscape_size,
                #                                            nhosts_A=(Nhosts - n_B), nhosts_B=n_B,
                #                                            nhosts_tot=Nhosts, num_landscapes=num_landscapes)
                landscapes = landscape_generators.neymanscott.get_ns(num_species=2, scaler=lscape_size, frac_B=frac_B, nhosts_tot=Nhosts, num_landscapes=5)
                
                for radius in np.arange(radius_low, radius_high+radius_step, step=radius_step):
                    print(f'NPr: set radius {radius} m')
                    lc = 0
                    fsizes = []
                    endtimes = []
                    audpc = []
                    sps_audpc = []  #holds the species-refined audpc 
                    audpc_a = [] # the type-specific area under disease progress curve
                    audpc_b = []
                    akes = [] # will hold the type-specific epidemic impact
                    bkes = []

                    for landscape in landscapes:
                        lc += 1
                        print(f'NPr: landscape {lc}/{num_landscapes} at radius of removal = {radius}m')
                        control_parameters = [90, 1, 1, 1, p_d_B, radius, radius]
                        [epi_array, host_landscape] = setup_system(initial_condition, landscape=landscape)  
                        [transmission_kernel, epi_array, distance_matrix] = make_kernel(system_state=epi_array,
                        landscape=landscape, kernel_function=kernel_function, scale_parameter=scale_parameter,
                        species_parameters=sps_parameters)  
                        rs = gillespie_prll(initial_condition, landscape, kernel_function, scale_parameter
                                , sps_parameters, control_parameters, control_on, control_start, 
                                return_fsize, True, True, debug, write_files, reused_kernel, 
                                True,reps)
                        for r in rs:
                            sps_audpc.append(r[0]) # the speciesspecific AUDPC
                            audpc.append(r[1])
                            fsizes.append(r[2])
                            endtimes.append(r[3])
                            a = r[4]
                            b = r[5]
                            audpc_a.append(a)
                            audpc_b.append(b)
                            [ake, bke] = r[6]
                            akes.append(ake) # the epidemic impact for the 
                            bkes.append(bke)
                            # now append the range of relevant values for the 

                    median_fsize = np.median(fsizes)
                    median_ake = np.median(akes)
                    median_bke = np.median(bkes)

                    median_audpc = np.median(audpc)
                    median_audpc_sps = np.median(sps_audpc)
                    median_endtimes = np.median(endtimes)

                    median_audpc_a = np.median(audpc_a)
                    median_audpc_b = np.median(audpc_b)

                    q25_audpc_a = np.percentile(audpc_a, 25)
                    q75_audpc_a = np.percentile(audpc_a, 75)

                    q25_audpc_b = np.percentile(audpc_b, 25)
                    q75_audpc_b = np.percentile(audpc_b, 75)                   

                    q25_audpc = np.percentile(audpc, 25)
                    q75_audpc = np.percentile(audpc, 75)

                    #d10_audpc = np.percentile(audpc,10)
                    #d30_audpc = np.percentile(audpc,30)
                    #d70_audpc = np.percentile(audpc,70)
                    #d90_audpc = np.percentile(audpc,90)

                    q25_audpc_sps = np.percentile(sps_audpc, 25)
                    q75_audpc_sps = np.percentile(sps_audpc, 75)
           
                    #d10_audpc_sps = np.percentile(sps_audpc,10)
                    #d30_audpc_sps = np.percentile(sps_audpc,30)
                    #d70_audpc_sps = np.percentile(sps_audpc,70)
                    #d90_audpc_sps = np.percentile(sps_audpc,90)

                    #d10 = np.percentile(fsizes, 10)
                    d25 = np.percentile(fsizes, 25)
                    d75 = np.percentile(fsizes, 75)
                    #d90 = np.percentile(fsizes, 90)

                    q25_endtime = np.percentile(endtimes,25)
                    q75_endtime = np.percentile(endtimes, 75)
                    #d30_endtime = np.percentile(endtimes,30)
                    #d70_endtime = np.percentile(endtimes,70)
                    #d90_endtime = np.percentile(endtimes,90)

                    md_endt.append(median_endtimes)
                    md_audpc.append(median_audpc)
                    md_audpc_sps.append(median_audpc_sps)
                    medians.append(median_fsize)

                    frac_Bs.append(frac_B)
                    radii.append(radius)

                    print(f'Frac_B: {frac_B}, Radius: {radius}, Median Final Size: {round(median_fsize)}')

                    f.write(f'{frac_B}\t{Nhosts}\t{Nhosts-n_B}\t{n_B}\t{median_ake}\t{median_bke}\t{radius}\t{round(median_fsize,1)}\t'
                            f'{round(d25,1)}\t{round(d75,1)}\t{round(median_endtimes,1)}\t{round(q25_endtime,1)}\t{round(q75_endtime,1)}\t'
                            f'{round(median_audpc,1)}\t{round(q25_audpc,1)}\t'
                            f'{round(q75_audpc,1)}\t{round(median_audpc_sps,1)}\t{round(q25_audpc_sps,1)}\t'
                            f'{round(q75_audpc_sps,1)}\t{round(q25_audpc_a,1)}\t{round(q75_audpc_a,1)}\t{round(median_audpc_a,1)}\t{round(q25_audpc_b,1)}\t{round(q75_audpc_b,1)}\t{round(median_audpc_b,1)}\n')
                    # f.write('Frac_B\tRadius\tMed_KE\tQ1_KE\tQ3_KE\tMed_time\tQ1_endtime\tQ3_endtime\tMed_AUDPC\tQ1_AUDPC\tQ3_AUDPC\tMed_AUDPC_SPS\tQ1_AUDPC_SPS\tQ3_AUDPC_SPS\n')
    except FileExistsError:
        print(f'the file already exists! Please move or delete the existing file before attempting another run')
        return
    return

def vary_B(nhosts, sps_parms, ctrl_params, initial_condition ,kernel, side_scale ,scale_parameter, resolution, nlandscapes, nreplicates):
    '''
    explores the effect of increasing the proportion of species B on the expected final epidemic size
    (ie, misspecification)

    1D scan, with two species

    ## Arguments
    -----------
    '''
    [survey_freq, frac_A, frac_B, prob_detec_A, prob_detec_B, radius_A, radius_B] = ctrl_params

    print(f'***Beginning 1D sweep of effect of proportion of type B host on final epidemic size***')
    print(f'{nlandscapes} landscapes each with {nreplicates} replicates\nTotal {nlandscapes*nreplicates*len(np.arange(0,1, step=resolution))} epis')
    prop_Bs = []
    median_fs = []
    fsizes = []
    d30s= []
    d70s  =[]
    for prop_B in np.arange(0,1+resolution, step=resolution): # the fraction
        # we also need to compute mean p_d here to illustrate the point that population heterogeneity matters.
         # mean_pd = pd / nhosts
        num_B = int(prop_B*nhosts)
        num_A = nhosts - num_B

        lscapes = landscape_generators.CSR.gen_csr(num_species=2, scaler=side_scale,
                                                   nhosts_A=int(num_A), nhosts_B=int(num_B),
                                                   nhosts_tot=int(num_A + num_B), num_landscapes=nlandscapes)

        lscape_results = [] # these will hold the results for this landscape parameter set
        refined_audpcs = []
        audpcs = []
        endtimes = []

        for lscape in lscapes: # we need to consider several simulated landscape instances to average out variability
            [epi_array, host_landscape] = setup_system(initial_condition=initial_condition, landscape=lscape) # we only need to calculate the kernel once
            reused_kernel = make_kernel(system_state=epi_array,
        landscape=host_landscape, kernel_function=kernel, scale_parameter=scale_parameter,
        species_parameters=sps_parms)
            #print(f'species parameters are {PS1.sps_parms}')
            print(f'frac B is {prop_B}')

            sim_paths = gillespie_prll(initial_condition, lscape, kernel, scale_parameter, sps_parms, ctrl_params, True, 'random',
                                       True, True ,False, False, reused_kernel, nreplicates)
            
            for sim_result in sim_paths: # for each replicate...
                refined_audpcs.append(sim_result[0]) # the speciesspecific AUDPC
                audpcs.append(sim_result[1])
                fsizes.apend(sim_result[2])
                endtimes.append(sim_result[3])

            lscape_results.append(sim_paths)
        #print(f'lscape results before flattening is {lscape_results}')
        #lscape_results = np.flatten(lscape_results)
        d10 = np.percentile(fsizes, 10)
        d30 = np.percentile(fsizes, 30)
        d70 = np.percentile(fsizes, 70)
        d90 = np.percentile(fsizes, 90)
        median_fsize = np.median(fsizes)

        # should also calculate Q1, Q3, epidemic time, audpc
        d30s.append(d30)
        d70s.append(d70)

        median_fs.append(median_fsize)
        fsizes.append(sim_paths) # raw data
        prop_Bs.append(prop_B) # append only once for each proportion of species B
    sns.set_theme()
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots()
    ax.fill_between(x=prop_Bs, y1=d30s, y2=d70s, alpha=0.2)
    ax.plot(prop_Bs, median_fs, 'o', color='tab:brown')
    plt.xlabel("Proportion of species B ")
    plt.ylabel(f"Median final epidemic size, {nreplicates} epidemics")
    plt.title("Effect of proportion of species B on median final epidemic size, f_s\nMisspecified Control")
    plt.savefig("prop_B_feb_frac.svg")
    plt.show()
    return

'''

#######################################################################
if __name__ == "__main__":
    initial_condition = [1100,10,0,0]
    # [survey_freq, frac_A, frac_B, prob_detec_A, prob_detec_B, radius_A, radius_B] = control_parameters
    control_parameters = [90, 1,1, 1, 0.2, 425, 425] 
    scale_parameter = 84.5
    sps_parms = np.array([[0.0068, 0.14, (1/100), 0],
                          [0.0068, 0.14, (1/100), 0]])
    


    #sps_parms = np.array([[0.00785, 0.14, (1/350), 0], # 4km by 4km for 
    #                       [0.00785, 0.14, (1/350), 0]])

    vary_B(int(np.sum(initial_condition)), sps_parms, control_parameters, initial_condition,cauchy_thick, 3500 ,84.5, 0.1, 10, 10)

if __name__ == "__main__": #31 jan
    initial_cond= [1100,10,0,0]
    species_parameters = np.array([[0.00785, 0.14, (1/350), 0],
                          [0.00785, 0.14, (1/350), 0]])
    
    control_parameters = [90, 1,1, 1, 0.2, 425, 425] 
    num_simulations = 200
    scaler = 4000
    landscape = landscape_generators.CSR.gen_csr(2, scaler, int(1110),0, 1110,1)
    kernel_function = cauchy_thick
    scale_parameter = 84.5
    radius_low = 200
    radius_high = 1100
    radius_step = 40
    radius_plot(initial_cond, species_parameters, control_parameters, num_simulations, landscape, kernel_function,
                scale_parameter, radius_low, radius_high, radius_step, scaler)

if __name__ == "__main__":
    p_d_B = 0.2
    frac_B_min = 0
    frac_B_max = 1
    frac_B_step = 0.2
    initial_condition =  [1100,10,0,0]
    radius_low = 0
    radius_high = 1200
    radius_step = 50
    reps = 50
    scale_parameter = 119
    kernel_function = exponential_kernel
    lscape_size = 4000

    #sps_parameters = np.array([[0.02     ,  0.09      , 0.00285714 ,0.00053   ], [0.02    ,   0.14     ,  (1/500),0.00053   ]])
    #sps_parameters  = np.array([[0.00833, 0.14 ,(1/350), 0.00053 ] ,[0.00833 ,0.14, (1/500), 0.00053 ]])
    
    #sps_parameters = np.array([[0.00785, 0.14, (1/350), 0.00053], [0.00785, 0.14, (1/500), 0.00053]]) 
    #sps_parameters = np.array([[0.015, 0.14, (1/350), 0.00053], [0.015, 0.14, (1/500), 0.00053]]) 
    num_landscapes = 5
    Nhosts = int(np.sum(initial_condition))


    #frac_B_radius_2d(p_d_B, frac_B_min, frac_B_max, frac_B_step, initial_condition,radius_low, radius_high,
    #
    #                     radius_step, reps, scale_parameter, kernel_function
     
     #                   ,lscape_size, sps_parameters, num_landscapes,Nhosts)
    
    #frac_B_radius_2dmod(p_d_B, frac_B_min, frac_B_max, frac_B_step, initial_condition,radius_low, radius_high,
    #                    radius_step, reps, scale_parameter, kernel_function
    #                    ,lscape_size, sps_parameters, num_landscapes,Nhosts)       


    frac_B_radius_ns(p_d_B, frac_B_min, frac_B_max, frac_B_step, initial_condition,radius_low, radius_high,
                        radius_step, reps, scale_parameter, kernel_function
                        ,lscape_size, sps_parameters, num_landscapes,Nhosts)      
    
'''

if __name__ == "__main__":
    initial_cond =  [1100,10,0,0]
    species_parameters =  np.array([[0.015, 0.14 ,0.00285714 ,0.00053 ] ,[0.015, 0.14, 0.00285714 ,0.00053 ]] )
    ctrl_parameters = [90, 1, 1, 1, 0.2, 0, 0]
    
    # verify that this parameterisation is correct...
    radius_low = 0
    radius_high = 1200
    radius_step = 50
    kernel_function = exponential_kernel
    scale_parameter = 119
    num_simulations = 250
    number_landscapes = 5
    landscapes = landscape_generators.CSR.gen_csr(2, 4000,1110,0,1110,number_landscapes)

    radius_responses(initial_cond, species_parameters, ctrl_parameters, num_simulations, landscapes, kernel_function,
                scale_parameter, radius_low, radius_high, radius_step, fname = "16km_response_smallerpts.svg")
    
    #landscapes = landscape_generators.CSR.gen_csr(2, 3000,1110,0,1110,number_landscapes)

    #radius_responses(initial_cond, species_parameters, ctrl_parameters, num_simulations, landscapes, kernel_function,
    #            scale_parameter, radius_low, radius_high, radius_step, fname = "9km_response_exp.svg")
    


    