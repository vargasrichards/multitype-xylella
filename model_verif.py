'''
Model verification using analytical deterministic comparison to ensembles of stochastic runs
Can also be used just to generate deterministic plots and stochastic plots

Also updated to contain code for generating histograms of expected final size

A. Vargas Richards, Dec. 2024. Jan. , Feb 2025
'''
try:
    import numpy as np, math, pandas as pd
    from scipy.spatial.distance import squareform, pdist
    from scipy.integrate import odeint
    import matplotlib.pyplot as plt
    from gillespie_dsa import *# provides the main stochastic alg., for either single or twospecies models
    from collections import Counter
    import scir_plotting
    import seaborn as sns
    import landscape_generators
    import multiprocessing as mp
    
except ImportError as ie:
    print(ie)

[HOST_ID, SPS_ID, INFEC_STATUS, T_RATE, T_SC, T_CI, T_IR] = [0, 1, 2, 3, 4, 5, 6]
[STATUS_S, STATUS_C, STATUS_I, STATUS_R] = [0,1,2,3]

class hyatttwynham:
    cauchy_optimum = 31
    exponential_optimum = 36
    cauchy_beta = 0.00036
    exp_beta = 0.0009 
    cauchy_sigma  =1/107
    alpha = 36.1
    cauch_sps_parameters = np.array([[math.sqrt(cauchy_beta),math.sqrt(cauchy_beta) , cauchy_sigma, 0],[0, 0, (0) , 0]])
    exp_sps_params = np.array([[math.sqrt(exp_beta),math.sqrt(exp_beta) , cauchy_sigma, 0],[0, 0, (0) , 0]])
    cauchy_control_parameters = [90, 1, 1, 1, cauchy_optimum, cauchy_optimum] 
    exponential_control_parms =  [90, 1, 1, 1, exponential_optimum, exponential_optimum] 


class kernels:
    def flat_kernel(distance_matrix, beta): #
        '''
        flat dispersal kernel which makes the effect of space zero.
        nb. NO SCALE PARAMETER, BETA is to normalise and corresponds to the nonspatial beta in SCIR analytical model
        '''
        return np.ones(np.shape(distance_matrix))
        return np.multiply(np.ones(np.shape(distance_matrix)), beta)

class test_landscapes:
    '''
    This class provides methods for the construction/reading in of host landscapes.
    '''
    def read_landscape (filestring):
        '''
        Read in a set of x, y coordinates from a filestring.
        For multispecies/  landscapes with more detail see elsewhere.
        '''
        return squareform(pdist(pd.read_csv(filestring, header=None)))

class single_species:
    '''
    Analytical ODE solvers for the SCIR model with arbitrary beta, sigma, gamma, neglecting space or control.
    '''
    def scir_nocont_singlesps(y, t, N, beta, sigma, gamma): # 
        '''
        Computes derivatives for current system state, based on mass action. 
        
        Cryptics and Symptomatics are treated as equivalent here, and both contribute equally to the force of infection onto susceptibles
        '''
        S, C, I, R = y
        dSdt = -beta *(C + I)*S
        dCdt = beta*(C + I)* S - sigma*C
        dIdt = sigma*C  - gamma * I
        dRdt = gamma * I
        return dSdt, dCdt, dIdt, dRdt

    def plot_singlesps(initial_cond, t_end, beta, sigma, gamma, helper_mode): # beta =  0.000016
        '''
        Plot the analytical solution for a nonspatial SCIR compartmental model. User specifies arbitrary initial condition.
        No control here, see elsewhere for simulated roguing / host-removal.

        '''
        [S0, C0, I0, R0] = initial_cond
        N = S0 + C0 + I0 + R0        
        t = np.linspace(0, t_end, t_end)
        y0 = S0, C0, I0, R0
        ret = odeint(single_species.scir_nocont_singlesps, y0, t, args=(N, beta, sigma, gamma))
        S, C, I, R = ret.T
        either_ci = C + I  # total of infected hosts

        fig = plt.figure()
        sns.set_style('whitegrid')
        ax = fig.add_subplot()
        #ax.plot(t, S, 'b--', label='Susceptible') # i've distinguished these by line style so hopefully the fig. should still work for colourblind ppl or in b/w.
        ax.plot(t, C, 'magenta', label='Cryptically Infected')
        ax.plot(t, I, 'green',label='Symptomatically Infected')
        ax.plot(t, either_ci, 'g--', label = 'Total Infected ')
        #ax.plot(t, R, 'black',  label='Removed') # don't think there's any point in plotting this since it's empty and the legend will be too big
        
        if helper_mode:
            return ax
        
        ax.set_ylabel('Hosts')
        ax.scatter(500, N/2, label="1/2 hosts infected, t = 500d")

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.suptitle(f"Deterministic SCIR Solution: No Control Implemented\n\n", size=12)
        plt.xlabel(u'Length ${\mu}m$')

        titlestr = str(f'N = {N},') +  r"$\beta$ = " + str(beta) + r"days$^{-1}$, " +str(u" , ${\\sigma}$ = ")+ str (u" 1/107, ") + "days$^-1" +str(u" ${\\gamma}$ = ") + str(gamma) + "days$^-1$" + str(f'\nInitial Condition: 10 C, 1101 S')
        plt.title(r"$\beta$ = "+ f"{beta}" +  r"$\mathrm{days}^{-1}$" + r"$\sigma = $"+ r"$\frac{1}{107}$, "
                  + r"$\mathrm{days}^{-1}$" + r"$\gamma = $" + f"{gamma}" + r"$\mathrm{days}^{-1}$"
                  + f"\nInitial Condition = {initial_cond}", size = 12)
        plt.legend(frameon = False)
        plt.xlabel("Time /days")
        plt.ylabel("Number of hosts")
        plt.savefig("deterministic_singlesps_calib.svg")
        plt.show()
        return
    
class comparisons:

    def check_correspondence(initial_cond, fitted_beta,sigma,  nsample_paths, stoch_alpha ):

        '''
        Plots a user-specified number of stochastic sample paths along with the deterministic case.
        Suggested inputs:
        fitted_beta = 0.00000845
        initial_cond = [1101,10,0,0]
        n.b. ctrl_params doesn't matter as CONTROL IS OFF...
        '''
        ctrl_params = [45,90,1,1,31,31]
     
        #spcs_params = np.array([[math.sqrt(fitted_beta), math.sqrt(fitted_beta), (1/107), 0],[0,0, (1/107) , 0]])
        spcs_params= np.array([[np.sqrt(fitted_beta),np.sqrt(fitted_beta), sigma, 0],[0,0,sigma , 0]])
        
        fig, ax = plt.subplots()
        ax = single_species.plot_singlesps(initial_cond, 1500, fitted_beta, sigma, 0, True)

        for _ in range(nsample_paths):
            epi = gillespie(initial_cond, kernels.flat_kernel, fitted_beta,
                        scir_plotting.citrus_pos, spcs_params, ctrl_params, False,False, False, False)
            path = scir_plotting.make_path(epi)
            [S_path, C_path, I_path, R_path, times] = path
            plt.plot(times, I_path, 'green', alpha=stoch_alpha)
            plt.plot(times, C_path, 'magenta', alpha=stoch_alpha)

        sns.set_context("paper")
        sns.set_theme(style="whitegrid")
        plt.grid(None)
        sns.despine()
        plt.legend()
        plt.title("Nonspatial SCIR: Comparison of Deterministic and Stochastic Models")
        plt.xlabel("Time /days")
        plt.ylabel("Number of hosts")
        plt.savefig("stoch_det.svg")
        plt.show()
        return plt

class two_species:
    def scir_nocont_multisp(y,t,N, species_parameters):
        '''
        Plot deterministic paths for the two-species case, with potential for asymmetric transmission
        '''
        S_s0, C_s0, I_s0, R_s0, S_s1, C_s1, I_s1, R_s1  = y

        [[nu0, phi0, sigma0, gamma0],[nu1, phi1, sigma1, gamma1]] = species_parameters # we take this directly from the gillespie multiple species
        dS_0_dt = - phi0 * S_s0 * (nu0 *(C_s0 + I_s0) + nu1 *(C_s1 + I_s1) )
        dC_0_dt = (phi0 * S_s0 * (nu0 *(C_s0 + I_s0) + nu1 *(C_s1 + I_s1) )) - sigma0*C_s0
        dI_0_dt = sigma0*C_s0  - gamma0 * I_s0
        dR_0_dt = gamma0 * I_s0

        dS_1_dt = - phi1 * S_s1 * (nu0 *(C_s0 + I_s0) + nu1 *(C_s1 + I_s1) )
        dC_1_dt = (phi1 * S_s1 * (nu0 *(C_s0 + I_s0) + nu1 *(C_s1 + I_s1)) )- sigma1*C_s1
        dI_1_dt = sigma1*C_s1 - gamma1 * I_s1
        dR_1_dt = gamma1 * I_s1

        return dS_0_dt, dC_0_dt, dI_0_dt, dR_0_dt, dS_1_dt, dC_1_dt, dI_1_dt, dR_1_dt
    
    def plot_multisps(initial_cond, t_end, sps_parms, helper_mode):  # plot the dynamics of the multispecies
        '''
        Plot the deterministic solution of the nonspatial multispecies case, with potential for asymmetric transmission parms.
        If `helper_mode` = True, then no plot is produced, simply the solution paths `S_0, C_0, I_0, R_0, S_1, C_1, I_1, R_1 = ret.T`
        are returned. This is so that it can be called by a different function 
        '''
        [St_0, Ct_0, It_0, Rt_0, St_1, Ct_1, It_1, Rt_1] = initial_cond
        N = np.sum(initial_cond)
        print(f'N = {N}')
        N_sps0 = St_0 +  Ct_0 +  It_0 + Rt_0
        N_sps1 = St_1 +  Ct_1 +  It_1 + Rt_1
        print(f'Number of species 0 = {N_sps0}\nNumber of species 1 = {N_sps1}')
        t = np.linspace(0, t_end, t_end)
        print(f'solving until t_end = {t_end}')
        y0 = St_0, Ct_0, It_0, Rt_0, St_1, Ct_1, It_1, Rt_1
        ret = odeint(two_species.scir_nocont_multisp, y0, t, args=(N, sps_parms))
        S_0, C_0, I_0, R_0, S_1, C_1, I_1, R_1 = ret.T
        either_ci_0 = C_0 + I_0 
        either_ci_1 = C_1 + I_1  
        total_infected = either_ci_0 + either_ci_1
        fig = plt.figure()
        sns.set_theme(style="whitegrid")
        ax = fig.add_subplot()
        #ax.plot(t, S_0, 'b--', label='S, sp. 0') # removed the susceptible plots to help readability
        #ax.plot(t, S_1, 'b', label='S, sp. 1')
        ax.plot(t, C_0, 'm-.', label='C, sp. 0')
        ax.plot(t, C_1, 'm', label='C, sp. 1')

        ax.plot(t, I_0, ':r',label='I, sp. 0')
        ax.plot(t, I_1, 'r--',label='I, sp. 1')

        ax.plot(t, either_ci_0, 'g-', label = 'C + I, sp. 0 ')
        ax.plot(t, either_ci_1, 'g--', label = 'C + I, sp. 1 ')

        ax.plot(t, total_infected, 'r-.', label = 'C + I')
        #ax.plot(t, R, 'black',  label='Removed')
        ax.set_ylabel('Hosts')
        
        ax.scatter(500, N/2, label="half hosts infected @ 500d", s = .4)
        if helper_mode:
            return ax
        ax.legend()
        plt.title("Analytical SCIR Solution, Two Species: No Control Implemented")
        plt.show()
        return

class stoch_onesps: 
    '''
    plot an ensemble of sample paths, single species
    '''
    #def make_paths():
    def plot_paths (initial_cond, kernel_function, scale_parameter ,
               host_positions, species_parameters, control_parameters, debug, add_deterministic, deterministic_parms):
        '''
        Function needs additional work
        '''
        if add_deterministic:
            [t_end, beta, sigma, gamma] = deterministic_parms
            ax = single_species.plot_singlesps(initial_cond, t_end, beta, sigma, gamma, True)

        
        sim = gillespie(initial_cond, kernel_function, scale_parameter ,
            host_positions, species_parameters, control_parameters, False, False, False, False, None)
        fdict = Counter(sim)
        s = fdict.get(STATUS_S)
        c = fdict.get(STATUS_C)
        i = fdict.get(STATUS_I)
        r = fdict.get(STATUS_R)
        ax.set_ylabel('Hosts')
    
        ax.scatter(500,.5, label="1/2 Hosts Infected, t=500d", s = 1)
        ax.legend()
        plt.title("Stochastic SCIR Epidemic")
        plt.grid()
        plt.show()
        return plt
    
class stoch_twosps:
    def paths_twosps(initial_cond, nhosts_A, nhosts_B, sps_parms, kernel_function, scale_parameter, add_deterministic, 
                     normalisation_plot,
                     control_applied, control_start, control_parameters, landscape_model, scaler,nlandscapes, n_paths):
        '''
        for two species:
        Plots a user-specified number of stochastic sample paths

        currently uses CSR but shortly to be extended
        '''
        assert nhosts_A + nhosts_B == int(np.sum(initial_cond))
        i = 0
        sns.set_theme()
        fig, ax = plt.subplots()
        det_initcond = [*np.divide(initial_cond,2),*np.divide(initial_cond,2)]
        if add_deterministic: # we superimpose the deterministic solution - needs to be added
            ax = two_species.plot_multisps(det_initcond, t_end=1400 ,sps_parms=sps_parms, helper_mode=True )
        N = int(np.sum(det_initcond))
        if landscape_model == 'csr':
            lscapes = landscape_generators.CSR.gen_csr(nhosts_A=nhosts_A, nhosts_B=nhosts_B,num_species=2,scaler=scaler, nhosts_tot=nhosts_A+nhosts_B
                                                    , num_landscapes=nlandscapes)
        elif landscape_model == 'ns':
            lscapes = landscape_generators.neymanscott.get_ns(num_species=2, scaler=scaler, num_landscapes=nlandscapes, frac_B = nhosts_B/N, nhosts_tot=N, 
                                                             )

        elif landscape_model == 'citrus': # the citrus landscape used in (Hyatt-Twynham et al., 2017) - useful for validation etc.
            lscapes = ["smallLandscape.txt"]

        else:
            print(f'please specify one of "csr" (complete spatial randomness), "ns" (neyman scott, cauchy kernel) or "citrus" rather than {landscape_model}')

        for lscape in lscapes:
            [epi_array, host_landscape] = setup_system(initial_condition=initial_cond, landscape=lscape) # we only need to calculate the kernel once
    
            reused_kernel = make_kernel(system_state=epi_array,
            landscape=host_landscape, kernel_function=kernel_function, scale_parameter=scale_parameter, 
            species_parameters=sps_parms)

            epis = gillespie_prll(initial_cond, lscape, kernel_function, scale_parameter 
               , sps_parms, control_parameters, num_simulations=n_paths, 
               control_on=control_applied, control_start=control_start, return_audpc=False,return_simple_audpc = False,return_fsize=False, debug=False, write_files=False, reused_kernel=reused_kernel, return_t50= True  )

            print(f'final sizes are \n{compute_fsize(epis)}')
            for epi in epis:    
                [[S1_path, C1_path, I1_path, R1_path],
                [S2_path, C2_path, I2_path, R2_path] , 
                times] = scir_plotting.make_sps_path(epi)
                    
                C1_path = [0 if v is None else v for v in C1_path] # need to replace None with 0 as otherwise calcs won't be possible
                C2_path = [0 if v is None else v for v in C2_path]
                I1_path = [0 if v is None else v for v in I1_path]
                I2_path = [0 if v is None else v for v in I2_path]
                
                tot_1 = np.add(C1_path, I1_path)
                tot_2 = np.add(C2_path, I2_path)
                tot = np.add(tot_1, tot_2)
                if not normalisation_plot:
                    ax.plot(times, I1_path, 'green', label='Species 1, I(t)' if i ==0 else "", alpha=.4) # can re-add alpha here as desired
                    ax.plot(times, C1_path, 'magenta', label = 'Species 1, C(t)' if i ==0 else "", alpha=.4)

                    ax.plot(times, tot_1, 'cyan', label='Species 1, total infecteds' if i ==0 else "",  alpha=0.4)
                    ax.plot(times, tot_2, 'yellow', label='Species 2, total infecteds' if i ==0 else "",  alpha=0.4)
                    ax.plot(times, I2_path, 'red', label ='Species 2, I(t)' if i ==0 else "", alpha=.4)
                    ax.plot(times, C2_path, 'orange', label='Species 2, C(t)' if i ==0 else "", alpha=.4)
                
                if normalisation_plot:
                    ax.plot(times, tot, 'black', label='Total C+I'  if i ==0 else "",alpha=0.4 )
                i += 1 # to ensure that we don't overlabel the graph
                # ax.plot(times, )
        #print(f'I1 path = {I1_path}')
        #print(f'I2 path = {I2_path}')
        sns.set_theme()
        if add_deterministic:
            plt.title("Nonspatial SCIR: Comparison of Deterministic and Stochastic Models")
        else:
            plt.title(f"Stochastic Epidemic {n_paths} Sample Paths\nSquare Landscape, Side Lenth {scaler} m\nSpecies Parameters {sps_parms}\nKernel {kernel_function} scale_parm {scale_parameter}")
        ax.scatter(x=500, y=(nhosts_A+nhosts_B)/2, label="1/2 hosts infected, 500 d.")
        plt.xlabel("Time /days")
        plt.ylabel("Number of hosts")
        plt.legend()
        plt.savefig("stoch_det.svg")
        plt.show()
        return plt

class histograms:
    '''
    # Class providing histograms to verify distribution of final sizes
    '''

    def make_gdsahist(num_simulations, kernel_func,kernel_parm, sps_parms, ctrl_parms, initial_condition):
        '''
        Make a histogram of the final epidemic sizes with fixed parameters.
        can be used for comparison to Hyatt-Twynham 2017.
        '''
        cols = ["HOST_ID", "SPS_ID", "INFEC_STATUS", "T_RATE", "T_SC", "T_CI", "T_IR"]
        citrus_pos = pd.read_csv("./smallLandscape.txt", header=None)
        [epi_array, host_landscape] = setup_system(initial_condition=initial_condition, landscape="smallLandscape.txt") # we only need to calculate the kernel once
        reused_kern = make_kernel(system_state=epi_array,
        landscape=host_landscape, kernel_function=kernel_func, scale_parameter=kernel_parm, species_parameters=sps_parms)
                
        with mp.Pool() as pool:
            sim_paths = pool.starmap(gillespie, [(initial_condition, "smallLandscape.txt", kernel_func, kernel_parm,
            sps_parms, ctrl_parms, True, True, False, False, reused_kern)for i in range(num_simulations)])
        
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
        plt.title(f"Final Size Distribution, {num_simulations} replicates\nConstant radius of removal")
        plt.show()
        return


class utility:

    """
    Contains useful functions
        - incl. finding normalisation for a given parameter set
    """

    def plot_normalisation(initial_condition, kernel, sps_parms, Nhosts, ctrl_params, landscapes, scale_parameter, nreplicates,
                           eta_array):
        '''
        #Find transmission parameters such that - in absence of control - about half of hosts are infected by 500 days.

        # Eta is the adjustable normalising factor
        '''
        error_list = []

        for eta in eta_array:
            sps_parms_new = sps_parms*eta
            fs = []
            for lscape in landscapes:
                
                epis  = gillespie_prll(initial_condition, lscape, kernel, scale_parameter, sps_parms, ctrl_params, False, 'random',
                                        False, False, False, None, nreplicates) # by default takes the mean of 
            # now need to compute number infected at 500d. 
                for epi in epis:
                    size = compute_fsize(scir_plotting.reconstruct_array(epi, 500))
                    fs.append(size)

            median_fs = np.median(fs)
            error = median_fs - Nhosts/2
            error_list.append(error)

        plt.scatter(x=eta_array, y=error_list)
        plt.show()
        return 
                    
    '''`
'''
if __name__ == "__main__":
    initial_condition = [1100,10,0,0]
    nsample_paths = 12
    scale_parameter = 119
    no_control = [90, 0, 0, 0, 0, 31, 31]

    # [survey_freq, frac_A, frac_B, prob_detec_A, prob_detec_B, radius_A, radius_B] = control_parameters
    #a = 0.00835
    sps_parameters = np.array([[0.0025  ,  0.09      , (1/350) ,0.00053   ], [0.0025  ,   0.09     ,  (1/500),0.00053   ]])
    #sps_parameters = np.array([[0.0082, 0.09, (1/350), 0.00053], [0.0082, 0.14, (1/350), 0.00053]]) 

    #sps_parameters = np.array([[0.007  ,  0.09      , (1/350) ,0.00053   ], [0.014   ,   0.14     ,  (1/350),0.00053   ]])
    #sps_parameters = np.array([[0.007 , 0.09 , (1/350) ,0.00053 ], [0.014 , 0.14 , (1/500),0.00053 ]])

    stoch_twosps.paths_twosps(initial_condition,int(np.sum(initial_condition)),int(0),sps_parameters, exponential_kernel,scale_parameter, 
                              add_deterministic=False, normalisation_plot= True,
                               control_applied=False, control_start='random',
                              control_parameters=no_control, 
                              landscape_model='ns', scaler=4000, nlandscapes=5, n_paths=nsample_paths)
    
'''

    # stoch_twosps.paths_twosps(initial_condition,int(1110/2),int(1110/2),sps_parms, exponential_kernel,scale_parameter, 
    #                          add_deterministic=True, normalisation_plot=True,
    #                           control_applied=False, control_start='random',
    ##                          control_parameters=no_control, 
    #                          landscape_model='csr', scaler=4000, nlandscapes=1, n_paths=nsample_paths)

    #stoch_twosps.paths_twosps(initial_condition,600,600,sps_parms, cauchy_thick,scale_parameter, 
    #                         add_deterministic=False, normalisation_plot= True,
    #                           control_applied=False, control_start='random',
    #                          control_parameters=no_control, 
    #                          landscape_model='csr', scaler=4000, nlandscapes=1, n_paths=nsample_paths)
    


'''                              

'''
if __name__ == "__main__": # to produce fig 1
    initial_cond_twosps = [495, 5, 0, 0, 495, 5, 0, 5]
    initial_cond = [1101,10,0,0]
    fitted_beta = 0.00000845
    t_end = 1400
    sigma = 1/400
    gamma = 0
    helper_mode = False
    #comparisons.stoch_det_singlesps(initial_cond, 1400, fitted_beta, (1/107), 0)
    #single_species.plot_singlesps(initial_cond, 1400, beta, (1/107), 0, 0
    dummy_kernel = kernels.flat_kernel
    deterministic_parms = [1400, fitted_beta, (1/107), 0]
    sps_params_fitted = np.array([[math.sqrt(fitted_beta), math.sqrt(fitted_beta), (1/107), 0],[0, 0, (0) , 0]])
    no_control = [90, 0, 0, 0, 31, 31] # 0 probability detection so no control implemented here for the comparison withg
    #single_species.plot_singlesps(initial_cond, t_end, fitted_beta, sigma, gamma, helper_mode)
    #stoch_onesps.plot_paths(initial_cond, dummy_kernel, 0, "smallLandscape.txt", sps_params_fitted, no_control, False, True, deterministic_parms )

'''
