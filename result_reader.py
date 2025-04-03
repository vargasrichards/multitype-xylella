'''
Tools for reading in a txt / csv report of parameter sweep in two dimensions and producing a 2D colourmap
as well as subsequently producing 1D plot of contrasting strategies (misspecified vs landscape adapted control)

A. Vargas Richards, Feb. 2025

'''

import pandas as pd, numpy as np, matplotlib.pyplot as plt
import csv
from collections import Counter
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_results(dataframe, response_var, response_var2, var_x, var_y, xlabel, ylabel, cmap, cbar_label1, cbar_label2, suptitle, filename, fsiz):
    '''
    Slightly generalized plotting function which plots a 2D parameter scan
    note that the axis tick labels are manually set, please take care...
    '''
    FONTSIZE = 15
    ANNOTATE_SIZE = 25

    fig, axes = plt.subplots(2, sharex = True, sharey = False) # can share the x axis to help illustrate the point about conflicting results...   
    plt.tight_layout()

    var_y = pd.to_numeric(dataframe[var_y]).to_numpy(); var_x = pd.to_numeric(dataframe[var_x]).to_numpy()

    unique_y = np.unique(var_y); unique_x = np.unique(var_x)
    plt.rcParams.update({'font.size': FONTSIZE})
    
    rspns = pd.to_numeric(dataframe[response_var]).to_numpy()
    Z = np.array(rspns)
    Z.shape = (len(unique_y), len(unique_x))
    row_opt = np.min(Z, axis=1, keepdims=True)
    optmat = (Z == row_opt).astype(int)

    imag = axes[0].imshow(Z, aspect=2.0, cmap=cmap, origin='lower')
    opt_indices = np.argwhere(optmat == 1)
    axes[0].scatter(opt_indices[:, 1], opt_indices[:, 0], s=100, edgecolors='green', 
               facecolors='none', 
               linewidths=2, marker="s", label="Optimal Radius of Control")
    
    cbar = plt.colorbar(imag)
    cbar.set_label(cbar_label1)

    axes[0].set_yticklabels(['0', '0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    axes[0].set_xticklabels(['0', '0', '250', '500', '750', '1000'])

    axes[0].tick_params(axis='both', labelsize=FONTSIZE)
    axes[0].text(1.2, 1, '(A)', transform=axes[0].transAxes, fontsize=ANNOTATE_SIZE, fontweight='bold', va='top', ha='left')
    rspns = pd.to_numeric(dataframe[response_var2]).to_numpy()
    rspns /= 10e2 # we divide the second response variable by 1000
    Z = np.array(rspns)
    Z.shape = (len(unique_y), len(unique_x))
    row_opt = np.min(Z, axis=1, keepdims=True)
    optmat = (Z == row_opt).astype(int)

    imag = axes[1].imshow(Z, aspect=2.0, cmap=cmap, origin='lower')
    opt_indices = np.argwhere(optmat == 1)
    axes[1].scatter(opt_indices[:, 1], opt_indices[:, 0], s=100, edgecolors='green', 
               facecolors='none', linewidths=2, marker="s")
    cbar = plt.colorbar(imag)
    cbar.set_label(cbar_label2)

    axes[1].set_yticklabels(['0', '0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    axes[1].set_xticklabels(['0', '0', '250', '500', '750', '1000'])
    
    axes[1].tick_params(axis='both', labelsize=FONTSIZE)
    axes[1].text(1.2, .5, '(B)', transform=axes[1].transAxes, fontsize=ANNOTATE_SIZE, fontweight='bold', va='top', ha='left')

    plt.tick_params(axis='both', labelsize=FONTSIZE)

    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)

    fig.legend(frameon=False)
    
    plt.suptitle(suptitle, fontsize=12)
    plt.savefig("experimenta_" + str(filename) + '.svg')
    plt.show()
    return

def plot_typed(dataframe, sps_parameters,response_var, response_var2, var_x, var_y, normalisation, xlabel, ylabel, cmap, cbar_label1, cbar_label2, suptitle, filename, fsiz):
    '''
    largely following the plot_results function, plots the Ke median for each of the 

    Plots are single-normalised by default (i.e., Single-normalised(KE(TYPE A)) = Ke(type A)/num_A for each landscape)

    Double-normalisation can be specified by setting normalisation = 'double':
    then double-normalised(Ke(type A)) = Ke(type A)/(num_A* nu_A) 

    which attempts to compensate for the infectivity of the type A host. 
    '''


    if normalisation == 'single': # this is the case where we divide by the number
        dataframe["A_Med_KE_single"] = dataframe["Med_KE"]/dataframe["n_A"]
        dataframe["B_Med_KE_single"] = dataframe["Med_KE"]/dataframe["n_B"]

        dataframe["A_Med_AUDPC_single"] = dataframe["Med_AUDPC_SPS"]/dataframe["n_A"]
        dataframe["B_Med_AUDPC_single"] = dataframe["Med_AUDPC_SPS"]/dataframe["n_B"]    

    elif normalisation == 'double': # we divide by the type-specific infectivity as well as

        # in this case we need to parse the species parameters 
        dataframe["A_Med_KE_double"] = dataframe["Med_KE"]/(dataframe["n_A"]*nu_A)
        dataframe["B_Med_KE_double"] = dataframe["Med_KE"]/(dataframe["n_B"]*nu_B)

        dataframe["A_Med_AUDPC_double"] = dataframe["Med_AUDPC_SPS"]/(dataframe["n_A"]*nu_A)
        dataframe["B_Med_AUDPC_double"] = dataframe["Med_AUDPC_SPS"]/(dataframe["n_B"]*nu_B)

    elif normalisation == None:
        pass
    else:
        print(f'Please specify either "single" or "double" or None for normalisation rather than {normalisation}')

    fig, axes = plt.subplots(2, sharex = True, sharey=True) # can share the x axis to help illustrate the point about conflicting results...   
    
    plt.tight_layout()
    var_y = pd.to_numeric(dataframe[var_y]).to_numpy()
    var_x = pd.to_numeric(dataframe[var_x]).to_numpy()

    unique_y = np.unique(var_y)
    unique_x = np.unique(var_x)
    print(f'unique_x is {unique_x}')
    plt.rcParams.update({'font.size': fsiz})
    rspns = pd.to_numeric(dataframe[response_var]).to_numpy()
    Z = np.array(rspns)
    Z.shape = (len(unique_y), len(unique_x))
    row_opt = np.min(Z, axis=1, keepdims=True)
    optmat = (Z == row_opt).astype(int)
    # Z = np.divide(Z, 1110)
    #fig, ax = plt.subplots(figsize=(6,6))
    imag = axes[1].imshow(Z, aspect=2.0, cmap=cmap, origin='lower')
    opt_indices = np.argwhere(optmat == 1)
    axes[1].scatter(opt_indices[:, 1], opt_indices[:, 0], s=200, edgecolors='green', 
               facecolors='none', linewidths=2, marker="s", label="Optimal Radius of Control")
    cbar = plt.colorbar(imag)
    cbar.set_label(cbar_label1)
    #plt.title(figtitle)

    #axes[0].set_yticklabels(['0', '0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    #axes[0].set_xticklabels(['0', '0', '250', '500', '750', '1000'])
    #plt.savefig("experimenta_" + str(filename) + '.svg')

    #var_x = pd.to_numeric(dataframe[var_x]).to_numpy() # we should first ensure that the df numbers are being read as numbers
    #var_y = pd.to_numeric(dataframe[var_y2]).to_numpy()
    #unique_y = np.unique(var_y2)
    #unique_x = np.unique(var_x)
    #plt.rcParams.update({'font.size': fsiz})
    rspns = pd.to_numeric(dataframe[response_var2]).to_numpy()
    Z = np.array(rspns)
    Z.shape = (len(unique_y), len(unique_x))
    row_opt = np.min(Z, axis=1, keepdims=True)
    optmat = (Z == row_opt).astype(int)
    # Z = np.divide(Z, 1110)
    #fig, ax = plt.subplots(figsize=(6,6))
    imag = axes[0].imshow(Z, aspect=2.0, cmap=cmap, origin='lower')
    opt_indices = np.argwhere(optmat == 1)
    axes[0].scatter(opt_indices[:, 1], opt_indices[:, 0], s=200, edgecolors='green', 
               facecolors='none', linewidths=2, marker="s", label="Optimal Radius of Control")
    cbar = plt.colorbar(imag)
    cbar.set_label(cbar_label2)
    #plt.title(figtitle)

    #axes[1].set_yticklabels(['0', '0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    #axes[1].set_xticklabels(['0', '0', '250', '500', '750', '1000'])

    #plt.xlabel(xlabel)
    #plt.ylabel(ylabel)

    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)

    #plt.legend()
    lgd = plt.legend(bbox_to_anchor=(2.01, 1), loc='upper left')
    #plt.legend()
    plt.suptitle(suptitle, fontsize=12)
    plt.savefig("experimenta_" + str(filename) + '.svg')

    plt.show()
    return

def coplot_controls(dataframe, response_vars):
    '''
    ## Plots different control strategies on the same graph

    ### Arguments:
    ----------

    `dataframe` - contains the necessary epidemic information to reconstruct the epi

    `response_vars` - array length two:strings indicating the metric of control (eg 'Med_KE') which is the primary outcome metric

    `titlestr` - gives the title for the plot     

    `ystr` - the y axis label (match to the response variable 

    `reoptimise` - whether to reoptimise the control strategy for the different response variables or to use the same 'misspecified radius'
    for both misspecified control regimes: the best-performing radius for response variable 1 when there is no species B.
    '''

    FONTSIZE = 15
    ANNOTATE_SIZE = 25

    plt.rcParams.update({'font.size': FONTSIZE})
    fig, axes = plt.subplots(2)
    plt.tight_layout()

    axes[0].tick_params(axis='both', labelsize=FONTSIZE)
    axes[1].tick_params(axis='both', labelsize=FONTSIZE)


    dataframe = dataframe.astype(float) # ensuring that everything is interpreted correctly as floats...
    [response_var1, response_var2] = response_vars
    print(f'response_var1 supplied as {response_var1}\nresponse_var2 supplied as {response_var2}') # maybe we don't need this??

    r1_q1str = str('Q1_') + str(response_var1) # constructng the strings used to index dataframes
    r1_q3str = str('Q3_') + str(response_var1)

    r2_q1str = str('Q1_') + str(response_var2)
    r2_q3str = str('Q3_') + str(response_var2)

    response_var1 = 'Med_' + response_var1 # the medians...
    response_var2 = 'Med_' + response_var2

    subset = dataframe[['Radius', 'Frac_B', r1_q1str, response_var1, r1_q3str, r2_q1str, response_var2, r2_q3str ]] # we take the useful  elements of the data via subsetting
     
    subset[response_var2] /= 1000 # hence, AUDPC MUST be the second response variable to avoid labelling errors
    subset[r2_q1str] /= 1000
    subset[r2_q3str] /= 1000

    initial_point = dataframe[(dataframe['Frac_B'] == '0') | (dataframe['Frac_B'] == 0) | (dataframe['Frac_B'] == '0.0')] # where there is no species B
    min_ke = float(np.min(pd.to_numeric(initial_point[response_var1])))

    # which implies that the optimal radius for the landscape where there is no species B is probably:
    initial_point[response_var1] = pd.to_numeric(initial_point[response_var1])
    misspecified_radius = int((initial_point[initial_point[response_var1] == min_ke]).Radius)
    print(f'misspecified radius computed as {int(misspecified_radius)} m')

    misspecified_res = subset[subset.Radius == misspecified_radius]
    misspecified_res = misspecified_res.astype(float)
    print(f'misspecified results are \n{misspecified_res}')

    optimal_radii = [] # this will store the landscape-adapted control radii 
    opt_responses = []
    opt_resp_q1 = []
    opt_resp_q3 = []

    fraB = []

    for fraction_B in np.unique(dataframe['Frac_B']):
        fraB.append(fraction_B)
        epi_results = subset[subset.Frac_B == fraction_B]
        min_response = float(np.min(pd.to_numeric(epi_results[response_var1])))
        try:
            q1mib = float((epi_results[epi_results[response_var1] == min_response])[r1_q1str])
        except:
            q1mib = np.mean((epi_results[epi_results[response_var1] == min_response])[r1_q1str])

        try:
            q3mib = float((epi_results[epi_results[response_var1] == min_response])[r1_q3str])
        except:
            q3mib = np.mean((epi_results[epi_results[response_var1] == min_response])[r1_q3str])
            
        try:
            opt_rad = float((epi_results[epi_results[response_var1] == min_response]).Radius)
        except:
            opt_rad = np.mean(epi_results[epi_results[response_var1] == min_response].Radius)

        optimal_radii.append(opt_rad)
        opt_responses.append(min_response)
        opt_resp_q1.append(q1mib)
        opt_resp_q3.append(q3mib)#
        
    lower_misspecified = np.abs(np.subtract(misspecified_res[r1_q1str] , misspecified_res[response_var1]))
    upper_misspecified = misspecified_res[r1_q3str] - misspecified_res[response_var1]
    
    lower_optimal = np.abs(np.subtract(opt_resp_q1 , opt_responses))
    upper_optimal = np.subtract(opt_resp_q3 , opt_responses)
    delta = 0.0025 # an offset so that both response variables can be coplotted without ovetrlapping
    fraB_plusd = list(list(np.asarray(fraB) + delta))
    fraB_minusd=  list(list(np.asarray(fraB) - delta))
    print(f'supplied yerrr array as \n{np.array([lower_misspecified,upper_misspecified])}')
    axes[0].errorbar(x=fraB_plusd, y=misspecified_res[response_var1], yerr=np.array([lower_misspecified,upper_misspecified]), capsize=6, dash_capstyle='butt', 
    marker = 'x',color = 'tab:brown', label = 'Misspecified Control: 25-50-75 percentile')
    axes[0].errorbar(x=fraB_minusd, y=opt_responses, yerr=np.array([lower_optimal,upper_optimal]), capsize = 6, dash_capstyle = 'butt',
    marker= 'o',label = 'Optimal Control: 25-50-75 percentile')
    # this is for the first response variable ie KE, 
    # for the second response variable then the same radii should be carried over unless explicit reoptimisation is desired in which case the code should be modified.

    optimal_radii = [] # this will store the landscape-adapted control radii 
    opt_responses = []
    opt_resp_q1 = []
    opt_resp_q3 = []

    fraB = []
    print(f'overall response variable 2 values are { dataframe[response_var2]}')
    print(f'min response variable 2 values are {np.min(dataframe[response_var2])}')

    for fraction_B in np.unique(dataframe['Frac_B']):
        fraB.append(fraction_B)
        epi_results = subset[subset.Frac_B == fraction_B]
        min_response = float(np.min(pd.to_numeric(epi_results[response_var2])))
        try:
            q1mib = float((epi_results[epi_results[response_var2] == min_response])[r2_q1str])
        except:
            q1mib = np.mean((epi_results[epi_results[response_var2] == min_response])[r2_q1str])
        try:
            q3mib = float((epi_results[epi_results[response_var2] == min_response])[r2_q3str])
        except:
            q3mib = np.mean((epi_results[epi_results[response_var1] == min_response])[r2_q3str])
        try:
            opt_rad = float((epi_results[epi_results[response_var2] == min_response]).Radius)
        except:
            opt_rad = np.mean(epi_results[epi_results[response_var2] == min_response].Radius)

        optimal_radii.append(opt_rad)
        opt_responses.append(min_response)
        opt_resp_q1.append(q1mib)
        opt_resp_q3.append(q3mib)#
    misspecified_res = subset[subset.Radius == misspecified_radius]
    print(f'misspecified res = \n{misspecified_res}')

    print(f'misspecified_res[response_var2] = {misspecified_res[response_var2]}')
    print(f'optimal response response variable 2: {opt_responses}')
    
    lower_misspecified = np.abs(np.subtract(misspecified_res[r2_q1str] , misspecified_res[response_var2])) # cannot just plot the value of the errors themselves...
    upper_misspecified = misspecified_res[r2_q3str] - misspecified_res[response_var2]
    
    lower_optimal = np.abs(np.subtract(opt_resp_q1 , opt_responses))
    upper_optimal = np.subtract(opt_resp_q3 , opt_responses)
    fraB_plusd = list(list(np.asarray(fraB) + delta))
    fraB_minusd=  list(list(np.asarray(fraB) - delta))

    axes[1].errorbar(x=fraB_plusd, y=misspecified_res[response_var2], yerr=np.array([lower_misspecified,upper_misspecified]), capsize=6, dash_capstyle='butt', 
    marker = 'x',color = 'tab:brown', label = '_Misspecified Control: Q25-Q50-Q75')
    axes[1].errorbar(x=fraB_minusd, y=opt_responses, yerr=np.array([lower_optimal,upper_optimal]), capsize = 6, dash_capstyle = 'butt',
    marker= 'o',label = '_Optimal Control: Q1-Q3')

    axes[0].set_ylabel(r"Epidemic impact $K_{E}$")
    axes[1].set_ylabel(r"AUDPC (x $10^{3}$)")

    fig.legend(frameon=False)
    axes[1].spines[['right', 'top']].set_visible(False)
    axes[0].spines[['right', 'top']].set_visible(False)
    plt.suptitle(suptitle)
    plt.xlabel("Proportion of Host Type B")
    axes[0].text(1, .5, '(C)', transform=axes[0].transAxes, fontsize=ANNOTATE_SIZE, fontweight='bold', va='top', ha='left')
    axes[1].text(1, .5, '(D)', transform=axes[1].transAxes, fontsize=ANNOTATE_SIZE, fontweight='bold', va='top', ha='left')
    plt.savefig("compare_strategies.svg")
    plt.show()
    return

def plot_sensitivities(filename):
    '''
    An interface to two different functions for plotting the output of a sensitivity analysis.
    '''


    return

def read_file(filename, delimiter):
    '''
    Reads a file into a dataframe 
    '''
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=delimiter)
        datadf = pd.DataFrame(reader)
        print(f'datadf read in as \n{datadf}')
    return datadf

if __name__ == "__main__":
    cauchy_rv2 = 'AUDPC_SPS'
    cauchy_rv1 = 'KE'
    exp_df = read_file('./FINALRESULTS/DELTARESULTS/4000/119.txt', delimiter=',')
    sigma_dataframe = read_file('./SIGMARESULTS/4000/119.txt', delimiter = '\t')
    cauchy_df = read_file('./FINALRESULTS/CAUCHYRESULTS/4000/84.5.txt', delimiter=',')
    #ns_df = read_file('./FINALRESULTS/CLUST/IDENTICAL_TYPES/119_ns_landscape.txt', delimiter= '\t')
    #ns_df = read_file('./FINALRESULTS/CLUST/DELTASIGMADELTAPD/119_ns_landscape.txt', delimiter= '\t')
    asym_df_pd4 = read_file('./FINALRESULTS/ASYM/4000/EXP_DELTAPD/119.txt', delimiter = ',')
    asym_df_sigma4 = read_file('./FINALRESULTS/ASYM/4000/EXP_DELTASIGMA/119.txt', delimiter = ',')

    #asym3_df = read_file('./FINALRESULTS/ASYM/3000/sigma/119.txt', delimiter = '\t')
    sym3_df = read_file('./FINALRESULTS/3K/119.txt', delimiter='\t')
    var_x = 'Radius'
    var_y = 'Frac_B'
    xlabel= 'Radius of removal /m'
    ylabel = 'Proportion of Species B'
    cbar_label1= r'Median Epidemic impact $K_{E}$'
    cbar_label2 = r'Median AUDPC (x $10^{3}$)'
    filename = 'test.svg'
    cmap = 'magma_r'
    var_x_two = 'Radius_A'
    var_y_two = 'Radius_B'
    response_var1='Med_KE'
    response_var2='Med_AUDPC_SPS'
    var_x = 'Radius'
    var_y = 'Frac_B'
    xlabel= 'Radius of removal /m'
    ylabel = 'Proportion of Host Type B'
    response_vars = [response_var1, response_var2]
    suptitle = ''

    plot_results(asym_df_pd4, response_var1, response_var2, var_x, var_y , xlabel, ylabel, cmap,cbar_label1, cbar_label2, suptitle, filename, fsiz =10)
    response_vars=['KE','AUDPC_SPS']
    coplot_controls(dataframe=asym_df_pd4, response_vars=response_vars)




