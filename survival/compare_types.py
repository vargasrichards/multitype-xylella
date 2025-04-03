'''
Provides methods for the computation / plotting of type-normaised metrics from dataframe written out by a sim. 

A. Vargas Richards, feb 2025

'''

import pandas as pd
import numpy as np
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt

def normalised_dataframe (dirpath, infile, outfile, species_parameters):
    '''
    ## Computes type-normalised values of the different epidemic outcome metrics 

    ### Arguments
    ----------------

    `dirpath` : the relative path of the directory where you want

    '''

    in_path = dirpath + '/' +  infile
    out_path = dirpath + '/' + outfile

    datafra = pd.read_csv(in_path, delimiter = '\t')
    print(f'original frame is \n {datafra}')
    [nu_A, nu_B] = species_parameters[:, 0] 
    # first we can compute all the KE-related metrics
    # in some ways these are easier to handle because the upper bound is obvious (=N)
    datafra['Med_KE_A_singlenorm'] = datafra['Med_KE_A']/datafra['N_a']
    datafra['Med_KE_B_singlenorm'] = datafra['Med_KE_B']/datafra['N_b']

    datafra['Med_KE_A_doublenorm'] = datafra['Med_KE_A_singlenorm'] / nu_A
    datafra['Med_KE_B_doublenorm'] = datafra['Med_KE_B_singlenorm'] / nu_B

    datafra['Med_AUDPC_A_single'] = datafra['Med_AUDPC_A']/datafra['N_a']
    datafra['Med_AUDPC_B_single'] = datafra['Med_AUDPC_B']/datafra['N_b']

    datafra['Med_AUDPC_A_double'] = datafra['Med_AUDPC_A']/(datafra['N_a'] * nu_A )
    datafra['Med_AUDPC_B_double'] = datafra['Med_AUDPC_B']/(datafra['N_b'] * nu_B )

    datafra['KE_ratio_single'] = datafra['Med_KE_B_singlenorm'] / datafra['Med_KE_A_singlenorm'] # this KE ratio is quite important ...

    datafra['KE_ratio_unnorm'] = datafra['Med_KE_A']/datafra['Med_KE_B'] # 
    print(f'dataframe = \n{datafra}')

    datafra.to_csv(out_path, sep = '\t')
    return datafra # the dataframe now has many more cols now

def compare_plot(response_var, response_var2, filename, results_file):
    '''
    Plots a comparison of response variables between different host types.

    Commonly, for example, 

    '''
    cmap = 'magma_r'
    cbar_label1 = r'Median $\frac{K_{E,A}}{N_{A}}$'
    cbar_label2 = r'Median $\frac{K_{E,B}}{N_{B}}$'

    dataframe = pd.read_csv(filename, sep='\t')
    var_x = 'Radius'
    var_y = 'Frac_B'

    fig, axes = plt.subplots(2, sharex = True, sharey=True) # can share the x axis to help illustrate the point about conflicting results...   
    
    plt.tight_layout()
    var_y = pd.to_numeric(dataframe[var_y]).to_numpy()
    var_x = pd.to_numeric(dataframe[var_x]).to_numpy()
    unique_y = np.unique(var_y)
    unique_x = np.unique(var_x)

    rspns = pd.to_numeric(dataframe[response_var]).to_numpy()
    #rspns /= 10e2 # rescale AUDPC slightly
    Z = np.array(rspns)
    Z.shape = (len(unique_y), len(unique_x))
    row_opt = np.min(Z, axis=1, keepdims=True)
    optmat = (Z == row_opt).astype(int)

    imag = axes[1].imshow(Z, aspect=2.0, cmap=cmap, origin='lower')
    opt_indices = np.argwhere(optmat == 1)
    #axes[1].scatter(opt_indices[:, 1], opt_indices[:, 0], s=100, edgecolors='green', 
    #           facecolors='none', 
    #           linewidths=2, marker="s", label="Optimal Radius of Control")
    cbar = plt.colorbar(imag)
    cbar.set_label(cbar_label1)
    #plt.title(figtitle)

    axes[1].set_yticklabels(['0', '0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    axes[1].set_xticklabels(['0', '0', '250', '500', '750', '1000'])
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
    axes[0].scatter(opt_indices[:, 1], opt_indices[:, 0], s=100, edgecolors='green', 
               facecolors='none', linewidths=2, marker="s")
    cbar = plt.colorbar(imag)
    cbar.set_label(cbar_label2)
    #plt.title(figtitle)

    axes[1].set_yticklabels(['0', '0','0.2', '0.4', '0.6', '0.8', '1.0'])
    axes[1].set_xticklabels(['0', '0', '250', '500', '750', '1000'])


    fig.supxlabel('Radius of Removal /m')
    fig.supylabel('Fraction of Type B Hosts')

    #plt.legend()

    #fig.legend(bbox_to_anchor=(-10, 0),loc = 'lower right')
    #plt.subplots_adjust(left=0.07, right=0.93, wspace=0.25, hspace=0.35)
    #axes[1].legend()
    #axes[0].legend()
    plt.legend()
    

    plt.show()

    return


if __name__ == "__main__":
    filename = './FINALRESULTS/ASYM/4000/EXP_DELTAPD/119.txt'
    normalised_filename = './FINALRESULTS/ASYM/4000/EXP_DELTAPD/normalised_results.txt'
    species_parameters = np.array([[0.02   ,    0.09 ,      0.00285714, 0.00053   ] ,[0.02     ,  0.14     ,  0.00285714 ,0.00053   ]])
    normed_df = normalised_dataframe(dirpath='./FINALRESULTS/ASYM/4000/EXP_DELTAPD', outfile='normalised_results.txt', species_parameters=species_parameters,
                        infile= '119.txt'  )
    
    response_var = 'Med_KE_A_singlenorm'
    response_var2 = 'Med_KE_B_singlenorm'
    compare_plot(response_var, response_var2, normalised_filename,'comparison_plot.svg')
    
