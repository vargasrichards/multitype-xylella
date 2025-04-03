'''
Script for verifying correctness of the dispersal kernel and asymmetric transmission in a subset of cases.

 - cauchy_thick
 - depends on 'smallLandscape.txt' file to read in host positions from

A. Vargas Richards, Dec. 2024

'''

import gillespie_dsa, pandas as pd
from gillespie_dsa import setup_system, cauchy_thick, exponential_kernel
from scipy.spatial.distance import pdist, squareform
import numpy as np

citrus_pos = pd.read_csv("./smallLandscape.txt", header=None)

def test_kernel(host_positions, kernel_function, sps_parms, scale_parameter):

    '''
    Function for use in model validation. Verifies that the disperal kernel is computed correctly 
    '''
    [HOST_ID, SPS_ID, INFEC_STATUS, T_RATE, T_SC, T_CI, T_IR] = [0, 1, 2, 3, 4, 5, 6]
    [[nu0, phi0, sigma0, gamma0],[nu1, phi1, sigma1, gamma1]] = sps_parms
    print(f'nu0: {nu0}; nu1: {nu1}\nnphi0: {phi0}; phi1: {phi1}\nsigma0: {sigma0}; sigma1: {sigma1}\n')
    beta0 = nu0 * phi0
    beta1 = nu1 * phi1
    #assert beta0 == beta1, "this test case is not currently implememtned"
    system_state = gillespie_dsa.setup_system([1101,10,0,0])
    symkern = np.multiply(beta0, kernel_function(squareform(pdist(host_positions)), scale_parameter)) # the symmetric kernel, multiplied by the transmission constant

    assymkern = gillespie_dsa.make_kernel(system_state, host_positions, kernel_function, scale_parameter, species_parameters=sps_parms) # the kernel with 

    assert symkern.all() == assymkern.all(), "Kernels did not match"
    print(f'Passed kernel check for parameter set')
    return 

if __name__ == "__main__": # runs a short list of tests
    for i in range(100): # for a range of scale parameters...
        test_kernel(citrus_pos, cauchy_thick, gillespie_dsa.sps_params, i)  #he case of the cauchy thick-tailed disperal kernel

    for i in range(100):
        test_kernel(citrus_pos, exponential_kernel, gillespie_dsa.sps_params, i) # the thin-tailed exponential kernel


