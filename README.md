# Spatial, Stochastic SCIR Epidemics on Multi-type Landscapes with Crypticity

## This repository contains code and some data from NST Part II project 

#### A. Vargas Richards, Feb 2025

### Guide to directory structure:
------------------------------------


Key files are the following:
----------------------------

'gillespie_dsa.py' is the main file, which performs stochastic epidemic simulation.

'scir_plotting.py' provides plotting methods.

'sensitivity_analyses.py' provides an extensible framework for the sensitivity analysis.

'spatiotemp.py' allows for epidemic state reconstruction at arbitrary time given the epi_array structure which can be output from the main simulation script.

'result_reader.py' is the plotter which can be used to produce the complex multipanel figures shown in the report. Manual specification of the relative path to a file in the results directory 'FINALRESULTS' is required.



Files of minor importance:
-----------------------

'fit_leastsq.R' was used to fit the Cauchy dispersal kernel in R^2 to the exponential kernel via a simple least-squares method. 'kernel_normalisation.pdf'shows the fitted distributions against one another.

'survival.py' used for survival analysis (not included in final report)

'sensitivity_analyses.py' used for sensitivity analysis.



