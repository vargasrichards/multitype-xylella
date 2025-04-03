'''
# File containing methods for the generation of multi- /single species landscapes


## In this project we consider:
    - aspatial, two species
    - complete spatial randomness, two species (varying proportions)
    - strauss, two-species
    - neyman-scott, two species
    - 

A Vargas Richards, 2025
'''
import pandas as pd
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class neymanscott:
    '''
    does not generate landscapes but retrieves pre-simulated landscapes from the requisite directory
    
    '''

    def get_ns(num_species, scaler, frac_B, nhosts_tot, num_landscapes):
        '''
        Gets several landscapes from the requisite directory
        '''
        landscapes= []

        scale = int(scaler/1000)

        for landscape_num in range(num_landscapes):
            #rands = np.random.uniform(0, 1, (nhosts_tot*2)) # these random floats have the necessary 
            path_tolandscape = './ns_landscapes/' + str(int(frac_B*10)) +  '/rep' + str(int(landscape_num)) +  'len' + str(scale)+ '.tsv'

            old = pd.read_csv(path_tolandscape, sep='\t', header=None)
            l = old.set_axis(['pos_X', 'pos_Y','sps_id', 'host_id'], axis=1)
            #l = 'pos_X', 'pos_Y','sps_id', 'host_id'
            print(f'read in landscape as \n{l}')
            #a_labels =  [0] * nhosts_A
            #b_labels  = [1] * nhosts_B

            #sps_labels = a_labels + b_labels  # the species identity labels 
            #assert len(sps_labels) == len(x_vals)

            # provides the size of the square domain in metres
            #host_ids =  range(nhosts_tot)
            #l#andscape_data = {'host_id': host_ids, 'sps_id': sps_labels, 'pos_X':scaled_x, 'pos_Y':scaled_y }
            #landscape_df = pd.DataFrame.from_dict(landscape_data)
            landscapes.append(l)    

        return landscapes



class CSR: 
    '''
    Generates complete spatial randomness landscape of arbitrary density (CSR) for 2 species.
    '''

    def gen_csr(num_species, scaler, nhosts_A, nhosts_B, nhosts_tot, num_landscapes):

        '''
        Complete spatial randomness
        Generates  `num_landscapes` instances of the landscape.
        '''

        # define index constants

        assert nhosts_A + nhosts_B == nhosts_tot, f"Total {nhosts_tot} not equal to {nhosts_A} A + {nhosts_B} B, check your calcs."
        landscapes= []
        for _ in range(num_landscapes):
            rands = np.random.uniform(0, 1, (nhosts_tot*2)) # these random floats have the necessary 

            x_vals, y_vals = rands[:nhosts_tot], rands[nhosts_tot: nhosts_tot*2]
            assert len(x_vals) == len(y_vals)
            scaled_x, scaled_y = x_vals*scaler, y_vals*scaler 

            a_labels =  [0] * nhosts_A
            b_labels  = [1] * nhosts_B

            sps_labels = a_labels + b_labels  # the species identity labels 
            assert len(sps_labels) == len(x_vals)

            # provides the size of the square domain in metres
            host_ids =  range(nhosts_tot)
            landscape_data = {'host_id': host_ids, 'sps_id': sps_labels, 'pos_X':scaled_x, 'pos_Y':scaled_y }
            landscape_df = pd.DataFrame.from_dict(landscape_data)
            print(f'CSR landscape with {num_species} species on square length {scaler} metres generated successfully')
            landscapes.append(landscape_df)            
        
        return landscapes
    

class visualise:
    '''
    Provides landscape plotting methods
    '''
    def plot_landscape(landscape, figtitle):
        '''
        Plot a landscape using points 

        nb Saves the fig as the title.svg in current wd
        '''

        markers = {"0": "s", "1": "X"}
        plt.title(figtitle)
        plt.xlabel("X position /m")
        plt.ylabel("Y position, /m")
        sns.despine()
        sns.set_style("whitegrid")
        host_plot = sns.scatterplot(data=landscape, x="pos_X", y="pos_Y", hue="sps_id", 
                                    legend=True, style="sps_id", markers=['o', 's'])
        host_plot.legend(bbox_to_anchor=(1, 0.5))

        #host_plot.legend(title='Species Identity', labels=['Species A', 'Species B'])
        fstring = figtitle + '.svg'
        plt.savefig(fstring)
        plt.show()
        return


if __name__ == "__main__":
    csrl = CSR.gen_csr(2, 1000, 500,500,1000, num_landscapes=1) # make the host landscape
    visualise.plot_landscape(csrl, "Two Host Species, Complete Spatial Randomness, 1000m by 1000m")