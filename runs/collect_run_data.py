"""
Read data from find_lbit() runs, convert the data
to a more compressed pandas DataFrame format that 
can be more easily manipulated to generate plots, 
and save the compressed format to file.
"""


import os
import pickle
import optparse
import json
import numpy as np
import scipy.optimize as so
import pandas as pd

import qosy as qy

import bioms
from bioms.tools import get_size

# Parse the input arguments.
parser = optparse.OptionParser()
parser.add_option('-I', '--input', type='str', dest='input_filename', help='Input file specifying the data to collect and analyze.')

(options, args) = parser.parse_args()

# The input filename.
input_filename = options.input_filename

# Read the input file.
input_file = open(input_filename, 'r')
input_args = json.load(input_file)
input_file.close()

# The folders to read data from.
folders = input_args['folders']

# The output filename to save the collected data to.
output_filename = input_args['output_filename']

# The parameters to NOT save to the pandas
# DataFrame because they take up too much space
# and/or are not necessary for making plots.
params_to_ignore = ['explored_com_data', \
                    'explored_anticom_data', \
                    'explored_basis', \
                    'taus', \
                    'basis_inds', \
                    'basis_sizes', \
                    'com_norms', \
                    'binarities', \
                    'tau_norms', \
                    'fidelities', \
                    'initial_fidelities', \
                    'final_fidelities', \
                    'proj_final_fidelities', \
                    'num_taus_in_expansion', \
                    'ind_expansion_from_ind_tau']

# Compute the "weight" of an operator,
# the probability of each site being
# a non-identity operator.
def operator_weights(operator, num_orbitals):
    weights = np.zeros(num_orbitals) + 1e-16
    for (coeff, op_string) in operator:
        for (op_name, orb_label) in op_string:
            weights[orb_label] += np.abs(coeff)**2.0
    
    weights /= np.sum(weights)

    return weights

# Compute the "localities" of an operator O = \sum_a g_a S_a,
# the probability \sum_{a for k-local S_a} g_a^2/\sum_b g_b^2
# of an operator string S_a in the expansion being k-local.
def operator_localities(operator, num_orbitals):
    localities = np.zeros(num_orbitals) + 1e-16
    for (coeff, op_string) in operator:
        # The integer k for a k-local operator string,
        # i.e., the number of sites that the operator
        # string acts on.
        k              = len(op_string.orbital_operators)
        localities[k] += np.abs(coeff)**2.0
    localities /= np.sum(localities)

    return localities

# TODO: test!
# Performs an exponential fit
# for the weights of the 1D lbit.
def exp_fit(weights, lattice_type):
    # For a 1D chain.
    if lattice_type == '1D_chain':
        x  = np.arange(len(weights))
        y  = weights

        ind0 = np.argmax(weights)
        x0   = x[ind0]

        inds_nonzero = np.where(np.abs(weights) > 2e-16)[0]
        x = x[inds_nonzero]
        y = y[inds_nonzero]

        # The exponential function to fit.
        def func(x, a, b):
            return b * np.exp(-np.abs(x - x0) / a)

        # TODO: write jacobian
        #def jac_func(x, a, b):
        #    return np.array()

        popt, pcov = so.curve_fit(func, x, y)
    # For a 2D square lattice.
    elif lattice_type == '2D_square':
        N  = len(weights)
        L  = int(np.round(np.sqrt(N)))
        Lx = L
        Ly = L
        
        xs = []
        ys = []
        for y in range(Ly):
            for x in range(Lx):
                xs.append(x)
                ys.append(y)
        xs = np.array(xs)
        ys = np.array(ys)
        
        

        ind0 = np.argmax(weights)
        x0   = xs[ind0]
        y0   = ys[ind0]
        
        inds_nonzero = np.where(np.abs(weights) > 2e-16)[0]
        sites   = inds_nonzero
        weights = weights[inds_nonzero]
        xs      = xs[inds_nonzero]
        ys      = ys[inds_nonzero]

        # The exponential function to fit.
        def func(sites, a, b):
            xs = xs[sites]
            ys = ys[sites]
            result = np.zeros(len(sites))
            for ind_s in range(len(sites)):
                result[ind_s] = b * np.exp(-nla.norm(np.array([xs[ind_s] - x0, ys[ind_s] - y0]))/ a)
            
            return result

        # TODO: write jacobian
        #def jac_func(x, a, b):
        #    return np.array()
        
        popt, pcov = so.curve_fit(func, sites, weights)
    else:
        raise ValueError('Invalid lattice_type: {}'.format(lattice_type))
        
    return [popt, pcov]
    
# Keep a list of dicts to transform into a pandas
# DataFrame later.
df_data = []

# Loop through each saved file in the run folder.
for folder in folders:
    ind_file = 0
    filename = folder + '/' + str(ind_file) + '.p'
    # Read the file if it exists and is not corrupted.
    while os.path.isfile(filename):
        try:
            datafile = open(filename, 'rb')
            data = pickle.load(datafile)
            datafile.close()
        except:
            # Could not successfully unpickle the file,
            # probably because it was not fully written.
            # Ignore it.
            break

        [args, results_data] = data

        # Put the saved data into a dictionary.
        data_dict = dict()
        for key in args:
            if key not in params_to_ignore:
                data_dict[key] = args[key]
        for key in results_data:
            if key not in params_to_ignore:
                data_dict[key] = results_data[key]

        # Calculate additional quantities, such as
        # the weights of the operators on different sites.
        L              = args['L']
        explored_basis = args['explored_basis']

        if args['ham_type'] == '1D_Heisenberg':
            lattice_type = '1D_chain'
            N            = L
        elif args['ham_type'] == '2D_Heisenberg':
            lattice_type = '2D_square'
            N            = L * L
        else:
            raise ValueError('Invalid ham_type: {}'.format(args['ham_type']))
        
        taus                       = results_data['taus']
        num_taus_in_expansion      = results_data['num_taus_in_expansion']
        ind_expansion_from_ind_tau = results_data['ind_expansion_from_ind_tau']
        
        num_taus = len(taus)
        for ind_tau in range(num_taus):
            # The dictionary to save this particular
            # operator's info to.
            tau_dict = dict()

            # The expansion index.
            ind_expansion = ind_expansion_from_ind_tau[ind_tau]
            
            coeffs              = taus[ind_tau]
            inds_explored_basis = results_data['basis_inds'][ind_expansion]
            op_strings          = [explored_basis[ind_os] for ind_os in inds_explored_basis]
            operator            = qy.Operator(coeffs, op_strings)

            # The weight of the operator on each site.
            weights = operator_weights(operator, N)

            # The distribution of k-local operator strings in the operator.
            localities = operator_localities(operator, N)

            # Find an exponential fit of the weights.
            [opt_params, opt_covs] = exp_fit(weights, lattice_type)
            corr_length            = opt_params[0]
            corr_length_err        = np.sqrt(np.abs(opt_covs[0,0])) 
            
            # The histogram of the amplitudes of the operator on each operator string.
            (coeff_hist, bin_edges) = np.histogram(-np.log(np.abs(coeffs)+1e-16)/np.log(10.0), bins = np.linspace(0.0, 16.0, 17))

            # The fidelity of the operator with the previous operator.
            fidelity            = np.nan
            proj_final_fidelity = np.nan
            if ind_tau > 0:
                fidelity            = results_data['fidelities'][ind_tau - 1]
                proj_final_fidelity = results_data['proj_final_fidelities'][ind_tau - 1]
                
            #plt.plot(bin_edges[0:16], coeff_hist, label='W={}'.format(Ws[indWs]))
            #print(coeff_hist)
            #plt.legend()
            #plt.show()
            
            # Report whether this operator is the final
            # optimized one in the current expansion.
            is_final_expanded_tau = (ind_tau == num_taus - 1) or (ind_expansion_from_ind_tau[ind_tau] + 1 == ind_expansion_from_ind_tau[ind_tau + 1])
            
            # Report whether this operator is the final
            # optimized one or not.
            is_final_tau = (ind_tau == num_taus - 1)
            
            # The results to save to the Pandas data frame.
            tau_dict = {
                'com_norm'              : results_data['com_norms'][ind_tau], \
                'binarity'              : results_data['binarities'][ind_tau], \
                'tau_norm'              : results_data['tau_norms'][ind_tau], \
                'fidelity'              : fidelity, \
                'initial_fidelity'      : results_data['initial_fidelities'][ind_tau], \
                'final_fidelity'        : results_data['final_fidelities'][ind_tau], \
                'proj_final_fidelity'   : proj_final_fidelity, \
                'basis_size'            : results_data['basis_sizes'][ind_expansion], \
                'weights'               : weights, \
                'localities'            : localities, \
                'coeff_hist'            : coeff_hist, \
                'corr_length'           : corr_length, \
                'corr_length_err'       : corr_length_err, \
                'ind_tau'               : ind_tau, \
                'ind_expansion'         : ind_expansion, \
                'num_taus'              : num_taus, \
                'is_final_expanded_tau' : is_final_expanded_tau, \
                'is_final_tau'          : is_final_tau
            }
            for key in data_dict:
                tau_dict[key] = data_dict[key]


            # DEBUGGING: Analyze size of tau_dict
            #print('==== TAU {}, {} ===='.format(ind_file, ind_tau))
            #for key in tau_dict:
            #    print('{} : {} GB'.format(key, get_size(tau_dict[key])/1e9))

            #exit(1)
                
            # Add the dictionary for the current operator
            # to the list of dictionaries.
            df_data.append(tau_dict)
            
        print(' Finished collecting data from {}'.format(filename))
                
        ind_file += 1
        filename = folder + '/' + str(ind_file) + '.p'
        
    print('Finished collecting data from {}'.format(folder))

# Create a pandas DataFrame.
df = pd.DataFrame(df_data)
    
# Pickle the pandas DataFrame.
df.to_pickle(output_filename)
