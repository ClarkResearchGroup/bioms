"""
Read data from find_lbit() runs, convert the data
to a more compressed pandas DataFrame format that 
can be more easily manipulated to generate plots, 
and save the compressed format to file.
"""


import os
import warnings
import pickle
import optparse
import json
import numpy as np
import numpy.linalg as nla
import scipy.sparse as ss
import scipy.optimize as so
import pandas as pd
import matplotlib.pyplot as plt

import qosy as qy

import bioms
from bioms.tools import get_size

# Ignore the OptimizeWarning that happens when scipy.optimize.curve_fit
# tries to fit a function with a single data point.
warnings.filterwarnings('ignore', category=so.OptimizeWarning)

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

# Specify whether to save the full gradient descent
# iteration data to file or only the final converged results.
record_only_converged = input_args['record_only_converged']

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

# Compute the approximate zero-th order estimate
# of the "energy" of the l-bit. This is the overlap
# coefficient <H, \tau^z_i> of the l-bit \tau^z_i with
# the Hamiltonian H, computed only by looking at Z_i.
def lbit_energy(random_potentials, operator, N):
    try:
        Z_center     = qy.opstring('Z {}'.format(N//2))
        ind_Z_center = operator._basis.index(Z_center)
        op_coeff     = operator.coeffs[ind_Z_center]
        H_coeff      = 0.5 * random_potentials[N//2]
        
        return np.real(H_coeff / op_coeff)
    except:
        return 0.0
    
# Compute the "weight" of an operator,
# the probability of each site being
# a non-identity operator.
def operator_weights(operator, num_orbitals, mode=None):
    row_inds = []
    data     = []
    for (coeff, op_string) in operator:
        if mode == 'Z':
            if len(op_string.orbital_operators) == 1 and op_string.orbital_operators[0] == 'Z':
                weight    = np.abs(coeff)**2.0
                orb_label = op_string.orbital_labels[0]
                
                data.append(weight)
                row_inds.append(orb_label)
        else:
            for (op_name, orb_label) in op_string:
                weight = np.abs(coeff)**2.0

                data.append(weight)
                row_inds.append(orb_label)
    col_inds = np.zeros(len(row_inds), dtype=int) 
    weights  = ss.csr_matrix((data, (row_inds, col_inds)), shape=(num_orbitals, 1), dtype=float)

    # Normalize to be a probability distribution.
    weights /= weights.sum()

    return weights

# Compute the "localities" of an operator O = \sum_a g_a S_a,
# the probability \sum_{a for k-local S_a} g_a^2/\sum_b g_b^2
# of an operator string S_a in the expansion being k-local.
def operator_localities(operator, num_orbitals):
    localities = np.zeros(num_orbitals) + 1e-16
    data       = np.zeros(len(operator))
    row_inds   = np.zeros(len(operator), dtype=int)
    col_inds   = np.zeros(len(operator), dtype=int)
    ind_os     = 0
    for (coeff, op_string) in operator:
        # The integer k for a k-local operator string,
        # i.e., the number of sites that the operator
        # string acts on.
        k = len(op_string.orbital_operators)

        data[ind_os]     = np.abs(coeff)**2.0
        row_inds[ind_os] = k
        
        ind_os += 1
        
    localities = ss.csr_matrix((data, (row_inds, col_inds)), shape=(num_orbitals,1), dtype=float)
    
    # Normalize to be a probability distribution.
    localities /= localities.sum()
    
    return localities

# Compute the inverse participation ratio. For a
# probability distribution p_i with \sum_i p_i = 1,
# the IPR is 1.0 / \sum_i p_i^2.
def inverse_participation_ratio(probabilities):
    if isinstance(probabilities, np.ndarray):
        probs = np.abs(probabilities) / np.sum(np.abs(probabilities))
        ipr   = 1.0 / np.sum(probs**2.0)
    else:
        probs = abs(probabilities) / abs(probabilities).sum()
        ipr   = 1.0 / probs.power(2.0).sum()
        
    return ipr

# Find the maximal distance between the orbital
# operators in an operator string.
def os_max_distance(op_string, coord_fun):
    os_range = 0.0
    for (orb_name1, orb_label1) in op_string:
        coords1 = coord_fun(orb_label1)
        for (orb_name2, orb_label2) in op_string:
            coords2  = coord_fun(orb_label2)
            os_range = np.maximum(os_range, nla.norm(coords1 - coords2))
    return os_range

# Find the distance between the orbital
# operators in an operator string and a given coordinate
# in the lattice.
def os_rel_distance(op_string, coord_fun, site0, dist_type):
    coords0  = coord_fun(site0)
    os_range = np.nan
    if dist_type == 'maximum' or dist_type == 'minimum':
        for (orb_name1, orb_label1) in op_string:
            coords1  = coord_fun(orb_label1)
            dist     = nla.norm(coords1 - coords0)
            if np.isnan(os_range):
                os_range = dist
            else:
                if dist_type == 'maximum':
                    os_range = np.maximum(os_range, dist)
                elif dist_type == 'minimum':
                    os_range = np.minimum(os_range, dist)
    elif dist_type == 'center_of_mass':
        coords_com = np.zeros_like(coords0)
        for (orb_name1, orb_label1) in op_string:
            coords1     = coord_fun(orb_label1)
            coords_com += coords1 / float(len(op_string.orbital_operators))

        os_range = nla.norm(coords_com - coords0)
    else:
        raise ValueError('Invalid dist_type: {}'.format(dist_type))    
    
    return os_range

# Compute many different types of "ranges" that measure
# spatial information about an operator based on the
# locations of the operator strings on the sites of the lattice.
def operator_range(operator, L, lattice_type, range_type):
    if lattice_type == '1D_chain':
        N = L
        def coords(site):
            return np.array([site], dtype=float)
    elif lattice_type == '2D_square':
        N = L * L
        def coords(site):
            nonlocal L
            return np.array([site % L, site // L], dtype=float)
    elif lattice_type == '3D_cubic':
        N = L * L * L
        def coords(site):
            nonlocal L
            return np.array([site % L, (site % (L * L)) // L, site // (L * L)], dtype=float)
    else:
        raise ValueError('Invalid lattice_type: {}'.format(lattice_type))

    # Find the average distance between endpoints
    # of operator strings in the operator
    # range_types = ['average', 'maximum', 'maximum_maximum_radius', 'maximum_com_radius', 'average_maximum_radius', 'average_minimum_radius', 'average_com_radius']
    op_norm  = operator.norm()
    op_range = 0.0
    if range_type == 'average':
        for (coeff, op_string) in operator:
            os_range  = os_max_distance(op_string, coords)
            op_range += np.abs(coeff/op_norm)**2.0 * os_range
    # Find the maximum distance between the endpoints of
    # the operator strings in the operator.
    elif range_type == 'maximum':
        for (coeff, op_string) in operator:
            os_range = os_max_distance(op_string, coords)
            op_range = np.maximum(op_range, os_range)
    # Find the maximum maximum distance from the central site.
    elif range_type == 'maximum_maximum_radius':
        for (coeff, op_string) in operator:
            os_range = os_rel_distance(op_string, coords, N//2, 'maximum')
            op_range = np.maximum(op_range, os_range)
    # Find the maximum center-of-mass distance from the central site.
    elif range_type == 'maximum_com_radius':
        for (coeff, op_string) in operator:
            os_range = os_rel_distance(op_string, coords, N//2, 'center_of_mass')
            op_range = np.maximum(op_range, os_range)
    # Find the average maximum distance from the central site. 
    elif range_type == 'average_maximum_radius':
        for (coeff, op_string) in operator:
            os_range  = os_rel_distance(op_string, coords, N//2, 'maximum')
            op_range += np.abs(coeff/op_norm)**2.0 * os_range
    # Find the average minimum distance from the central site.
    elif range_type == 'average_minimum_radius':
        for (coeff, op_string) in operator:
            os_range  = os_rel_distance(op_string, coords, N//2, 'minimum')
            op_range += np.abs(coeff/op_norm)**2.0 * os_range
    # Find the average center-of-mass distance from the central site.
    elif range_type == 'average_com_radius':
        for (coeff, op_string) in operator:
            os_range  = os_rel_distance(op_string, coords, N//2, 'center_of_mass')
            op_range += np.abs(coeff/op_norm)**2.0 * os_range
    else:
        raise ValueError('Invalid range_type: {}'.format(range_type))    
            
    return op_range

# TODO: test!
# Performs an exponential fit
# for the weights of the 1D lbit.
def exp_fit(weights, L, lattice_type):
    # For a 1D chain.
    if lattice_type == '1D_chain':
        x  = []
        y  = []
        for ind_row, ind_col in zip(*weights.nonzero()):
            x.append(ind_row)
            y.append(weights[ind_row, ind_col])
        x = np.array(x)
        y = np.array(y)
        
        ind0 = np.argmax(y)
        x0   = x[ind0]
        
        # The exponential function to fit.
        def func(x, a):
            nonlocal x0
            result = np.exp(-np.abs(x - x0) / a)
            result /= np.sum(result) # Normalize the probability distribution.
            return result
        
        # TODO: write jacobian
        #def jac_func(x, a, b):
        #    return np.array()
        
        try:
            popt, pcov = so.curve_fit(func, x, y, bounds=(1e-16, L))
        except:
            popt, pcov = np.array([0.0]), np.array([[0.0]])
            
    # For a 2D square lattice.
    elif lattice_type == '2D_square':
        Lx = L
        Ly = L
        N  = Lx*Ly

        # The index transformation from a
        # site index to the (x,y) coordinates.
        def coords(site):
            nonlocal Lx
            return [site % Lx, site // Lx]

        # The site where the weights are non-zero.
        sites_nz   = [] 
        weights_nz = []
        for ind_row, ind_col in zip(*weights.nonzero()):
            sites_nz.append(ind_row)
            weights_nz.append(weights[ind_row, ind_col])
        sites_nz   = np.array(sites_nz, dtype=int)
        weights_nz = np.array(weights_nz)
            
        ind0     = np.argmax(weights_nz)
        site0    = sites_nz[ind0]
        [x0, y0] = coords(site0)
        
        # The exponential function to fit.
        def func(sites, a):
            nonlocal x0, y0
            
            result = np.zeros(len(sites))
            for ind_s in range(len(sites)):
                site          = sites[ind_s]
                [x, y]        = coords(site)
                result[ind_s] = np.exp(-nla.norm(np.array([x - x0, y - y0])) / a)
            result /= np.sum(result) # Normalize the probability distribution.
                
            return result
        
        # TODO: write jacobian
        #def jac_func(x, a, b):
        #    return np.array()
        try:
            popt, pcov = so.curve_fit(func, sites_nz, weights_nz, bounds=(1e-16, L))
        except:
            popt, pcov = np.array([0.0]), np.array([[0.0]])
    # For a 3D cubic lattice.
    elif lattice_type == '3D_cubic':
        Lx = L
        Ly = L
        Lz = L
        N  = Lx*Ly*Lz
        
        # The index transformation from a
        # site index to the (x,y) coordinates.
        def coords(site):
            nonlocal Lx, Ly
            return [site % Lx, (site % (Lx * Ly)) // Lx, site // (Lx * Ly)]

        # The site where the weights are non-zero.
        sites_nz   = [] 
        weights_nz = []
        for ind_row, ind_col in zip(*weights.nonzero()):
            sites_nz.append(ind_row)
            weights_nz.append(weights[ind_row, ind_col])
        sites_nz   = np.array(sites_nz, dtype=int)
        weights_nz = np.array(weights_nz)
            
        ind0         = np.argmax(weights_nz)
        site0        = sites_nz[ind0]
        [x0, y0, z0] = coords(site0)
        
        # The exponential function to fit.
        def func(sites, a):
            nonlocal x0, y0, z0
            
            result = np.zeros(len(sites))
            for ind_s in range(len(sites)):
                site          = sites[ind_s]
                [x, y, z]     = coords(site)
                result[ind_s] = np.exp(-nla.norm(np.array([x - x0, y - y0, z - z0])) / a)
            result /= np.sum(result) # Normalize the probability distribution.
                
            return result
        
        # TODO: write jacobian
        #def jac_func(x, a, b):
        #    return np.array()
        try:
            popt, pcov = so.curve_fit(func, sites_nz, weights_nz, bounds=(1e-16, L))
        except:
            popt, pcov = np.array([0.0]), np.array([[0.0]])
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
        elif args['ham_type'] == '3D_Heisenberg':
            lattice_type = '3D_cubic'
            N            = L * L * L
        else:
            raise ValueError('Invalid ham_type: {}'.format(args['ham_type']))
        
        taus                       = results_data['taus']
        num_taus_in_expansion      = results_data['num_taus_in_expansion']
        ind_expansion_from_ind_tau = results_data['ind_expansion_from_ind_tau']
        
        num_taus = len(taus)
        for ind_tau in range(num_taus):

            # Report whether this operator is the final
            # optimized one in the current expansion.
            is_final_expanded_tau = (ind_tau == num_taus - 1) or (ind_expansion_from_ind_tau[ind_tau] + 1 == ind_expansion_from_ind_tau[ind_tau + 1])
            
            # Report whether this operator is the final
            # optimized one or not.
            is_final_tau = (ind_tau == num_taus - 1)

            # Skip the data collection if the operator is not the converged
            # one and we only care about recording the converged data.
            if record_only_converged and not is_final_expanded_tau:
                continue
            
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

            # The weights of the single-site Pauli Z matrices on each site.
            weights_z = operator_weights(operator, N, mode='Z')
            
            # The distribution of k-local operator strings in the operator.
            localities = operator_localities(operator, N)

            # The operator inverse participation ratio
            # based on the weights.
            op_ipr = inverse_participation_ratio(weights)

            # The zero-th order l-bit "energy".
            random_potentials = args['random_potentials']
            op_energy         = lbit_energy(random_potentials, operator, N)
            
            # The distribution of "ranges" of the operator. There are many
            # different types I calculate, so save them to a dictionary.
            range_dict  = dict()
            range_types = ['average', 'maximum', 'maximum_maximum_radius', 'maximum_com_radius', 'average_maximum_radius', 'average_minimum_radius', 'average_com_radius']
            for range_type in range_types:
                range_dict[range_type+'_range'] = operator_range(operator, L, lattice_type, range_type)
            
            # Find an exponential fit of the weights.
            [opt_params, opt_covs] = exp_fit(weights, L, lattice_type)
            corr_length            = opt_params[0]
            corr_length_err        = np.sqrt(np.abs(opt_covs[0,0]))

            # Find an exponential fit of the weights_z.
            [opt_params_z, opt_covs_z] = exp_fit(weights_z, L, lattice_type)
            corr_length_z              = opt_params_z[0]
            corr_length_z_err          = np.sqrt(np.abs(opt_covs_z[0,0])) 
            
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
            
            # The results to save to the Pandas data frame.
            tau_dict = {
                'com_norm'              : results_data['com_norms'][ind_tau], \
                'binarity'              : results_data['binarities'][ind_tau], \
                'tau_norm'              : results_data['tau_norms'][ind_tau], \
                'op_ipr'                : op_ipr, \
                'op_energy'             : op_energy, \
                'fidelity'              : fidelity, \
                'initial_fidelity'      : results_data['initial_fidelities'][ind_tau], \
                'final_fidelity'        : results_data['final_fidelities'][ind_tau], \
                'proj_final_fidelity'   : proj_final_fidelity, \
                'basis_size'            : results_data['basis_sizes'][ind_expansion], \
                'weights'               : weights, \
                'weights_z'             : weights_z, \
                'localities'            : localities, \
                'coeff_hist'            : coeff_hist, \
                'corr_length'           : corr_length, \
                'corr_length_err'       : corr_length_err, \
                'corr_length_z'         : corr_length_z, \
                'corr_length_z_err'     : corr_length_z_err, \
                'ind_tau'               : ind_tau, \
                'ind_expansion'         : ind_expansion, \
                'num_taus'              : num_taus, \
                'is_final_expanded_tau' : is_final_expanded_tau, \
                'is_final_tau'          : is_final_tau
            }
            for key in data_dict:
                tau_dict[key] = data_dict[key]
            for key in range_dict:
                tau_dict[key] = range_dict[key]
            
            # DEBUGGING: Analyze size of tau_dict
            #print('==== TAU {}, {} ===='.format(ind_file, ind_tau))
            #for key in tau_dict:
            #    print('{} : {} GB'.format(key, get_size(tau_dict[key])/1e9))

            #exit(1)
                
            # Add the dictionary for the current operator
            # to the list of dictionaries.
            df_data.append(tau_dict)
            
        print(' Finished collecting data from {}'.format(filename), flush=True)
        
        ind_file += 1
        filename = folder + '/' + str(ind_file) + '.p'
        
    print('Finished collecting data from {}'.format(folder), flush=True)

# Create a pandas DataFrame.
df = pd.DataFrame(df_data)
    
# Pickle the pandas DataFrame.
df.to_pickle(output_filename)
