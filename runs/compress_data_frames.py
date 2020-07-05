# Saves compacted version of the pandas data frames that only contains
# the numbers (ranges, comnorms, etc) and not the arrays (weights, localities, etc).

import time
import numpy as np
import pandas as pd

# For comparing value to zero.
tol = 1e-15

# For computing avg_k from localities.
def avg_k(localities):
    if isinstance(localities, np.ndarray):
        locs = localities
    else:
        locs = localities.toarray().flatten()
    
    locs /= np.sum(locs)
    
    ks     = np.arange(len(locs))
    result = np.sum(locs * ks)
    
    return result

# For computing the two-point estimate
# of the correlation length in the x-direction
# from the weights.
def corr_length_2pt(weights):
    # The number of sites
    N            = weights.shape[0]
    # The center site (where weight is maximized)
    ind_center   = weights.argmax()
    # Should be one site over from the center
    # site in the x-direction for all of the models.
    ind_neighbor = ind_center + 1
    
    if np.abs(weights[ind_neighbor,0]) > tol and np.abs(weights[ind_center,0]) > tol:
        log_slope      = np.log(weights[ind_neighbor,0]/weights[ind_center,0])
        corr_length2pt = -1.0/log_slope
    else:
        corr_length2pt = N

    return corr_length2pt
    
    
# Folder to save data to.
output_folder = 'compressed_data/'

# Folder where the run files are read from.
input_folder = '' #'/media/echertkov/My Passport/Research/runs/forward_problems/lbits/'

# The run files whose data to save.
# Lists of files while have their data combined.
run_files = [['output_run1D_heisenberg1'],
             ['output_run2D_heisenberg1'],
             ['output_run3D_heisenberg1'],
             ['output_run2D_bosehubbard1']]

run_names = ['heisenberg1D', 'heisenberg2D', 'heisenberg3D', 'bosehubbard2D']

extra_filename_suffix = ''

# List of variables we DO NOT want to save in the data frame because they take up too much space.
vars_to_ignore = ['Ws', 'weights', 'weights_z', 'localities', 'coeff_hist', 'weights_abs', 'weights_z_abs', 'localities_abs', 'random_potentials']

for (run_files, run_name) in zip(run_files, run_names):
    print('=== RUN {} ==='.format(run_name), flush=True)
    
    # The data to save to file.
    list_of_dfs = []
    for run_file in run_files:
        start = time.time()
        run_filename  = input_folder + run_file + '.p'
        data_frame    = pd.read_pickle(run_filename)
        end   = time.time()
        print(' READ FILE: {} in {} seconds'.format(run_filename, end-start), flush=True)
        
        # Remove variables from the vars_to_ignore list,
        # add an "avg_k" variable computed from the localities,
        # add two-point correlation lengths computed from the
        # weights.
        avg_ks     = data_frame['localities'].apply(avg_k)
        avg_abs_ks = data_frame['localities_abs'].apply(avg_k)

        corr_length2pt_weights       = data_frame['weights'].apply(corr_length_2pt)
        corr_length2pt_weights_abs   = data_frame['weights_abs'].apply(corr_length_2pt)
        corr_length2pt_weights_z     = data_frame['weights_z'].apply(corr_length_2pt)
        corr_length2pt_weights_z_abs = data_frame['weights_z_abs'].apply(corr_length_2pt)


        data_frame.drop(vars_to_ignore, axis='columns', inplace=True)
        data_frame.loc[:, 'average_k']     = avg_ks
        data_frame.loc[:, 'average_abs_k'] = avg_abs_ks
            
        data_frame.loc[:, 'corr_length2pt_weights']       = corr_length2pt_weights
        data_frame.loc[:, 'corr_length2pt_weights_abs']   = corr_length2pt_weights_abs
        data_frame.loc[:, 'corr_length2pt_weights_z']     = corr_length2pt_weights_z
        data_frame.loc[:, 'corr_length2pt_weights_z_abs'] = corr_length2pt_weights_z_abs
        
        list_of_dfs.append(data_frame)
        
        # Clear the memory of the data frames.
        del data_frame
    
    compressed_df = pd.concat(list_of_dfs)
    
    out_filename  = output_folder + run_name + extra_filename_suffix + '_compressed.p'
    compressed_df.to_pickle(out_filename)
    
    print('  SAVED FILE: {}'.format(out_filename), flush=True)
    
    # Clear the memory of the data frame.
    del compressed_df
    del list_of_dfs
