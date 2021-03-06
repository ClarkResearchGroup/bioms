"""
A script for finding many approximate l-bits for the 
disordered Heisenberg chain for many different random
realizations and disorder strengths.

To use this script, enter
  >>> python3 -u run_disordered_heisenberg_scan.py -I INPUT -S SEED -P PROC
where "FOLDER" is the folder you would like to save the results
to, "SEED" is an integer seed for the random number generator,
and "PROC" is an integer labeling the current processor ID number.
"""

from mpi4py import MPI

import sys
import os
import psutil
import optparse
import json
import numpy as np
import random
import pickle

import qosy as qy

import bioms
from operators import single_site_parity

# Parse the input arguments.
parser = optparse.OptionParser()
parser.add_option('-I', '--input', type='str', dest='input_filename', help='Input file specifying the run parameters.')
parser.add_option('-S', '--seed', type='int', dest='seed', help='Random number generator seed.')
parser.add_option('-P', '--proc', type='int', dest='proc', help='Process ID number offset.', default=0)

(options, args) = parser.parse_args()

# The input filename.
input_filename = options.input_filename

# Read the input file.
input_file = open(input_filename, 'r')
args       = json.load(input_file)
input_file.close()

# Read the run parameters that specify the Hamiltonians
# to consider and the number of samples to try.
L           = args['L'] 
periodic    = args['periodic']
Ws          = args['Ws']
num_Ws      = len(Ws)
start_mode  = args['start_mode']
num_samples = args['num_samples']
folder      = args['folder']
ham_type    = args['ham_type']

# The name of the output file to print info to.
if 'out_name' in args:
    out_name = args['out_name']
else:
    out_name = 'out.txt'
    
if ham_type == '1D_Heisenberg':
    N = L
elif ham_type == '2D_Heisenberg':
    N = L*L
elif ham_type == '3D_Heisenberg':
    N = L*L*L
else:
    raise ValueError('Invalid ham_type: {}'.format(ham_type))

# Instantiate the random number generator.
seed = options.seed
proc = options.proc + MPI.COMM_WORLD.rank # The processor id offset plus the MPI rank.
# Each of the parallel processes uses the same seed.
random.seed(seed)
# But starts at a different point in the RNG.
for step in range(proc * N * num_samples):
    random.random()
# Modify the folder name to include the seed and processor info.
folder += '_{}_{}'.format(seed, proc)

# Create the folder to save information to.
if not os.path.isdir(folder):
    os.mkdir(folder)

# Set the output file to pipe standard out to.
out_file   = open('{}/{}'.format(folder, out_name), 'w')
sys.stdout = out_file

nprocs = MPI.COMM_WORLD.Get_size()
print('Rank: {} ({} processes)'.format(MPI.COMM_WORLD.rank, nprocs), flush=True)

# Save the random number generator and parallel processor info.
args['seed'] = seed
args['proc'] = proc

# Store all evaluated commutators and products of Majorana
# strings encountered during the calculations.

# Args are used for two things:
# 1. To specify the input parameters to find_binary_iom()
# 2. To save run information to file for later reference.
args['explored_basis']        = qy.Basis()
args['explored_com_data']     = dict()
args['explored_anticom_data'] = dict()

# Loop through disordered realizations and the disorder
# strengths. Find approximate l-bits and save the results
# to file.
ind_file = 0
for ind_sample in range(num_samples):
    print('++++ SAMPLE {}/{} ++++'.format(ind_sample + 1, num_samples), flush=True)
    
    args['ind_sample'] = ind_sample
    
    # Random realization of magnetic field strengths.
    # Note that the same random pattern is used for
    # all of the disorder strengths.
    random_potentials = 2.0*np.array([random.random() for i in range(N)]) - 1.0
    args['random_potentials'] = random_potentials
    
    print('random_potentials = {}'.format(random_potentials), flush=True)
    
    for ind_W in range(num_Ws):
        print('---- W = {} ({}/{}) ----'.format(Ws[ind_W], ind_W + 1, num_Ws), flush=True)
        
        W                        = Ws[ind_W]
        args['W']                = W
        args['results_filename'] = '{}/{}.p'.format(folder, ind_file)
        
        # Skip this calculation if it has already been performed.
        if os.path.isfile(args['results_filename']):
            # Check if the file is a valid pickled file that can be read.
            # If the pickled file is corrupted (because it was not fully written
            # when the program crashed), then it would be missing an EOF character
            # and produce an error when read.
            correctly_pickled = True
            try:
                datafile = open(args['results_filename'], 'rb')
                data     = pickle.load(datafile)
                datafile.close()
                del data
            except:
                correctly_pickled = False
                print('The pickled file {} was corrupted and could not be read. Calculation is being redone.'.format(args['results_filename']), flush=True)
                
            # Only if the file was correctly pickled then you can safely go
            # to the next calculation.
            if correctly_pickled:
                print('Skipping calculation since it has already been done!', flush=True)
                ind_file += 1
                continue
        
        # Hard-coded parameters for Heisenberg model.
        J_xy = 1.0
        J_z  = 1.0
        
        # The Heisenberg chain.
        if ham_type == '1D_Heisenberg':
            H = bioms.xxz_chain(L, J_xy, J_z, periodic=periodic)
        elif ham_type == '2D_Heisenberg':
            H = bioms.xxz_square(L, J_xy, J_z, periodic=periodic)
        elif ham_type == '3D_Heisenberg':
            H = bioms.xxz_cubic(L, J_xy, J_z, periodic=periodic)
        else:
            raise ValueError('Invalid ham_type: {}'.format(ham_type))
        
        # Perturbing magnetic fields.
        H += bioms.magnetic_fields(W * random_potentials)
        
        # The initial operator centered at the center of the lattice.
        initial_op = single_site_parity(N//2, N, mode=start_mode)
        
        ### Run find_binary_iom().
        [op, com_norm, binarity, results_data] = bioms.find_binary_iom(H, initial_op, args)
        
        # Clear some unnecessary memory.
        del results_data

        sys.stdout.flush()
        
        ind_file += 1

# Free memory at the end of the run (important for MPI applications).
del args

# Reset standard output and close the output file.
sys.stdout = sys.__stdout__
out_file.close()
