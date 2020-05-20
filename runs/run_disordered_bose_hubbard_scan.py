"""
A script for finding many approximate l-bits for the 
disordered hard-core Bose-Hubbard model in 2D with
Gaussian disorder for many different random
realizations and disorder strengths. The model implemented
is the one used in the numerics paper (https://arxiv.org/pdf/1805.00056.pdf) 
to match the experiment in (https://science.sciencemag.org/content/sci/352/6293/1547.full.pdf).

To use this script, enter
  >>> python3 -u run_disordered_bose_hubbard_scan.py -I INPUT -S SEED -P PROC
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
    
if ham_type == '1D':
    N = L
elif ham_type == '2D':
    N = L*L
elif ham_type == '3D':
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
    random.gauss(0.0, 1.0)
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
    random_potentials = np.array([random.gauss(0.0, 1.0) for i in range(N)])
    args['random_potentials'] = random_potentials
    
    print('random_potentials = {}'.format(random_potentials), flush=True)
    
    for ind_W in range(num_Ws):
        print('---- W = {} ({}/{}) ----'.format(Ws[ind_W], ind_W + 1, num_Ws), flush=True)
        
        W                        = Ws[ind_W]
        args['W']                = W
        args['results_filename'] = '{}/{}.p'.format(folder, ind_file)
        
        # Skip this calculation if it has already been performed.
        if os.path.isfile(args['results_filename']):
            print('Skipping calculation since it has already been done!', flush=True)
            ind_file += 1
            continue
        
        # The Heisenberg chain.
        if ham_type == '2D':
            H = bioms.bose_hubbard_square(L, periodic=periodic)
        else:
            raise ValueError('Invalid ham_type: {}'.format(ham_type))
        
        # Perturbing magnetic fields.
        # For the Bose-Hubbard model, W is the full-width-half-maximum
        # of a Gaussian distribution, which is related to the standard
        # deviation s by s = W/(2 * sqrt(2 * log(2))).
        std_dev = W / (2.0 * np.sqrt(2.0 * np.log(2.0) ) )
        H += bioms.magnetic_fields(std_dev * random_potentials)
        
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
