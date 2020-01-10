"""
A script for finding many approximate l-bits for the 
disordered Heisenberg chain for many different random
realizations and disorder strengths.

To use this script, enter
  >>> python3 -u run_disordered_heisenberg_scan.py -I INPUT -S SEED
where "FOLDER" is the folder you would like to save the results
to and "SEED" is an integer seed for the random number generator.
"""

import os
import optparse
import json
import numpy as np

import qosy as qy

from context   import bioms
from operators import single_site_parity

# Parse the input arguments.
parser = optparse.OptionParser()
parser.add_option('-I', '--input', type='str', dest='input_filename', help='Input file specifying the run parameters.')
parser.add_option('-S', '--seed', type='int', dest='seed', help='Random number generator seed.')

(options, args) = parser.parse_args()

# The input filename.
input_filename = options.input_filename

# The seed for the random number generator.
seed = options.seed
np.random.seed(seed)

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
num_samples = args['num_samples']
folder      = args['folder']

# Create the folder to save information to.
if not os.path.isdir(folder):
    os.mkdir(folder)

# Store all evaluated commutators and products of Majorana
# strings encountered during the calculations.
explored_com_data     = [qy.Basis(), qy.Basis(), dict()]
explored_anticom_data = [qy.Basis(), qy.Basis(), dict()]

# Args are used for two things:
# 1. To specify the input parameters to find_binary_iom()
# 2. To save run information to file for later reference.
args['explored_com_data']     = explored_com_data
args['explored_anticom_data'] = explored_anticom_data

# Loop through disordered realizations and the disorder
# strengths. Find approximate l-bits and save the results
# to file.
ind_file = 0
for ind_sample in range(num_samples):
    print('++++ SAMPLE {}/{} ++++'.format(ind_sample + 1, num_samples))
    
    args['ind_sample'] = ind_sample
    
    # Random realization of magnetic field strengths.
    # Note that the same random pattern is used for
    # all of the disorder strengths.
    random_potentials = 2.0*np.random.rand(L) - 1.0
    args['random_potentials'] = random_potentials
    
    for ind_W in range(num_Ws):
        print('---- W = {} ({}/{}) ----'.format(Ws[ind_W], ind_W + 1, num_Ws))
        
        W                        = Ws[ind_W]
        args['W']                = W
        args['results_filename'] = '{}/{}.p'.format(folder, ind_file)
        
        # Hard-coded parameters for Heisenberg model.
        J_xy = 1.0
        J_z  = 1.0

        # The Heisenberg chain.
        H = bioms.xxz_chain(L, J_xy, J_z, periodic=periodic)
        # Perturbing magnetic fields.
        H += bioms.magnetic_fields(W * random_potentials)

        # The initial operator centered at the center of the chain.
        initial_op = single_site_parity(L//2, L)

        ### Run find_binary_iom().
        [op, com_norm, binarity, results_data] = bioms.find_binary_iom(H, initial_op, args)
        
        ind_file += 1
