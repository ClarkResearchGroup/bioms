"""
Combine multiple saved pickled
Pandas data frames into a single
pickled Pandas data frame.
"""

import pickle
import optparse
import json
import pandas as pd

# Parse the input arguments.
parser = optparse.OptionParser()
parser.add_option('-I', '--input', type='str', dest='input_filename', help='Input file specifying the data to collect and analyze.')
parser.add_option('-N', '--nfiles', type='int', dest='num_files', help='Number of files to combine.')

(options, args) = parser.parse_args()

# The input filename.
input_filename = options.input_filename
num_files      = options.num_files

# Read the input file.
input_file = open(input_filename, 'r')
input_args = json.load(input_file)
input_file.close()

# The combined output filename to save the collected data to.
combined_output_filename = '{}.p'.format(input_args['output_filename'])

list_of_dfs = []
for ind_file in range(num_files):
    output_filename = '{}_{}.p'.format(input_args['output_filename'], ind_file)
    df_file         = pd.read_pickle(output_filename)
    list_of_dfs.append(df_file)
    
# Create a combined pandas DataFrame.
df = pd.concat(list_of_dfs)
    
# Pickle the pandas DataFrame.
df.to_pickle(combined_output_filename)
