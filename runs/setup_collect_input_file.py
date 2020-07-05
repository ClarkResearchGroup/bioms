"""
Helper script for writing a "input_collect_{run_name}.json" input file (see README.md).
Below is an example of it for the 1D Heisenberg model ({run_name} = 1D_heisenberg1)
which has a seed of 1 and 1600 samples.
"""


info = 'heisenberg1'
run_name        = 'run1D_{}_1_'.format(info)
input_filename  = 'input_collect_run1D_{}.json'.format(info)
output_filename = '"output_run1D_{}"'.format(info) 

input_file = open(input_filename, 'w')

folders_str = ''
run0        = 0
run1        = 1600
num_runs    = run1 - run0
for n in range(run0, run1):
    folders_str += '"{}{}"'.format(run_name, n)
    if n != run1 - 1:
        folders_str += ', '
folders_str = '['+folders_str+']'

file_contents = """{
    "folders"               : """+folders_str

file_contents += """,    
    "output_filename"       : """+output_filename+""",
    "record_only_converged" : true
}
"""

input_file.write(file_contents)
input_file.close()

