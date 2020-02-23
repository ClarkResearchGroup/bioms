import os

num_badfiles = 0
num_runs     = 480
for ind_run in range(num_runs):
    folder_name  = 'run3D_constant_3_{}'.format(ind_run)
    out_filename = '{}/out.txt'.format(folder_name)

    out_file = open(out_filename, 'r')

    indW          = -1
    ind_expansion = -1
    currW_good    = True
    for line in out_file:
        if "W = " in line:
            if not currW_good:
                result_filename     = '{}/{}.p'.format(folder_name, indW)
                new_result_filename = '{}/{}_bad.p'.format(folder_name, indW)

                if os.path.isfile(result_filename):
                    os.rename(result_filename, new_result_filename)
                    print('{} is bad! Renamed it to {}'.format(result_filename, new_result_filename), flush=True)
                    num_badfiles += 1
            indW         += 1
            ind_expansion = -1
            currW_good    = True
        if "Iteration " in line:
            ind_expansion += 1

        # Mark the whole run as bad if at any point
        # the memory was emptied.
        if "vmem after emptying" in line:
            currW_good = False
    
    out_file.close()
    
    # An extra check for the final file in the list.
    if not currW_good:
        result_filename     = '{}/{}.p'.format(folder_name, indW)
        new_result_filename = '{}/{}_bad.p'.format(folder_name, indW)
        
        if os.path.isfile(result_filename):
            os.rename(result_filename, new_result_filename)
            print('{} is bad! Renamed it to {}'.format(result_filename, new_result_filename), flush=True)
            num_badfiles += 1

print('Number of bad files found: {}'.format(num_badfiles), flush=True)

