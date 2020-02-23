import os

num_runs = 1600
for ind_run in range(num_runs):
    folder_name  = 'run2D_constant_2_{}'.format(ind_run)
    out_filename = '{}/out.txt'.format(folder_name)

    out_file = open(out_filename, 'w')

    indW          = -1
    ind_expansion = -1
    currW_good    = True
    for line in out_file:
        if "W = " in line:
            if not currW_good:
                result_filename     = '{}/{}.p'.format(folder_name, indW)
                new_result_filename = '{}/{}_bad.p'.format(folder_name, indW)

                if os.isfile(result_filename):
                    #os.rename(result_filename, new_result_filename)
                    print('{} is bad! Renamed it to {}'.format(result_filename, new_result_filename), flush=True)
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
