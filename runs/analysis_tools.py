import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.size'] = 16

# TODO: document
def mean_stderr_stats(data_frame, x_var_name, x_var_values, y_var_name, extra_mask=None, fun=None, array_data=False, tol=1e-10):
    """Read a variable from the data frame and plot its
    average and stderr versus another variable.
    """
    
    num_x_data_points = len(x_var_values)

    xs      = []
    means   = []
    stderrs = []
    
    for ind_data_point in range(num_x_data_points):
        x_var_value = x_var_values[ind_data_point]
        
        mask = np.abs(data_frame[x_var_name] - x_var_value) < tol

        if extra_mask is not None:
            mask = mask & extra_mask
        
        data_column = data_frame[mask][y_var_name]
        
        num_samples = data_column.size

        if num_samples == 0:
            continue

        if fun is not None:
            data_column = data_column.apply(fun)

        if array_data:
            mean_column     = data_column.mean()
            mean_sqr_column = (data_column ** 2.0).mean()
            stderr_column   = np.sqrt(np.abs(mean_sqr_column - mean_column ** 2.0)) / np.sqrt(num_samples)
        else:
            mean_column   = data_column.mean()
            stderr_column = data_column.std() / np.sqrt(num_samples)

        xs.append(x_var_value)
        means.append(mean_column)
        stderrs.append(stderr_column)

    inds_sort = np.argsort(x_var_values)
    xs        = [x_var_values[ind] for ind in inds_sort]
    means     = [means[ind] for ind in inds_sort]
    stderrs   = [stderrs[ind] for ind in inds_sort]
        
    return [xs, means, stderrs]

def plot_mean_vs_disorder(data_frame, y_var_name, plot_y_var_name, xaxis_scale=None, yaxis_scale=None, label=None, extra_mask=None):
    Ws = data_frame.W.unique()

    [Ws, means, stderrs] = mean_stderr_stats(data_frame, 'W', Ws, y_var_name, extra_mask=extra_mask)

    plt.errorbar(Ws, means, yerr=stderrs, linewidth=2, fmt='o-', label=label)

    plt.xlabel('$W$')
    plt.ylabel('{}'.format(plot_y_var_name))
    
    if xaxis_scale is not None:
        plt.xscale(xaxis_scale)
    
    if yaxis_scale is not None:
        plt.yscale(yaxis_scale)
        
    plt.show()
    
def plot_mean_arrays_vs_disorder(data_frame, x_vars, plot_x_var_name, y_var_name, plot_y_var_name, Ws=None, xaxis_scale=None, yaxis_scale=None, extra_mask=None):
    if Ws is None:
        Ws = data_frame.W.unique()
    
    for W in Ws:
        [_, means, stderrs] = mean_stderr_stats(data_frame, 'W', [W], y_var_name, array_data=True, extra_mask=extra_mask)

        plt.errorbar(x_vars, means[0], yerr=stderrs[0], linewidth=2, fmt='o-', label='$W =${:10.2f}'.format(W))

    plt.xlabel('{}'.format(plot_x_var_name))
    plt.ylabel('{}'.format(plot_y_var_name))
    plt.legend()
    
    if xaxis_scale is not None:
        plt.xscale(xaxis_scale)
    
    if yaxis_scale is not None:
        plt.yscale(yaxis_scale)
        
    plt.show()
