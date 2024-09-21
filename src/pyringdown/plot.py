import matplotlib.pyplot as plt
import h5py
import numpy as np
import os

def parameter_group_plot(filenames, parameter, mapping_function=None, outname=None, xvalues=None, xlabel=None, ylabel=None, parent_dir=None, plot_kwargs=None):
    if xvalues is None:
        xvalues = np.arange(len(filenames))
    if parent_dir is None:
        parent_dir = ''
    if xlabel is None:
        xlabel = ''
    if plot_kwargs is None:
        plot_kwargs = {}

    to_plot = []

    for fn in filenames:
        with h5py.File(os.path.join(parent_dir, fn)) as f:
            if mapping_function is None:
                to_plot.append(f['posterior_samples'][parameter][:].mean())
            else:
                samples = f['posterior_samples']
                remapped = mapping_function(samples)
                to_plot.append(remapped.mean())

    fig, ax = plt.subplots()
    ax.scatter(xvalues, to_plot, **plot_kwargs)
    ax.set_xlabel(xlabel)
    
    if ylabel is None:
        ax.set_ylabel(parameter)
    else:
        ax.set_ylabel(ylabel)

    if outname is None:
        return fig, ax
    else:
        fig.savefig(outname, bbox_inches="tight")

def parameter_group_plot_pair(filenames, parameter_x, parameter_y, mapping_function_x=None, mapping_function_y=None, outname=None, xlabel=None, ylabel=None, parent_dir=None, plot_kwargs=None):
    x_plot = []

    for fn in filenames:
        with h5py.File(os.path.join(parent_dir, fn)) as f:
            if mapping_function_x is None:
                x_plot.append(f['posterior_samples'][parameter_x][:].mean())
            else:
                samples = f['posterior_samples']
                remapped = mapping_function_x(samples)
                x_plot.append(remapped.mean())
    if xlabel is None:
        xlabel = parameter_x
    
    return parameter_group_plot(filenames, parameter_y, mapping_function=mapping_function_y, outname=outname, xvalues=x_plot, xlabel=xlabel, ylabel=ylabel, parent_dir=parent_dir, plot_kwargs=plot_kwargs)