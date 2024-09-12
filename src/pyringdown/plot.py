import matplotlib.pyplot as plt
import h5py
import numpy as np
import os

def parameter_group_plot(filenames, parameter, mapping_function=None, outname=None, xvalues=None, xlabel=None, ylabel=None, parent_dir=None):
    if xvalues is None:
        xvalues = np.arange(len(filenames))
    if parent_dir is None:
        parent_dir = ''
    if xlabel is None:
        xlabel = ''

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
    ax.plot(xvalues, to_plot)
    ax.set_xlabel(xlabel)
    
    if ylabel is None:
        ax.set_ylabel(parameter)
    else:
        ax.set_ylabel(ylabel)

    if outname is None:
        return fig, ax
    else:
        fig.savefig(outname, bbox_inches="tight")

