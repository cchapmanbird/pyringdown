from nessai.utils.logging import setup_logger
from nessai.flowsampler import FlowSampler
from nessai.plot import corner_plot
from pyringdown.inference.small_angle import TimeDomainModel
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import time
from typing import Union, Optional
from pathlib import Path

def run_on_data(
        data: np.ndarray, 
        model: TimeDomainModel, 
        bounds: dict, 
        outdir: Union[str, Path], 
        ncores: int=1, 
        model_kwargs: Optional[dict]=None, 
        logger_kwargs: Optional[dict]=None, 
        plot_posterior: bool=True, 
        plot: bool=True, 
        **sampler_kwargs
    ):
    """
    A convenience function for running a time domain analysis.

    Args:
        data: A 1-d array of time domain data to analyse. 
        model: An uninitialised model class to run the analysis with (currently only supports pyringdown.inference.small_angle.TimeDomainModel).
        bounds: A dictionary of parameter-(min, max) pairs to define the prior bounds on each parameter.
        outdir: A path to a directory in which to place the analysis results and diagnostic plots (will be created if it does not exist).
        ncores: Number of cores to use for multiprocessing. If equal to 1, no pool is initialised.
        model_kwargs: Keyword arguments to supply to `model`.
        logger_kwargs: Keyword argmuents to supply to the `nessai` logger.
        plot_posterior: A boolean to determine whether `nessai` produces a scatter corner plot of the posterior samples. Defaults to True.
        plot: A boolean to determine whether pyringdown will produce some final diagnostic plots of the fit residuals. Defaults to True.
        **sampler_kwargs: Keyword arguments to pass to the `nessai` `FlowSampler` class.
    """
    _sampler_kwarg_defaults = {
        "nlive": 1000,
        "seed": 42,
        "plot": True,
        "checkpointing": True,
        # "log_on_iteration": True,
        # "logging_interval": 1000,
        "resume": True,
    }

    if model_kwargs is None:
        model_kwargs = {}
    if logger_kwargs is None:
        logger_kwargs = {}
    if sampler_kwargs is None:
        sampler_kwargs = {}
    
    _sampler_kwarg_defaults.update(sampler_kwargs)

    inference_model = model(bounds, data, **model_kwargs)
    setup_logger(outdir, **logger_kwargs)

    if ncores != 1:
        from multiprocessing import Pool
        from nessai.utils.multiprocessing import initialise_pool_variables

        pool = Pool(
            processes=ncores,
            initializer=initialise_pool_variables,
            initargs=(inference_model,),
        )
    else:
        pool = None

    point = inference_model.new_point()
    lh = inference_model.log_likelihood(point)
    st = time.perf_counter()
    lh = inference_model.log_likelihood(point)
    et = time.perf_counter()
    print("Lhood cost:", et-st)
    fs = FlowSampler(inference_model, output=outdir, pool=pool, **_sampler_kwarg_defaults)

    fs.run(plot_posterior=plot_posterior)

    if plot:
        corner_plot(
            fs.posterior_samples,
            exclude=["logL","logP","it"],
            filename=os.path.join(outdir,"corner_plot.pdf")
        )
        plt.close()

        best_point = np.atleast_1d(fs.posterior_samples[-1])
        # waveform = np.squeeze(inference_model.waveform(best_point['A'], best_point['b'], best_point['fN'], best_point['phi0'], best_point['offset'], best_point['m_drift']))
        names_no_sigma = []
        for nm in inference_model.names:
            if "sigma" not in nm:
                names_no_sigma.append(nm)
        waveform = np.squeeze(inference_model.waveform(*[best_point[nm].item() for nm in names_no_sigma]))

        plt.figure(figsize=(15, 8))
        plt.plot(inference_model.t_eval, inference_model.data, c='k', label="Data")
        plt.plot(inference_model.t_eval, waveform, ls='--', c='r', label="Fit ringdown")
        plt.legend()
        plt.xlim(0, 15)
        plt.savefig(os.path.join(outdir,"fit_ringdown.pdf"))
        plt.close()

        try:
            var = (waveform * best_point["sigma_A"])**2 + best_point["sigma"]**2
            residuals = (inference_model.data - waveform) / var**0.5
        except ValueError:
            var = best_point["sigma"]**2
            residuals = (inference_model.data - waveform) / var**0.5

        plt.figure(figsize=(15, 8))
        plt.scatter(inference_model.t_eval, residuals, s=0.5, c='k',rasterized=True)
        # plt.legend()
        # plt.xlim(0, 15)
        plt.savefig(os.path.join(outdir,"fit_ringdown_residual.pdf"))
        plt.close()

        plt.hist(residuals, bins='auto',density=True)
        plotvec = np.linspace(-4, 4, 1001)
        plt.plot(plotvec, norm.pdf(plotvec))
        plt.savefig(os.path.join(outdir,"fit_residuals_hist.pdf"))
        plt.close()

    if pool is not None:
        pool.close()
