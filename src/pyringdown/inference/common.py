from nessai.utils.logging import setup_logger
from nessai.flowsampler import FlowSampler
from nessai.plot import corner_plot
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import time


def run_on_data(data, model, bounds, outdir, ncores=1, model_kwargs=None, logger_kwargs=None, plot_posterior=True, **sampler_kwargs):

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
