from pyringdown.inference.common import run_on_data
from pyringdown.inference.small_angle import TimeDomainModel
import numpy as np

data = np.loadtxt("example_data.txt")

### filter the data or whatever you want... we'll just do nothing for now
pass
###

bounds = dict(
    A=[6.,8.],
    b=[0.3,0.5],
    fN=[4.84,4.86],
    phi0=[0.,2*np.pi],
    offset=[-1.,1.],
    sigma=[0.,5e-3]
)

outdir = "synthetic_example/"

run_on_data(data, TimeDomainModel, bounds, outdir, ncores=4, model_kwargs=dict(downsample=50), resume=False)
