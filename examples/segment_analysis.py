from pyringdown.inference.small_angle import maximum_likelihood_segmented_analysis
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("example_data.txt")

bounds = dict(
    A=[1e-5,8.],
    b=[0.3,0.5],
    fN=[4.84,4.86],
    phi0=[0.,2*np.pi],
    offset=[-1.,1.],
    sigma=[1e-5,5e-3]
)

stride = 0.2  # time in seconds between each segment
segwidth = 10  # duration of each segment in seconds

result = maximum_likelihood_segmented_analysis(data, bounds, stride, segwidth, downsample=50)

# plot the variation in each parameter

fig, ax = plt.subplots(nrows=6, figsize=(6,15))

xvals = np.arange(result.shape[0]) * stride

for i, (nm, vals) in enumerate(zip(bounds.keys(), result.T)):
    ax[i].scatter(xvals, vals, marker='+', s=20)
    ax[i].set_ylabel(nm)
    if i != 5:
        ax[i].set_xticklabels([])

ax[-1].set_xlabel("Time [s]")
ax[0].set_title(f"Stride, segwidth = {stride:.2f}, {segwidth:.2f}")
plt.tight_layout()
plt.savefig("segments_example.pdf", bbox_inches="tight")
plt.close()
