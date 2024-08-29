from pyringdown.waveform.small_angle import small_angle_approx
import numpy as np

A = 7.
b = 0.4
fN = 4.85
phi0 = np.pi/6
offset = 0.04
sigma = 1e-3

fs = 1000
duration = 60
nt = int(duration * fs)
t = np.arange(nt) / fs

wave = np.squeeze(small_angle_approx(t, A, b, fN, phi0, offset=offset))

#add noise
wave += np.random.normal(scale=sigma, size=nt)

data_all = np.r_[fs, 0., wave]

np.savetxt("example_data.txt", data_all)