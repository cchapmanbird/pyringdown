import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.interpolate import CubicSpline

def compute_frequency_series(data, fs, freq_est=4.848, t_trim=6, freq_bandwidth=0.5):
    """
    Compute the frequency series of a signal by finding the zero crossings of the signal.

    Parameters
    ----------
    data : np.ndarray
        The data to analyse.
    fs : float
        The sampling frequency of the data.
    freq_est : float
        The frequency estimate of the signal.
    t_trim : float
        The time to trim from the edges of the data.
    freq_bandwidth : float
        The bandwidth of the filter to use.
    """
    time = np.arange(len(data)) / fs

    data_in = data.copy()
    data_in -= np.mean(data_in)

    # filter the data with a butterworth bandpass filter from scipy
    filt = butter(4, [freq_est-freq_bandwidth, freq_est+freq_bandwidth], btype='band', fs=fs, output='sos')
    data_in = sosfiltfilt(filt, data_in)

    # now we trim the data edges to remove the filter effects
    ntrim = int(t_trim * fs)
    data_in = data_in[ntrim:-ntrim]

    samples_per_half_rough = int(fs / freq_est / 2)

    zeroes_inds = np.where(data_in[1:] * data_in[:-1] < 0)[0][:-2] # indices of a sign change

    starts = zeroes_inds + samples_per_half_rough // 2 + samples_per_half_rough // 8  # start a bit after the peak

    zero_crossings = np.zeros(len(starts))

    time_seg = np.arange(samples_per_half_rough - samples_per_half_rough // 8) / fs

    for i in range(len(starts)):
        data_seg = data_in[starts[i] : starts[i] + samples_per_half_rough - samples_per_half_rough // 8]
        try:
            desc = (data_seg[10] - data_seg[0] < 0)
        except IndexError:
            zero_crossings[i] = np.nan
            continue
        if desc:
            crossing = CubicSpline(data_seg[::-1], time_seg[::-1])(0)
        else:
            crossing = CubicSpline(data_seg, time_seg)(0)
        
        if crossing > time_seg[-1] or crossing < 0:
            zero_crossings[i] = np.nan
        else:
            zero_crossings[i] = crossing
    
    zero_crossings += time[starts]

    # estimate the period as twice the time between crossings
    periods = np.diff(zero_crossings) * 2
    return zero_crossings[:-1], 1 / periods

