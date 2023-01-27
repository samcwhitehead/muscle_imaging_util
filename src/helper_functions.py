"""
Misc. handy functions that are generally useful for data analysis

Based heavily on:
    https://github.com/psilentp/flylib/blob/master/flylib/util.py
"""
import re
import itertools
import numpy as np
# import pandas as pd


def idx_by_thresh(signal, thresh=0.1):
    """
    Returns a list of index lists, where each index indicates where signal > thresh

    If I'm remembering correctly, there's some ambiguity in the edge cases
    """
    # import numpy as np
    idxs = np.squeeze(np.argwhere((signal > thresh).astype(np.int)))
    try:
        split_idxs = np.squeeze(np.argwhere(np.diff(idxs) > 1))
    except IndexError:
        # print 'IndexError'
        return None
    # split_idxs = [split_idxs]
    if split_idxs.ndim == 0:
        split_idxs = np.array([split_idxs])
    # print split_idxs
    try:
        idx_list = np.split(idxs, split_idxs)
    except ValueError:
        # print 'value error'
        np.split(idxs, split_idxs)
        return None
    idx_list = [x[1:] for x in idx_list]
    idx_list = [x for x in idx_list if len(x) > 0]
    return idx_list


def jpg2np(jpg):
    """convert a cv2 jpg byte string image into a numpy array. Deals
    with nan padding as used in my encoding code"""
    import cv2
    jpgimg = jpg[~np.isnan(jpg)].astype(np.uint8)
    return cv2.imdecode(jpgimg, cv2.IMREAD_COLOR)


def rewrap(trace, offset=np.pi / 2.):
    unwrapped = np.unwrap(trace, np.pi * 1.8)
    vel = np.diff(unwrapped)
    return np.mod(unwrapped + np.deg2rad(offset), 1.9 * np.pi), vel


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.
    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        # linear interpolation of NaNs
        nans, x= nan_helper(y)
        y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """
    import numpy as np
    return ~np.isfinite(y), lambda z: z.nonzero()[0]


def fill_nan(y):
    import numpy as np
    nans, x = nan_helper(y)
    # print np.sum(nans)
    y[nans] = np.interp(x(nans), x(~nans), y[~nans])
    return y


def butter_bandpass(lowcut, highcut, sampling_period, order=5):
    import scipy.signal
    sampling_frequency = 1.0 / sampling_period
    nyq = 0.5 * sampling_frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, sampling_period, order=5):
    import scipy.signal
    b, a = butter_bandpass(lowcut, highcut, sampling_period, order=order)
    y = scipy.signal.filtfilt(b, a, data)
    return y


def butter_lowpass(lowcut, sampling_period, order=5):
    import scipy.signal
    sampling_frequency = 1.0 / sampling_period
    nyq = 0.5 * sampling_frequency
    low = lowcut / nyq
    b, a = scipy.signal.butter(order, low, btype='low')
    return b, a


def butter_lowpass_filter(data, lowcut, sampling_period, order=5):
    import scipy.signal
    b, a = butter_lowpass(lowcut, sampling_period, order=order)
    y = scipy.signal.filtfilt(b, a, data)
    return y


def butter_highpass(highcut, sampling_period, order=5):
    import scipy.signal
    sampling_frequency = 1.0 / sampling_period
    nyq = 0.5 * sampling_frequency
    high = highcut / nyq
    b, a = scipy.signal.butter(order, high, btype='high')
    return b, a


def butter_highpass_filter(data, highcut, sampling_period, order=5):
    import scipy.signal
    b, a = butter_highpass(highcut, sampling_period, order=order)
    y = scipy.signal.filtfilt(b, a, data)
    return y


def scatterplot_matrix(data, names, **kwargs):
    """Plots a scatterplot matrix of subplots.  Each row of "data" is plotted
    against other rows, resulting in a nrows by nrows grid of subplots with the
    diagonal subplots labeled with "names".  Additional keyword arguments are
    passed on to matplotlib's "plot" command. Returns the matplotlib figure
    object containg the subplot grid."""
    import itertools
    from matplotlib import pyplot as plt
    numvars, numdata = data.shape
    fig, axes = plt.subplots(nrows=numvars, ncols=numvars, figsize=(16, 16))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    for ax in axes.flat:
        # Hide all ticks and labels
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        # Set up ticks only on one side for the "edge" subplots...
        if ax.is_first_col():
            ax.yaxis.set_ticks_position('left')
        if ax.is_last_col():
            ax.yaxis.set_ticks_position('right')
        if ax.is_first_row():
            ax.xaxis.set_ticks_position('top')
        if ax.is_last_row():
            ax.xaxis.set_ticks_position('bottom')

    # Plot the data.
    for i, j in zip(*np.triu_indices_from(axes, k=1)):
        for x, y in [(i, j), (j, i)]:
            axes[x, y].plot(data[x], data[y], **kwargs)

    # Label the diagonal subplots...
    for i, label in enumerate(names):
        axes[i, i].annotate(label, (0.5, 0.5), xycoords='axes fraction',
                            ha='center', va='center')

    # Turn on the proper x or y axes ticks.
    for i, j in zip(range(numvars), itertools.cycle((-1, 0))):
        axes[j, i].xaxis.set_visible(True)
        axes[i, j].yaxis.set_visible(True)

    return fig


def symm_matrix_half(m, exclude_diagonal=True):
    # Returns a 1D array of the values in the triangle half of a symmetric matrix
    # Excluding the diagonal optional
    n_rows = np.shape(m)[0]
    if exclude_diagonal:
        collector_size = n_rows * (n_rows + 1) / 2 - n_rows
    else:
        collector_size = n_rows * (n_rows + 1) / 2
    if len(np.shape(m)) > 2:
        collector = np.zeros((collector_size, 2)).astype(type(m[0, 0]))
    else:
        collector = np.zeros(collector_size)
    counter = 1  # 1 if true, 0 if false
    last_endpoint = 0
    for row in range(int(exclude_diagonal), n_rows):
        entries_to_add = m[row, 0:counter]
        collector[last_endpoint:last_endpoint + counter] = entries_to_add
        last_endpoint += counter
        counter += 1
    return collector
