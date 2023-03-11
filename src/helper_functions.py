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


def get_rframe_transform(self,other):
    """get transform into self from other frame (muscle imaging reference frames)"""
    A1 = np.hstack((self['A'],self['p'][:,np.newaxis]))
    A2 = np.hstack((other['A'],other['p'][:,np.newaxis]))
    A1 = np.vstack((A1, [0,0,1]))
    A2 = np.vstack((A2, [0,0,1]))

    return np.dot(A1, np.linalg.inv(A2))


def construct_rframe_dict(e1, e2):
    """Make a dictionary containing refrence frame info"""
    frame = dict()

    frame['e1'] = e1
    frame['e2'] = e2

    frame['a2'] = e1[1] - e2[0]
    frame['a1'] = e2[1] - e2[0]
    frame['p'] = e2[0]

    # also get transformation matrices based on these vectors
    frame['A'] = np.vstack((frame['a1'], frame['a2'])).T
    frame['A_inv'] = np.linalg.inv(frame['A'])

    return frame


###################################################################################################
# Adjust Spines (Dickinson style, thanks to Andrew Straw)
###################################################################################################

def adjust_spines(ax, spines, spine_locations={}, xticks=None, yticks=None, linewidth=1,
                  tight_tick_bounds=True):
    if type(spines) is not list:
        spines = [spines]

    # get ticks
    if xticks is None:
        xticks = ax.get_xticks()
    if yticks is None:
        yticks = ax.get_yticks()

    # remove axis ticks beyond data range
    if tight_tick_bounds:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        xticks = [tick for tick in xticks if (tick >= xlim[0]) and (tick <= xlim[1])]
        yticks = [tick for tick in yticks if (tick >= ylim[0]) and (tick <= ylim[1])]

    spine_locations_dict = {'top': 10, 'right': 10, 'left': 10, 'bottom': 10}
    for key in spine_locations.keys():
        spine_locations_dict[key] = spine_locations[key]

    if 'none' in spines:
        for loc, spine in ax.spines.items():
            spine.set_color('none')  # don't draw spine
        ax.yaxis.set_ticks([])
        ax.xaxis.set_ticks([])
        return

    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', spine_locations_dict[loc]))  # outward by x points
            spine.set_linewidth(linewidth)
            spine.set_color('black')
        else:
            spine.set_color('none')  # don't draw spine

    # smart bounds, if possible
    for loc, spine in ax.spines.items():
        ticks = None
        if loc in ['left', 'right']:
            ticks = yticks
        if loc in ['top', 'bottom']:
            ticks = xticks
        if ticks is not None and len(ticks) > 0:
            spine.set_bounds(ticks[0], ticks[-1])

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    elif 'right' in spines:
        ax.yaxis.set_ticks_position('right')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    if 'top' in spines:
        ax.xaxis.set_ticks_position('top')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])

    if 'left' in spines or 'right' in spines:
        ax.set_yticks(yticks)
    if 'top' in spines or 'bottom' in spines:
        ax.set_xticks(xticks)

    for line in ax.get_xticklines() + ax.get_yticklines():
        # line.set_markersize(6)
        line.set_markeredgewidth(linewidth)


###################################################################################################
# Axis not so tight (based on some MATLAB code)
###################################################################################################
def axis_not_so_tight(ax=None, axis='both', add_percent=0.1):
    """
    Function to add a little white space after an axis tight call

    :param ax: axis object to enjorce limits on
    :param axis: string to determine whether we alter x axis, y axis,
                    or both ('x' | 'y' | 'both')
    :param add_percent: percentage of total axis limit to add

    :return: ax, the adjusted axis object
    """
    import matplotlib.pyplot as plt

    # get axis if we have to
    if ax is None:
        ax = plt.gca()

    # set axis tight
    ax.autoscale(enable=True, axis=axis, tight=True)

    # get current x and y limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # enlarge with X percent of this range
    x_add = add_percent * np.diff(xlim)
    y_add = add_percent * np.diff(ylim)

    # set axes with updated limits
    ax.set_xlim(xlim[0] - x_add, xlim[-1] + x_add)
    ax.set_ylim(ylim[0] - y_add, ylim[-1] + y_add)

    # return adjusted axis
    return ax

