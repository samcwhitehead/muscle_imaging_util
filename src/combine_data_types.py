"""
Code to take data from multiple modalities acquired during muscle imaging experiments and both:
    1) resample all the data to a common frequency/clock
    2) combine useful data types in a single file

NB: in contrast to previous analysis pipelines, this one requires that you do the muscle unmixing first, so
everything is basically 1D time series

"""
# -----------------------------------------------------------------
import sys
import os
import h5py

import pandas as pd
import numpy as np

################################################################################################
################################ GENERAL INPUTS ################################################
################################################################################################
FLY_DB_PATH = '/media/sam/SamData/FlyDB'  # directory containing fly data
# FLY_DB_PATH = '/media/imager/DataExternal/FlyDB'

FLIES = None  # [18]  # None  # list of flies to analyze. If None, look for command line input
FOLDER_STR = 'Fly%04d'  # general format of folders containing fly data -- only used if FLIES contains ints

# from Thad's code, frequency in Hz to resample data at:
RESAMPLE_FREQ = 50

# interpolation method (scipy.interpolate.interp1d)
INTERP_KIND = 'previous'  # 'nearest'

# define a dictionary that will let us look up the appropriate filename per variables
GENERAL_H5_STR = 'converted%s.hdf5'
SIGNALS_STR = 'ca_camera_%s%s_model_fits.hdf5'  # 1st string should be side ('right' or 'left'); 2nd should be file suffix
KINE_CAM_STR = 'kine_camera_1%s.hdf5'  # kind of silly, but I need to get the time stamps from this

# in this dict, the key will be a variable and the entry will be the hdf5 group
# NB: this doesn't include extracted muscle signals, which we should be able to load in a more straightfwd way
VAR_DICT = {'left_amp': 'flystate_langles',
            'left_amp_t': 'flystate_tstamps',
            'right_amp': 'flystate_rangles',
            'right_amp_t': 'flystate_tstamps',
            'exp_block': 'exp_block/msgs',
            'exp_block_t': 'exp_block/ros_tstamps',
            'exp_state': 'exp_state/msgs',
            'exp_state_t': 'exp_state/ros_tstamps',
            'daq_channels': 'daq_channels',
            'daq_vals': 'daq_value',
            'daq_t': 'daq_ros_tstamps',
            }

# make a little helper list to deal with DAQ fields
DAQ_FIELDS = {'wb_freq': 'freq',
              'x_pos': 'xpos',
              'y_pos': 'ypos',
              }


################################################################################################
########################### HELPER FUNCTIONS ###################################################
################################################################################################
def load_daq_data(daq_channels, daq_vals, var_name):
    """
    helper function to load daq data, since it's stored in channels
    """
    if sys.version_info[0] < 3:
        pass
    else:
        var_name = var_name.encode('utf-8')
    idx = np.where(daq_channels == var_name)
    return daq_vals[idx[0], idx[1]]


# -----------------------------------------------------------------------------------
def get_fly_path(fly, fly_db_path=FLY_DB_PATH, folder_str=FOLDER_STR):
    """
    helper function to get filename for fly based on some identifier
    """
    # get path to fly data folder and filename suffix (depends on whether or not we're using integer index or filename)
    if isinstance(fly, int):
        fly_path = os.path.normpath(os.path.join(fly_db_path, folder_str % (fly)))
        file_suffix = ''
    else:
        fly_path = os.path.normpath(fly_db_path)
        file_suffix = '_' + fly
        file_suffix = os.path.splitext(file_suffix)[0]  # just to make sure '.bag' isn't at the end

    # return both
    return (fly_path, file_suffix)


# -----------------------------------------------------------------------------------
def my_interp(t, dat, resamp_t, interp_kind=INTERP_KIND):
    """
    helper function to do the interpolation to common clock
    """
    # define interpolant
    from scipy.interpolate import interp1d
    interp_f = interp1d(t, dat, kind=interp_kind, fill_value='extrapolate')

    # evaluate interpolant at new time points
    return interp_f(resamp_t)


################################################################################################
####################### FUNCTIONS TO LOAD AND INTERP ###########################################
################################################################################################
def load_all_data(fly, var_dict=VAR_DICT, general_h5_str=GENERAL_H5_STR, signals_str=SIGNALS_STR,
                  kine_cam_str=KINE_CAM_STR, fly_db_path=FLY_DB_PATH, folder_str=FOLDER_STR):
    """
    Function to go over the various hdf5 files in which different signals are saved, and group them into one dict
    """
    # intialize storage for dict that we'll load data into
    load_dict = dict()

    # get path to folder containing data for current fly
    fly_path, file_suffix = get_fly_path(fly, fly_db_path=fly_db_path, folder_str=folder_str)

    # locate hdf5 file containing non-imaging variables
    converted_h5_path = os.path.join(fly_path, general_h5_str %(file_suffix))

    # loop over variable dict and load data into load_dict
    with h5py.File(converted_h5_path, 'r') as h5f:
        for var in var_dict.keys():
            load_dict[var] = h5f[var_dict[var]][:]

    # also grab muscle data
    for side in ['left', 'right']:
        signals_path = os.path.join(fly_path, signals_str %(side, file_suffix))

        # make sure data exists for the side we're talking about
        if not os.path.exists(signals_path):
            print('Error: could not locate %s' %(signals_path))
            continue

        # assuming it exists, transfer data to load_dict, but append 'left' or 'right' and 'muscle' to the key
        with h5py.File(signals_path, 'r') as h5f:
            for key in h5f.keys():
                load_dict[str('_'.join([side, 'muscle', key]))] = h5f[key][:]

    # finally, get ros time stamps (should be same as "regular" tstamps) for kinefly camera -- used for globat t var
    kine_cam_path = os.path.join(fly_path, kine_cam_str %(file_suffix))
    with h5py.File(kine_cam_path, 'r') as h5f:
        load_dict['kine_cam_t'] = h5f['cam_ros_tstamps'][:]

    # now return dictionary, which should have everything we want to interpolate/save
    return load_dict


# -----------------------------------------------------------------------------------------------------------
def combine_fly_data(fly, var_dict=VAR_DICT, general_h5_str=GENERAL_H5_STR, signals_str=SIGNALS_STR,
                     daq_fields=DAQ_FIELDS, resample_freq=RESAMPLE_FREQ, resample_t=None,
                     interp_kind=INTERP_KIND, save_flag=True):
    """
    main function to load in data from different files, interpolate, and then save in new format
    """
    # ----------------------------------------------------------
    # get dictionary containing the various signals we want
    print('Loading data ...')
    loaddata = load_all_data(fly, var_dict=var_dict, general_h5_str=general_h5_str, signals_str=signals_str)
    print('Done loading')

    # ---------------------------------------------------------
    # get global time variable
    # Thad's code uses ca_cam signal zeroed at the first kinefly time stamp, so first get ca and kinefly cam tstamps
    muscle_t_keys = ['right_muscle_t', 'left_muscle_t']
    if resample_t is not None:
        # in this case, we don't need to worry about loading muscle camera clocks, so can skip
        pass
    elif all([(k in loaddata.keys()) for k in muscle_t_keys]):
        # get muscle timing for both sides, take the shorter of the two (so we don't extrapolate)
        muscle_t_list = [loaddata[key] for key in muscle_t_keys]
        mt_idx = np.argmin(np.array([np.max(mt) for mt in muscle_t_list]))
        muscle_t = np.array(muscle_t_list[mt_idx])
    elif any([(k in loaddata.keys()) for k in muscle_t_keys]):
        # read out key for muscle time signal on the side we have data for
        mt_key = [k for k in muscle_t_keys if k in loaddata.keys()][0]
        muscle_t = np.array(loaddata[mt_key])
    else:
        print('Error: could not find time stamps for any epi cameras')
        return

    # also get kinefly CAMERA tstamps
    kine_cam_t = np.array(loaddata['kine_cam_t'])
    t0 = kine_cam_t[0]  # value to subtract off of other times to keep things aligned

    # use muscle camera and kinefly camera time stamps to get global clock time (per Thad's code)
    # (unless we've provided it as input)
    if resample_t is None:
        muscle_t -= t0
        end = np.floor(muscle_t[-1])
        num_new_samples = end * resample_freq

        resample_t = np.linspace(0, end, num_new_samples)

    # ------------------------------------------------------------------------------------------------------------------
    # loop through variables and interpolate them to new, global time (diff data types will require diff cases)
    interp_dict = dict()  # initialize storage
    interp_dict['time'] = resample_t

    # loop over keys in loaddata dict to interpolate these variables
    data_keys = sorted(loaddata.keys())
    # data_keys.sort()  # sort keys to keep output organized?
    for key in data_keys:
        # if unicode, convert to string
        key = str(key)
        print(key)

        # deal with some of the special cases first
        if key.endswith('_t'):
            # most variables should have 'var' and 'var_t' fields -- skip the '*_t' version to avoid redundancy
            continue

        # deal with DAQ data
        if ('daq' in key) and not all([(dk in interp_dict.keys()) for dk in daq_fields.keys()]):
            # DAQ data requires a little special treatment -- first read out channels/values
            daq_channels = loaddata['daq_channels']
            daq_vals = loaddata['daq_vals']
            daq_t = loaddata['daq_t'] - t0

            # loop over fields (e.g. 'wb_freq', 'x_pos', etc)
            for dkey in daq_fields.keys():
                # load data
                daq_dat = load_daq_data(daq_channels, daq_vals, daq_fields[dkey])

                # interpolate from daq time to resampled time and add to dict
                interp_dict[dkey] = my_interp(daq_t, daq_dat, resample_t, interp_kind=interp_kind)
            continue

        elif ('daq' in key) and all([(dk in interp_dict.keys()) for dk in daq_fields.keys()]):
            # this means we've already done the DAQ data extraction, so just skip
            continue

        # load in data for all other cases
        if 'muscle' in key:
            # we just have one dict entry for the Ca signals from each muscle
            t = loaddata['%s_muscle_t'%(key.split('_')[0])].copy()
        else:
            # for other data types, each will have its own time
            t = loaddata[key + '_t'].copy()

        t -= t0   # NB: following Thad's code, I'm subtracting off intial kine cam time
        dat = loaddata[key]

        # special case when 'dat' is a string list/array (e.g. for exp_block and exp_state).
        # otherwise just do normal interp. NB: doing this separately due to version issues
        if sys.version_info[0] < 3:
            # python 2 case, where basestring is still valid
            is_string_flag = isinstance(dat[0], basestring)
        else:
            # python 3 case, where we can just look for string
            is_string_flag = isinstance(dat[0], str) | isinstance(dat[0], bytes)

        if is_string_flag:
            # first, check if we just have one entry -- if so just repeat it
            if len(dat) == 1:
                interp_dict[key] = np.tile(dat.astype('S56'), resample_t.size)
                continue

            # otherwise we'll have to do something fancier
            else:
                # ... convert from strings to index and then interpolate on that?
                dat_idx = np.arange(len(dat))
                try:
                    idx_interp = my_interp(t, dat_idx, resample_t, interp_kind=interp_kind)
                except ValueError as err:
                    print('failed to interpolate %s' %(key))
                    print(err)
                    continue
                interp_dict[key] = np.array([dat[int(ind)] for ind in idx_interp], dtype='S256')
                continue
        else:
            interp_dict[key] = my_interp(t, dat, resample_t, interp_kind=interp_kind)

    # ----------------------------------------------------------------------------------------
    # convert new interpolated data dict to pandas array, then save some versions
    # (will eventually pick just one option, but would like to have options for now)
    interp_df = pd.DataFrame.from_dict(interp_dict)

    if save_flag:
        fly_path, file_suffix = get_fly_path(fly)
        # save pandas
        save_name = 'combined_df' + file_suffix
        interp_df.to_hdf(os.path.join(fly_path, save_name + ".hdf5"), key='df')
        interp_df.to_pickle(os.path.join(fly_path, save_name + ".cpkl"))

        # save directly from dict
        dict_save_name = 'combined_dict' + file_suffix
        with h5py.File(os.path.join(fly_path, dict_save_name + ".hdf5"), 'w') as sf:
            for kkey in interp_dict.keys():
                sf.create_dataset(kkey, data=interp_dict[kkey][:])

    # return pandas dataframe? idk, just want to return something
    return interp_df


################################################################################################
########################### RUN SCRIPT #########################################################
################################################################################################
if __name__ == '__main__':
    # read in flies from terminal input or specified list
    if not FLIES:
        flies = [int(x) for x in sys.argv[1:]]
    else:
        flies = FLIES

    # loop through flies and do interpolation/saving
    for fly in flies:
        combine_fly_data(fly)

