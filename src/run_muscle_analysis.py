"""
Code to wrap 'extract_muscle_signals.py' and 'combine_data_types.py,' thereby performing analyses on muscle data

"""
# -----------------------------------------------------------------
import os
import sys

from extract_muscle_signals import run_gcamp_extraction
from combine_data_types import combine_fly_data


################################################################################################
################################ GENERAL INPUTS ################################################
################################################################################################
FLY_DB_PATH = '/media/sam/SamData/FlyDB'  # directory containing fly data
# FLY_DB_PATH = '/media/imager/DataExternal/FlyDB'

FLIES = [59]  # None  # list of flies to analyze. If None, look for command line input
FOLDER_STR = 'Fly%04d'  # general format of folders containing fly data -- only used if FLIES contains ints
FN_STR = 'ca_camera'  # string to search for when looking for files containing image data
DRIVER = 'GMR22H05'

SAVE_FLAG = True  # save output?
OVERWRITE_FLAG = True  # overwrite gcamp extraction if we've done it already?

################################################################################################
########################### RUN SCRIPT #########################################################
################################################################################################
if __name__ == '__main__':
    # read in flies from terminal input or specified list
    if not FLIES:
        flies = [int(x) for x in sys.argv[1:]]
    else:
        flies = FLIES

    # loop through flies and do extraction then interpolation/saving
    for fly in flies:
        print('Fly: ', fly)
        # extract muscle signals
        run_gcamp_extraction(fly, fly_db_path=FLY_DB_PATH, fn_str=FN_STR, save_flag=SAVE_FLAG,
                             folder_str=FOLDER_STR, driver=DRIVER, overwrite_flag=OVERWRITE_FLAG)

        # create combined/interpolated data structure
        combine_fly_data(fly, save_flag=SAVE_FLAG)

