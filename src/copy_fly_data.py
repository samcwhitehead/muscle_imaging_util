""" quick script to copy data from an external drive to PC without taking large bag files """
import os
import shutil
import re

# -------------------------------------
# source and destination directory
# -------------------------------------
SRC_DIR = os.path.normpath(r"F:\HingeImaging\ThadSetup\FlyDB")
DEST_DIR = os.path.normpath(r"C:\Users\samcw\Dropbox\Hinge_Imaging\Muscle_and_Tendon\FlyDB")

# which flies to copy over
fly_num_list = 'all'  # list(range(1,19))  # use 'all' if we should grab all

# folder string(s)
folder_str = 'Fly%04d'
folder_search = r"Fly(?P<fly_num>\d{4})"
folder_pattern = re.compile(folder_search)

# should we overwrite files?
OVERWRITE_FLAG = False

# suffices for filenames to exclude
EXCLUDE_SUFFIX_LIST = ['.bag']

# -------------------------------------
# MAIN FUNCTION
# -------------------------------------
if __name__ == '__main__':

    # figure out if we should grab all folders or just some
    if fly_num_list == 'all':
        # in this case, just get all
        src_folders = os.listdir(SRC_DIR)

        # initialize new list
        fly_num_list_new = []

        # loop through folders and get fly numbers
        for sf in src_folders:
            pat_match = folder_pattern.match(sf)
            if pat_match:
                fly_num_list_new.append(int(pat_match.group('fly_num')))

        # make this the new fly number list
        fly_num_list = fly_num_list_new

    # loop through fly numbers
    for fnum in fly_num_list:
        # get name for directory in "DEST_DIR" for current fly (as well as folder in src)
        dest_folder = os.path.normpath(os.path.join(DEST_DIR, folder_str%(fnum)))
        src_folder = os.path.normpath(os.path.join(SRC_DIR, folder_str % (fnum)))

        # first check if source directory exists (if not, skip)
        if not os.path.exists(src_folder):
            print('Cannot find folder ' + src_folder)
            continue

        # then check if dest directory already exists (if so, and we're no overwriting, skip)
        if os.path.exists(dest_folder) and not OVERWRITE_FLAG:
            print('Folder already exists for ' + dest_folder)
            continue

        # ...otherwise make new directory, start copying files over
        os.mkdir(dest_folder)

        # get files from SRC that we should copy over
        src_files = os.listdir(src_folder)
        for fname in src_files:
            # check current filename to make sure it doesn't end with the extension we want to include
            if any([fname.endswith(sufx) for sufx in EXCLUDE_SUFFIX_LIST]):
                # in this case, we don't like the suffix, so skip
                continue
            else:
                # in this case, copy the file
                src_curr_full = os.path.normpath(os.path.join(src_folder, fname))
                dest_curr_full = os.path.normpath(os.path.join(dest_folder, fname))

                shutil.copy2(src_curr_full, dest_curr_full)

                # print update
                print('Copied %s to %s'%(src_curr_full, dest_curr_full))