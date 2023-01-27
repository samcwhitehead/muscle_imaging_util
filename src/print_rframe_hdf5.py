"""
Temp script to print rframe hdf5 contents

"""
import os 
import sys
import h5py
import cPickle

FLYDB_PATH = '/media/sam/SamData/FlyDB'
FLIES = range(41, 54)  # None
SIDES = ['left', 'right']
GROUP_KEY = 'LogRefFrame'
NODE_NAME = 'unmixer'  # 'live_viewer'
SAVE_FLAG = True

def print_rframe(fly_num):
    fly_path = os.path.join(FLYDB_PATH, 'Fly%04d'%(fly_num))
    
    # loop over sides
    for side in SIDES:
        rframe_fn = '%s_%s_rframe_fits.hdf5'%(NODE_NAME, side)
        
        with h5py.File(os.path.join(fly_path, rframe_fn), 'r') as h5f:
            for key in h5f[GROUP_KEY].keys():
                print(key, ' = ', h5f[GROUP_KEY][key][()])
        
        if SAVE_FLAG:
            rframe_dict = dict()
            with h5py.File(os.path.join(fly_path, rframe_fn), 'r') as h5f:
                for key in h5f[GROUP_KEY].keys():
                    try:
                        rframe_dict[key] = h5f[GROUP_KEY][key][()][0]
                    except IndexError as err:
                        print('Error with reading %s, %s: %s' %(rframe_fn, key, err))
            
            save_path = os.path.join(fly_path, 'ca_camera_%s_rframe_fits.cpkl'%(side))
            with open(save_path, 'wb') as fp:
                print(save_path)
                cPickle.dump(rframe_dict, fp)
        
    
if __name__ == '__main__':
    # read in flies from terminal input or specified list
    if not FLIES:
        flies = [int(x) for x in sys.argv[1:]]
    else:
        flies = FLIES
    
    print(flies)
    
    for fly in flies:
	    print_rframe(fly)
