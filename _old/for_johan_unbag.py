#! /usr/bin/python

import sys
import cv2
import os
import cPickle
# coding: utf-8

# Updating notebook for richer bagfiles

def unbag_fly(fly_num):
    print fly_num
    resample_freq = 50 # a little bit faster than the Ca-cam will get wing aliasing

    fly_db_path = '/media/imager/DataExternal/FlyDB'
    fly_path = fly_db_path + '/' + 'Fly%04d'%fly_num
    bagfile = None

    if not(type(bagfile) is str):
        file_list = [f for f in os.listdir(fly_path) if '.bag' in f]
        if (type(bagfile) is int):
            bagfile_name = file_list[bagfile]
        else:
            bagfile_name = file_list[0]
    else:
        bagfile_name = bagfile
    bag_path = fly_path + '/' + bagfile_name

    import rosbag
    import numpy as np
    inbag = rosbag.Bag(bag_path)

    #get a list of topics in the bag file
    topics = inbag.get_type_and_topic_info()[1].keys()
    types = []
    for i in range(0,len(inbag.get_type_and_topic_info()[1].values())):
        types.append(inbag.get_type_and_topic_info()[1].values()[i][0])

#    #save the metadata
#    metadata_msgs =  [(topic,msg,t) for topic,msg,t in inbag.read_messages(topics = '/exp_scripts/exp_metadata')]
#    
#    mtd = cPickle.loads(str(metadata_msgs[0][1].data))
#    git_sha = mtd['git_SHA']#mtd.split(';')[0].split('=')[1].lstrip()
#    exp_description = mtd['exp_description']#mtd.split(';')[2].split('=')[1].lstrip()
#    fly_genotype = mtd['fly_genotype']#"""w[1118]/+[HCS];P{y[+t7.7] w[+mC]=GMR22H05-GAL4}attP2/P{20XUAS-IVS-GCaMP6f}attP40;+/+"""
#    genotype_nickname = mtd['genotype_nickname']#22H05-GCaMP6f'
#    script_code = mtd['script_code']

#    with open(os.path.join(fly_path,'git_SHA.txt'),'wt') as f:
#        f.write(git_sha)
#    sp = os.path.split(mtd['script_path'])[-1]
#    with open(os.path.join(fly_path,sp),'wt') as f:
#        f.write(script_code)
#    with open(os.path.join(fly_path,'exp_description.txt'),'wt') as f:
#        f.write(exp_description)
#    with open(os.path.join(fly_path,'fly_genotype.txt'),'wt') as f:
#        f.write(fly_genotype)
#    with open(os.path.join(fly_path,'genotype_nickname.txt'),'wt') as f:
#        f.write(genotype_nickname)

    print 'loading messages'
    img_msgs = [(topic,msg,t) for topic,msg,t in inbag.read_messages(topics = ['/arena_camera/image_color/compressed',
                                                                               '/kine_camera_1/image_raw/compressed',
                                                                               '/ca_camera_left/image_raw',
                                                                               '/ca_camera_right/image_raw'])]
    condition_msgs = [(topic,msg,t) for topic,msg,t in inbag.read_messages(topics = ['/exp_scripts/exp_state'])]
    block_msgs = [(topic,msg,t) for topic,msg,t in inbag.read_messages(topics = ['/exp_scripts/exp_block'])]
    daq_msgs = [(topic,msg,t) for topic,msg,t in inbag.read_messages(topics = ['/phidgets_daq/all_data'])]
    kinefly_msgs = [(topic,msg,t) for topic,msg,t in inbag.read_messages(topics = ['/kinefly/flystate'])]

    kfly_tstamps = [km[1].header.stamp.to_time() for km in kinefly_msgs]
    left = [km[1].left.angles for km in kinefly_msgs]
    right = [km[1].right.angles for km in kinefly_msgs]
    f = lambda x: {0:(np.nan,),1:x}[len(x)]
    right = map(f,right)
    left = map(f,left)

    freq_idx = daq_msgs[0][1].channels.index('freq')
    x_idx = daq_msgs[0][1].channels.index('xpos')
    y_idx = daq_msgs[0][1].channels.index('ypos')

    daq_tstamps = [dm[2].to_time() for dm in daq_msgs]
    freq = [dm[1].values[freq_idx] for dm in daq_msgs]
    x_pos = [dm[1].values[x_idx] for dm in daq_msgs]
    y_pos = [dm[1].values[y_idx] for dm in daq_msgs]


    def fix_tstamps(tstamps):
        """this function fixes the timestamps from the cameras 
        since we know that they are coming in at a constant clock rate"""
        from numpy import linalg
        num_frames = len(tstamps)
        a = np.vstack([np.arange(num_frames),np.ones(num_frames)]).T
        b = np.array(tstamps)
        slope,intercept = linalg.lstsq(a,b)[0]
        fixed_tstamps = slope*a[:,0] + intercept
        return fixed_tstamps

    from cv_bridge import CvBridge, CvBridgeError
    cv_bridge = CvBridge()

    arena_cam1_tstamps = list()
    arena_cam1_ros_tstamps = list()
    arena_cam1_imgs = list()
    arena_jpeg_len = list()

    kine_cam1_tstamps = list()
    kine_cam1_ros_tstamps = list()
    kine_cam1_imgs = list()

    ca_cam_right_tstamps = list()
    ca_cam_right_ros_tstamps = list()
    ca_cam_right_imgs = list()

    ca_cam_left_tstamps = list()
    ca_cam_left_ros_tstamps = list()
    ca_cam_left_imgs = list()

    print 'loading images'
    for i in range(len(img_msgs)):
        #if img_msgs[i][0] == '/arena_camera/image_color/compressed':
        #    #the ros tstamp is when the msg was saved
        #    arena_cam1_ros_tstamps.append(float(img_msgs[i][2].to_time()))
        #    #the header tstamp is when the msg was generated
        #    arena_cam1_tstamps.append(float(img_msgs[i][1].header.stamp.to_time()))
        #    img_np_arr = np.fromstring(img_msgs[i][1].data, np.uint8)
        #    #encoded_img = cv2.imdecode(img_np_arr,cv2.IMREAD_COLOR)
        #    arena_cam1_imgs.append(img_np_arr)
        #    arena_jpeg_len.append(len(img_np_arr))

        if img_msgs[i][0] == '/kine_camera_1/image_raw/compressed':
            #the ros tstamp is when the msg was saved
            kine_cam1_ros_tstamps.append(float(img_msgs[i][2].to_time()))
            #the header tstamp is when the msg was generated
            kine_cam1_tstamps.append(float(img_msgs[i][1].header.stamp.to_time()))
            img_np_arr = np.fromstring(img_msgs[i][1].data, np.uint8)
            encoded_img = cv2.imdecode(img_np_arr,cv2.IMREAD_GRAYSCALE)
            kine_cam1_imgs.append(encoded_img)

        if img_msgs[i][0] == '/ca_camera_right/image_raw':
            ca_cam_right_ros_tstamps.append(float(img_msgs[i][2].to_time()))
            ca_cam_right_tstamps.append(float(img_msgs[i][1].header.stamp.to_time()))
            ca_cam_right_imgs.append(cv_bridge.imgmsg_to_cv2(img_msgs[i][1]))
        if img_msgs[i][0] == '/ca_camera_left/image_raw':
            ca_cam_left_ros_tstamps.append(float(img_msgs[i][2].to_time()))
            ca_cam_left_tstamps.append(float(img_msgs[i][1].header.stamp.to_time()))
            ca_cam_left_imgs.append(cv_bridge.imgmsg_to_cv2(img_msgs[i][1]))
    #we can't assume a constant clock now, but we should be able to
    #align the three streams to one another.
    #ca_cam1_tstamps = fix_tstamps(ca_cam1_tstamps)
    
    #kine_cam1_tstamps = fix_tstamps(kine_cam1_tstamps)

    #we should throw out the kine timestamps that occur before the first ca timestamp
    criterion =  np.argwhere(kine_cam1_tstamps < ca_cam_right_tstamps[0])

    if len(criterion) > 0:
        criterion = criterion[0]
    if len(criterion)>0:
        first_kine_frame = criterion[-1]
        kine_cam1_imgs = kine_cam1_imgs[first_kine_frame:]
        kine_cam1_tstamps = kine_cam1_tstamps[first_kine_frame:]

    #make t=0 the time of the first kine_cam_frame
    first_ca_tstamp_ros = ca_cam_right_ros_tstamps[0]
    first_kine_tstamp_ros = kine_cam1_ros_tstamps[0]
    #first_arena_tstamp_ros = arena_cam1_ros_tstamps[0]

    #arena_cam1_tstamps = np.array(arena_cam1_tstamps)
    #arena_cam1_tstamps -= first_arena_tstamp_ros

    kine_cam1_tstamps = np.array(kine_cam1_tstamps)
    kine_cam1_tstamps -= first_kine_tstamp_ros

    ca_cam_right_tstamps = np.array(ca_cam_right_tstamps)
    ca_cam_right_tstamps -= first_kine_tstamp_ros

    kfly_tstamps = np.array(kfly_tstamps)
    kfly_tstamps -= first_kine_tstamp_ros

    daq_tstamps = np.array(daq_tstamps)
    daq_tstamps -= first_kine_tstamp_ros

    cond_tstamps = np.array([cm[2].to_time() for cm in condition_msgs])
    cond_tstamps -= first_kine_tstamp_ros

    block_tstamps = np.array([bm[2].to_time() for bm in block_msgs])
    block_tstamps -= first_kine_tstamp_ros

    last_ca_tstamp_exp = ca_cam_right_tstamps[-1]

    #arena_jpeg_max_len = np.max(np.array(arena_jpeg_len))

    def choose_last(idx_array):
        if np.shape(idx_array)[0]==0:
            return 0
        else:
            return np.squeeze(idx_array[-1])
        
    def get_data(t):
        """ return the closest images to time t"""
        ca_cam_idx = choose_last(np.argwhere(ca_cam_right_tstamps<t))
        #arena_cam1_idx = choose_last(np.argwhere(arena_cam1_tstamps<t))
        kine_cam1_idx = choose_last(np.argwhere(kine_cam1_tstamps<t))
        kfly_idx = choose_last(np.argwhere(kfly_tstamps<t))
        daq_idx = choose_last(np.argwhere(daq_tstamps<t))
        cond_idx = choose_last(np.argwhere(cond_tstamps<t))
        block_idx = choose_last(np.argwhere(block_tstamps<t))
        #get the jpeg array
        #ar_image = arena_cam1_imgs[arena_cam1_idx]
        #arena_array[:len(ar_image)] = ar_image
        return {'ca_cam_right':ca_cam_right_imgs[ca_cam_idx],
                'ca_cam_left':ca_cam_left_imgs[ca_cam_idx],
                'kine_cam_1':kine_cam1_imgs[kine_cam1_idx],
                'left_amp':np.array(left[kfly_idx]),
                'right_amp':np.array(right[kfly_idx]),
                'wb_freq':np.array([freq[daq_idx],]),
                'x_pos':np.array([x_pos[daq_idx],]),
                'y_pos':np.array([y_pos[daq_idx],]),
                'experimental_condition':np.array(condition_msgs[cond_idx][1].data,dtype = 'S256'),
                'experimental_block':np.array(block_msgs[block_idx][1].data,dtype = 'S256')}

        #return {'ca_cam_right':ca_cam_right_imgs[ca_cam_idx],
        #        'ca_cam_left':ca_cam_left_imgs[ca_cam_idx],
        #        'kine_cam_1':kine_cam1_imgs[kine_cam1_idx],
        #        'arena_cam_1':arena_array,
        #        'left_amp':np.array(left[kfly_idx]),
        #        'right_amp':np.array(right[kfly_idx]),
        #        'wb_freq':np.array([freq[daq_idx],]),
        #        'x_pos':np.array([x_pos[daq_idx],]),
        #        'y_pos':np.array([y_pos[daq_idx],]),
        #        'experimental_condition':np.array(condition_msgs[cond_idx][1].data,dtype = 'S256'),
        #        'experimental_block':np.array(block_msgs[block_idx][1].data,dtype = 'S256')}

    data_structure = get_data(0)

    #resample array
    end = np.floor(last_ca_tstamp_exp)

    num_new_samples = end*resample_freq
    resample_array = np.linspace(0,end,num_new_samples)

    unbag_dir = fly_path

    import h5py

    h5file_dict = dict()

    for key in data_structure.keys():
        fpath = unbag_dir + '/' + key + '.hdf5'
        if os.path.exists(fpath):
            os.remove(fpath)
        h5file_dict.update({key:h5py.File(unbag_dir + '/_' + key + '.hdf5','a')})

    for key,value in h5file_dict.items():
        dsetshape = list(data_structure[key].shape)
        dsetshape = [int(num_new_samples)] + dsetshape
        dsettype = data_structure[key].dtype
        value.create_dataset(name = key,
                             shape = dsetshape,
                             dtype = dsettype)

    # fill the dataset
    print 'filling dataset'
    import time
    stime = time.time()
    for i,t in enumerate(resample_array):
        datum = get_data(t)
        for key,value in h5file_dict.items():
            value[key][i] = datum[key]
    print time.time()-stime

    for value in h5file_dict.values():
        value.close()

    # compress the data
    for key in data_structure.keys():
        print 'compressing: %s'%key
        fpath = unbag_dir + '/' + key + '.hdf5'
        infile = h5py.File(unbag_dir + '/_' + key + '.hdf5','a')
        outfile = h5py.File(unbag_dir + '/' + key + '.hdf5','a')
        outfile.create_dataset(key,data = infile[key],compression = 'gzip')
        infile.close()
        os.remove(unbag_dir + '/_' + key + '.hdf5')

    tfile = h5py.File(unbag_dir + '/' + 'time' + '.hdf5','a')
    tfile.create_dataset('time',data = resample_array,compression = 'gzip')
    tfile.close()

#    import datetime
#    import yaml
#    info_dict = yaml.load(inbag._get_yaml_info())
#    dtm = datetime.datetime.fromtimestamp(info_dict['start'])
#    m,d,y = mtd['fly_dob'].split('.')##'8.13.2017'.split('.')
#    dob_dt = datetime.datetime(year = int(y),month = int(m),day = int(d))
#    delta_dt = dtm-dob_dt
#    with open(os.path.join(fly_path,'age.txt'),'wt') as f:
#        f.write(str(delta_dt))

if __name__ == '__main__':
    flies = [int(x) for x in sys.argv[1:]]
    #from multiprocessing import Pool
    #p = Pool(3)
    #unbag_fly(flies[0])
    #rint(p.map(unbag_fly,flies))
    # for fly in flies:
    #     try:
    #         unbag_fly(fly)
    #     except Exception as er:
    #         print 'error with fly %s'%(fly)
    #     #[unbag_fly(fly) for fly in flies]
    for fly in flies:
        unbag_fly(fly)
        
# unbag_fly(19)
#if __name__ == '__main__':
#    flies = [int(x) for x in sys.argv[1:]]
    #from multiprocessing import Pool
    #p = Pool(3)
    #unbag_fly(flies[0])
    #rint(p.map(unbag_fly,flies))
    # for fly in flies:
    #     try:
    #         unbag_fly(fly)
    #     except Exception as er:
    #         print 'error with fly %s'%(fly)
    #     #[unbag_fly(fly) for fly in flies]
#    for fly in flies:
#        unbag_fly(fly)
        #except Exception as er:
        #    print 'error with fly %s'%(fly)
        #[unbag_fly(fly) for fly in flies]
