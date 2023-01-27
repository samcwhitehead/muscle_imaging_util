#! /usr/bin/python
"""
Code to take the raw bag files output in Thad/Alyshas's rig, extract relevant data, and save to external source

Adapted from Thad's "unbag.py" and Francesca's "bag2hdf5.py"

Usage depends on structure of fly data directory.
    - If each .bag file is stored in a folder of the form 'Fly0000', then set the variable fly_db_path to be the
    directory containing each of these folders and input fly ids as integers (Sam). So this looks like:
        > flies = [2]
        > myBagToHDF5 = BagToHDF5(fly_db_path='/media/sam/SamData/FlyDB')
        > for fly in flies:
        >     myBagToHDF5.set_fly_num(fly)
        >     myBagToHDF5.do_conversion_all()
        > myBagToHDF5.close_files()

    - If .bag files are stored separately, or with different directory format, set fly_db_path to be the full folder
    to the data, and self.fly to be the .bag filename (Francesca). This would look like
        > flies = ['MI_2022-08-11-16-44-33.bag', ...]
        > myBagToHDF5 = BagToHDF5(fly_db_path=[PATH TO DATA])
        > for fly in flies:
        >     myBagToHDF5.set_fly_fn(fly)
        >     myBagToHDF5.do_conversion_all()
        > myBagToHDF5.close_files()

TO DO:
    - left and right reference frame distinction
    - compression for image files (using gzip right now, but make sure that's okay)
    - time alignment of video frames

"""
import cPickle
import cv2
import h5py
import os
import rosbag
import sys

import numpy as np

# coding: utf-8

################################################################################################
########################### GENERAL CONSTANTS ##################################################
################################################################################################
# FLY_DB_PATH = '/media/imager/DataExternal/FlyDB'
FLY_DB_PATH = '/media/sam/SamData/FlyDB'
FLIES = None  # list of flies to analyze. If None, look for command line input
CHUNK_SIZE = 100  # how large of an image stack to save as a chunk inside hdf5 (only for cam data)

################################################################################################
########################### DEFINE CONVERTER CLASS #############################################
################################################################################################
class BagToHDF5:
    # ----------------------------------------------------------------------
    def __init__(self, fly_db_path=FLY_DB_PATH, fly_path=None, chunk_size=CHUNK_SIZE):
        """
        intialization function
        """
        # read in vars
        self.fly_db_path = fly_db_path

        # set placeholders for current bag/hdf5 files
        self.h5f = None
        self.inbag = None
        self.fly_id = None  # this can either be an integer (assuming Thad/Alysha's storage) or a filename
        self.fly_id_str = None  # just a handy string reference to call for print statements etc
        self.fly_path = fly_path  # path to folder containing .bag file
        self.bagfile_name = None  # filename for .bag *** not including ext!

        # other misc. vars that we might want to edit
        self.fly_folder_str = 'Fly%04d'
        self.chunk_size = chunk_size

        # ROS topics that we'll read from
        self.kinefly_topic_names = ['/kinefly/flystate']
        self.led_panel_topic_names = ['/ledpanels/command']
        self.daq_phidget_topic_names = ['/phidgets_daq/all_data']
        self.cam_topic_names = ['/ca_camera_left/image_raw',
                                '/ca_camera_right/image_raw',
                                '/kine_camera_1/image_raw/compressed']
        self.exp_msg_topic_names = ['/exp_scripts/exp_state',
                                    '/exp_scripts/exp_block']
        self.metadata_topic_names = ['/exp_scripts/exp_metadata']
        self.rframe_topic_names = ['/exp_scripts/right_RefFrameServer',
                                   '/exp_scripts/left_RefFrameServer']
        viewer_name = 'live_viewer'  # 'unmixer' or 'live_viewer'
        self.unmixer_topic_names = ['_left/b1',
                                    '_left/b2',
                                    '_left/b3',
                                    '_left/bkg',
                                    '_left/hg1',
                                    '_left/hg2',
                                    '_left/hg3',
                                    '_left/hg4',
                                    '_left/i1',
                                    '_left/i2',
                                    '_left/iii1',
                                    '_left/iii24',
                                    '_left/iii3',
                                    '_left/nm',
                                    '_left/pr',
                                    '_left/tpd',
                                    '_right/b1',
                                    '_right/b2',
                                    '_right/b3',
                                    '_right/bkg',
                                    '_right/hg1',
                                    '_right/hg2',
                                    '_right/hg3',
                                    '_right/hg4',
                                    '_right/i1',
                                    '_right/i2',
                                    '_right/iii1',
                                    '_right/iii24',
                                    '_right/iii3',
                                    '_right/nm',
                                    '_right/pr',
                                    '_right/tpd']
        self.unmixer_topic_names = [('/' + viewer_name + tn) for tn in self.unmixer_topic_names]

    # ----------------------------------------------------------------------
    def set_fly_num(self, fly_num):
        """
        function to set the fly identifier as fly number we're focusing on
        NB: this is the function we would use for data stored in Thad/Alysha's format
        """
        # write current fly number to self
        self.fly_id = fly_num
        self.fly_id_str = 'Fly%04d' %(fly_num)
        
        # use fly number to define the folder where the .bag file should be stored
        self.fly_path = os.path.normpath(os.path.join(self.fly_db_path, self.fly_folder_str % fly_num))
        
        # get corresponding bag and hdf5 filenames
        self.get_file_paths()

    # ----------------------------------------------------------------------
    def set_fly_fn(self, fn):
        """
        function to set the fly identifier as the filename for the .bag file we're focusing on
        NB: this is the function we would use for data stored NOT in Thad/Alysha's format
        """
        # write current fly number to self
        self.fly_id = fn
        self.fly_id_str = fn

        # use fly number to define the folder where the .bag file should be stored
        self.fly_path = os.path.normpath(self.fly_db_path)

        # get corresponding bag and hdf5 filenames
        self.get_file_paths(bagfile=self.fly_id)
        
    # ----------------------------------------------------------------------
    def get_file_paths(self, bagfile=None):
        """
        function to set the input file for the .bag file we're focusing on
        NB: this is the function we would use for data stored NOT in Thad/Alysha's format
        """
        # make sure that self.fly_path (where bag file should be located) exists
        if not os.path.exists(self.fly_path):
            self.fly_id = None
            self.fly_path = None
            self.fly_id_str = None
            print('Could not locate data for %s -- quitting' % (self.fly_id_str))
            return

        # taken from Thad's code -- I guess just being thorough about getting the right filename?
        if not (type(bagfile) is str):
            file_list = [f for f in os.listdir(self.fly_path) if f.endswith('.bag')]
            if (type(bagfile) is int):
                bagfile_name = file_list[bagfile]
            else:
                bagfile_name = file_list[0]
        else:
            bagfile_name = bagfile
        
        # make sure we have file extension (ideally we can enter in filename without ext)
        if not bagfile_name.endswith('.bag'):
            bagfile_name += '.bag'
            
        # load bag file
        bag_path_full = os.path.normpath(os.path.join(self.fly_path, bagfile_name))
        if not os.path.exists(bag_path_full):
            self.fly_id = None
            self.fly_path = None
            self.fly_id_str = None
            print('Error: could not locate file %s -- quitting' %(bag_path_full))
            return
        self.inbag = rosbag.Bag(bag_path_full)

        # also store .bag filename
        self.bagfile_name = bagfile_name[:-4]

        # get corresponding hdf5 path
        if isinstance(self.fly_id, int):
            file_suffix = ''
        else:
            file_suffix = '_' + self.bagfile_name

        h5_fn = 'converted' + file_suffix + '.hdf5'
        h5_path = os.path.normpath(os.path.join(self.fly_path, h5_fn))
        self.h5f = h5_path

        print('Now converting %s' % self.fly_id_str)

    # ----------------------------------------------------------------------
    def close_files(self):
        """
        function to close current file(s)
        """
        # write over all stored file info
        self.fly_id = None
        self.fly_id_str = None
        self.fly_path = None
        self.bagfile_name = None

        # close files
        self.inbag.close()
        # self.h5f.close()

    # ----------------------------------------------------------------------
    def read_kinefly_data(self, topic_name):
        """
        read Kinefly data

        """
        # make sure we have a valid fly number entered
        if not self.fly_id:
            print('No valid bag file selected -- stopping')
            return

        # read various Kinefly messages from bag file
        tstamps = [msg[1].header.stamp.to_time() for msg in self.inbag.read_messages(topics=topic_name)
                   if len(msg[1].left.angles) > 0 and len(msg[1].right.angles) > 0]
        seq = [msg[1].header.seq for msg in self.inbag.read_messages(topics=topic_name)
               if len(msg[1].left.angles) > 0 and len(msg[1].right.angles) > 0]
        langles = [msg[1].left.angles[0] for msg in self.inbag.read_messages(topics=topic_name)
                   if len(msg[1].left.angles) > 0 and len(msg[1].right.angles) > 0]
        lgradients = [msg[1].left.gradients[0] for msg in self.inbag.read_messages(topics=topic_name)
                      if len(msg[1].left.angles) > 0 and len(msg[1].right.angles) > 0]
        lintensity = [msg[1].left.intensity for msg in self.inbag.read_messages(topics=topic_name)
                      if len(msg[1].left.angles) > 0 and len(msg[1].right.angles) > 0]
        rangles = [msg[1].right.angles[0] for msg in self.inbag.read_messages(topics=topic_name)
                   if len(msg[1].left.angles) > 0 and len(msg[1].right.angles) > 0]
        rgradients = [msg[1].right.gradients[0] for msg in self.inbag.read_messages(topics=topic_name)
                      if len(msg[1].left.angles) > 0 and len(msg[1].right.angles) > 0]
        rintensity = [msg[1].right.intensity for msg in self.inbag.read_messages(topics=topic_name)
                      if len(msg[1].left.angles) > 0 and len(msg[1].right.angles) > 0]
        # count=        [msg[1].count                  for msg in self.inbag.read_messages(topics=topic_name) if len(msg[1].left.angles)>0 and len(msg[1].right.angles)>0]
        try:
            hangles = [msg[1].head.angles[0] for msg in self.inbag.read_messages(topics=topic_name)
                       if len(msg[1].left.angles) > 0 and len(msg[1].right.angles) > 0]
        except:
            pass

        # write bag file messages to hdf5
        print('writing %s data for %s' % (topic_name, self.fly_id_str))
        with h5py.File(self.h5f, 'a') as h5f:
            h5f.create_dataset('flystate_tstamps', data=tstamps)
            h5f.create_dataset('flystate_seq', data=seq)
            h5f.create_dataset('flystate_langles', data=langles)
            h5f.create_dataset('flystate_lgrandients', data=lgradients)
            h5f.create_dataset('flystate_lintensity', data=lintensity)
            h5f.create_dataset('flystate_rangles', data=rangles)
            h5f.create_dataset('flystate_rgradients', data=rgradients)
            h5f.create_dataset('flystate_rintensity', data=rintensity)
            # h5f.create_dataset('flystate_count',data=count)
            try:
                h5f.create_dataset('flystate_hangles', data=hangles)
            except:
                pass

    # ----------------------------------------------------------------------
    def read_ledpanels_data(self, topic_name):
        """
        read led panel data

        """
        # make sure we have a valid fly number entered
        if not self.fly_id:
            print('No valid bag file selected -- stopping')
            return

        # read various LED panelmessages from bag file
        ros_tstamps = [msg[2].to_time() for msg in self.inbag.read_messages(topics=topic_name)]
        panels_command = [msg[1].command for msg in self.inbag.read_messages(topics=topic_name)]
        panels_arg1 = [msg[1].arg1 for msg in self.inbag.read_messages(topics=topic_name)]
        panels_arg2 = [msg[1].arg2 for msg in self.inbag.read_messages(topics=topic_name)]
        panels_arg3 = [msg[1].arg3 for msg in self.inbag.read_messages(topics=topic_name)]
        panels_arg4 = [msg[1].arg4 for msg in self.inbag.read_messages(topics=topic_name)]
        panels_arg5 = [msg[1].arg5 for msg in self.inbag.read_messages(topics=topic_name)]
        panels_arg6 = [msg[1].arg6 for msg in self.inbag.read_messages(topics=topic_name)]

        # write LED panel data to hdf5
        print('writing %s data for %s' % (topic_name, self.fly_id_str))
        with h5py.File(self.h5f, 'a') as h5f:
            h5f.create_dataset('ledpanels_ros_tstamps', data=ros_tstamps)
            h5f.create_dataset('ledpanels_panels_command', data=panels_command)
            h5f.create_dataset('ledpanels_panels_arg1', data=panels_arg1)
            h5f.create_dataset('ledpanels_panels_arg2', data=panels_arg2)
            h5f.create_dataset('ledpanels_panels_arg3', data=panels_arg3)
            h5f.create_dataset('ledpanels_panels_arg4', data=panels_arg4)
            h5f.create_dataset('ledpanels_panels_arg5', data=panels_arg5)
            h5f.create_dataset('ledpanels_panels_arg6', data=panels_arg6)

    # ----------------------------------------------------------------------
    def read_muscle_unmixer_data(self, topic_name):
        """
        read muscle unmixer data

        """
        # make sure we have a valid fly number entered
        if not self.fly_id:
            print('No valid bag file selected -- stopping')
            return

        # read a given muscle unmixer message from bag file
        ros_tstamps = [msg[2].to_time() for msg in self.inbag.read_messages(topics=topic_name)]
        tstamps = [msg[1].header.stamp.to_time() for msg in self.inbag.read_messages(topics=topic_name)]
        seq = [msg[1].header.seq for msg in self.inbag.read_messages(topics=topic_name)]
        value = [msg[1].value for msg in self.inbag.read_messages(topics=topic_name)]
        muscle = [msg[1].muscle for msg in self.inbag.read_messages(topics=topic_name)]

        # write muscle messages to hdf5
        print('writing %s data for %s' % (topic_name, self.fly_id_str))
        with h5py.File(self.h5f, 'a') as h5f:
            h5f.create_dataset(str(topic_name[1:]) + '/muscleunmixer_ros_tstamps', data=ros_tstamps)
            h5f.create_dataset(str(topic_name[1:]) + '/muscleunmixer_tstamps', data=tstamps)
            h5f.create_dataset(str(topic_name[1:]) + '/muscleunmixer_seq', data=seq)
            h5f.create_dataset(str(topic_name[1:]) + '/muscleunmixer_value', data=value)
            h5f.create_dataset(str(topic_name[1:]) + '/muscleunmixer_muscle', data=muscle)

    # ----------------------------------------------------------------------
    def read_muscle_unmixer_data_all(self):
        """
        read ALL muscle unmixer data

        """
        for topic_name in self.unmixer_topic_names:
            self.read_muscle_unmixer_data(topic_name)

    # ----------------------------------------------------------------------
    def read_daq_phidget_data(self, topic_name):
        """
        read daq phidget data

        """
        # make sure we have a valid fly number entered
        if not self.fly_id:
            print('No valid bag file selected -- stopping')
            return

        # read DAQ messages
        ros_tstamps = [msg[2].to_time() for msg in self.inbag.read_messages(topics=topic_name)]
        time = [msg[1].time for msg in self.inbag.read_messages(topics=topic_name)]
        channels = [msg[1].channels for msg in self.inbag.read_messages(topics=topic_name)]
        values = [msg[1].values for msg in self.inbag.read_messages(topics=topic_name)]

        # write DAQ messages to hdf5
        print('writing %s data for %s' % (topic_name, self.fly_id_str))
        with h5py.File(self.h5f, 'a') as h5f:
            h5f.create_dataset('daq_ros_tstamps', data=ros_tstamps)
            h5f.create_dataset('daq_time', data=time)
            h5f.create_dataset('daq_channels', data=channels)
            h5f.create_dataset('daq_value', data=values)
    
    # ----------------------------------------------------------------------
    def read_general_msg_data(self, topic_name):
        """
        read messages like exp_state and exp_block

        """
        # make sure we have a valid fly number entered
        if not self.fly_id:
            print('No valid bag file selected -- stopping')
            return

        # read string messages
        ros_tstamps = [msg[2].to_time() for msg in self.inbag.read_messages(topics=topic_name)]
        msg_data = [msg[1].data for msg in self.inbag.read_messages(topics=topic_name)]

        # write general messages to hdf5
        # NB: name is based on SECOND element of topic name
        print('writing %s data for %s' % (topic_name, self.fly_id_str))
        with h5py.File(self.h5f, 'a') as h5f:
            h5f.create_dataset(str(topic_name.split('/')[-1]) + '/ros_tstamps', data=ros_tstamps)
            h5f.create_dataset(str(topic_name.split('/')[-1]) + '/msgs', data=msg_data)
            # h5f.create_dataset(str(topic_name[1:]) + '/msgs', data=np.array(msg_data,dtype = 'S256'))

    # ----------------------------------------------------------------------
    def read_metadata(self, topic_name='/exp_scripts/exp_metadata'):
        """
        read experiment metadata. NB: this should be a little different than the above
         NB: currently writing to txt files but should probably add this info to hdf5 file as well...

        """
        # make sure we have a valid fly number entered
        if not self.fly_id:
            print('No valid bag file selected -- stopping')
            return

        # get file suffix depending on whether or not we'd like to keep the bag file datestr (assuming that if we're
        # Thad/Alysha fly number convention, then we don't want this
        if isinstance(self.fly_id, int):
            file_suffix = ''
        else:
            file_suffix = '_' + self.bagfile_name

        # read metadata messages
        metadata_msgs = [(topic, msg, t) for topic, msg, t in self.inbag.read_messages(topics=topic_name)]
        if len(metadata_msgs) < 1:
            print('No meta data saved -- skipping')
            return
        
        # unpickle metadata and read out entries
        mtd = cPickle.loads(str(metadata_msgs[0][1].data))
        mtd_keys = ['git_SHA', 'exp_description', 'fly_genotype', 'fly_genotype', 'genotype_nickname']

        # write entries to text files
        print('writing %s data for %s' % (topic_name, self.fly_id_str))
        for key in mtd_keys:
            try:
                with open(os.path.join(self.fly_path, '%s%s.txt' % (key, file_suffix)), 'wt') as f:
                    f.write(mtd[key])
            except KeyError:
                print('Error: no %s meta data for %s' %(key, self.fly_id_str))

        # copy script code
        sp = os.path.split(mtd['script_path'])[-1]
        with open(os.path.join(self.fly_path, sp), 'wt') as f:
            f.write(mtd['script_code'])

        # --------------------------------
        # also get fly DOB
        import datetime
        import yaml
        info_dict = yaml.load(self.inbag._get_yaml_info(), yaml.Loader)
        dtm = datetime.datetime.fromtimestamp(info_dict['start'])
        m, d, y = mtd['fly_dob'].split('.')  ##'8.13.2017'.split('.')
        dob_dt = datetime.datetime(year=int(y), month=int(m), day=int(d))
        delta_dt = dtm - dob_dt

        with open(os.path.join(self.fly_path, 'age%s.txt' % (file_suffix)), 'wt') as f:
            f.write(str(delta_dt))

    # --------------------------------------------------------------------------------
    def read_cam_data(self, topic_name):
        """
        read cam images.
        NB: these are saved to separate hdf5 files, as in Thad's code

        """
        # make sure we have a valid fly number entered
        if not self.fly_id:
            print('No valid bag file selected -- stopping')
            return

        # get file suffix depending on whether or not we'd like to keep the bag file datestr (assuming that if we're
        # Thad/Alysha fly number convention, then we don't want this
        if isinstance(self.fly_id, int):
            file_suffix = ''
        else:
            file_suffix = '_' + self.bagfile_name

        # read camera messages
        ros_tstamps = [msg[2].to_time() for msg in self.inbag.read_messages(topics=topic_name)]
        tstamps = [msg[1].header.stamp.to_time() for msg in self.inbag.read_messages(topics=topic_name)]
        img_msgs = [msg[1] for msg in self.inbag.read_messages(topics=topic_name)]

        # convert images (the method for ROS msg -> cv2 image depends on the camera)
        if "compressed" in topic_name:
            imgs = [cv2.imdecode(np.fromstring(imm.data, np.uint8), cv2.IMREAD_GRAYSCALE) for imm in img_msgs]

        else:
            from cv_bridge import CvBridge
            cv_bridge = CvBridge()
            imgs = [cv_bridge.imgmsg_to_cv2(imm) for imm in img_msgs]

        # TEST -- get tuple corresponding to how we should save image data to hdf5
        im_h, im_w = imgs[0].shape[:2]
        chunks = (self.chunk_size, im_h, im_w)

        # write everything to SEPARATE hdf5
        save_name = [s for s in topic_name.split('/') if not s == ""][0] + file_suffix + '.hdf5'
        save_path = os.path.normpath(os.path.join(self.fly_path, save_name))

        print('writing %s data for %s' % (save_name, self.fly_id_str))
        with h5py.File(save_path, 'a') as h5f:
            h5f.create_dataset('cam_ros_tstamps', data=ros_tstamps)
            h5f.create_dataset('cam_tstamps', data=tstamps)
            h5f.create_dataset('cam_imgs', data=np.asarray(imgs), compression="gzip", chunks=chunks)

    # --------------------------------------------------------------------------------
    def read_rframe_data(self, topic_name):
        """
        read reference frames that are hopefully logged from experiments

        """
        # make sure we have a valid fly number entered
        if not self.fly_id:
            print('No valid bag file selected -- stopping')
            return

        # get file suffix depending on whether or not we'd like to keep the bag file datestr (assuming that if we're
        # Thad/Alysha fly number convention, then we don't want this
        if isinstance(self.fly_id, int):
            file_suffix = ''
        else:
            file_suffix = '_' + self.bagfile_name

        # read various reference frame messages from bag file
        ros_tstamps = [msg[2].to_time() for msg in self.inbag.read_messages(topics=topic_name)]
        ref_frame_components = [msg[1].components for msg in self.inbag.read_messages(topics=topic_name)]
        ref_frame_p = [msg[1].p.data for msg in self.inbag.read_messages(topics=topic_name)]
        ref_frame_a1 = [msg[1].a1.data for msg in self.inbag.read_messages(topics=topic_name)]
        ref_frame_a2 = [msg[1].a2.data for msg in self.inbag.read_messages(topics=topic_name)]
        ref_frame_A = [msg[1].A.data for msg in self.inbag.read_messages(topics=topic_name)]
        ref_frame_A_inv = [msg[1].A_inv.data for msg in self.inbag.read_messages(topics=topic_name)]
        
        # write reference frame data to SEPARATE hdf5
        # *** NB: need to update this to save left and right side, eventually
        save_name = [s for s in topic_name.split('/') if not s == ""][0] + file_suffix + '_rframe_fits.hdf5'
        save_path = os.path.normpath(os.path.join(self.fly_path, save_name))
        sub_topic = [s for s in topic_name.split('/') if not s == ""][-1]
        
        print('writing %s data for %s' % (topic_name, self.fly_id_str))
        with h5py.File(save_path, 'a') as h5f:
            h5f.create_dataset('/%s/ros_tstamps'%(sub_topic), data=ros_tstamps)
            h5f.create_dataset('/%s/components'%(sub_topic), data=ref_frame_components)
            h5f.create_dataset('/%s/p'%(sub_topic), data=np.array(ref_frame_p))
            h5f.create_dataset('/%s/a1'%(sub_topic), data=np.array(ref_frame_a1))
            h5f.create_dataset('/%s/a2'%(sub_topic), data=np.array(ref_frame_a2))
            h5f.create_dataset('/%s/A'%(sub_topic), data=np.array(ref_frame_A))
            h5f.create_dataset('/%s/A_inv'%(sub_topic), data=np.array(ref_frame_A_inv))
      
    # --------------------------------------------------------------------------------
    def do_conversion_all(self):
        """
        function to execute the various read/write functions for a given file
        NB: can edit as needed/demands for different data types change
        """
        # read/write camera data. NB: this will be to separate files
        for topic in self.cam_topic_names:
            self.read_cam_data(topic)
           
        # read/write muscle reference frame info
        for topic in self.rframe_topic_names:
            self.read_rframe_data(topic)

        # read/write kinefly data
        for topic in self.kinefly_topic_names:
            self.read_kinefly_data(topic)

        # read/write led panel data
        for topic in self.led_panel_topic_names:
            self.read_ledpanels_data(topic)

        # read/write daq phidget data
        for topic in self.daq_phidget_topic_names:
            self.read_daq_phidget_data(topic)

        # read/write experiment 'state' messages like 'exp_state' and 'exp_block'
        for topic in self.exp_msg_topic_names:
            self.read_general_msg_data(topic)

        # read/write muscle data
        for topic in self.unmixer_topic_names:
            self.read_muscle_unmixer_data(topic)

        # read/write metadata. NB: this will be to separate files
        for topic in self.metadata_topic_names:
            self.read_metadata(topic_name=topic)


################################################################################################
########################### RUN SCRIPT #########################################################
################################################################################################
if __name__ == '__main__':
    # read in flies from terminal input or specified list
    if not FLIES:
        flies = [int(x) for x in sys.argv[1:]]
    else:
        flies = FLIES

    # create instance of converter object
    myBagToHDF5 = BagToHDF5(fly_db_path=FLY_DB_PATH)

    # read out all data for each fly
    for fly in flies:
        # make sure we're looking at current fly
        if isinstance(fly, int):
            myBagToHDF5.set_fly_num(fly)
            print('Unbagging Fly%04d...' %(fly))
        else:
            myBagToHDF5.set_fly_fn(fly)
            print('Unbagging %s...' %(fly))

        # grab data out
        myBagToHDF5.do_conversion_all()

    # close any files open in reader
    try:
        myBagToHDF5.close_files()
    except AttributeError:
        print('No flies selected for unbagging')
