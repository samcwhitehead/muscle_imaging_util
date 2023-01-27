#! /usr/bin/python

import sys, cv2, os, rosbag, h5py
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import fnmatch

class bag2hdf5BR:

    def __init__(self,data_loc='/home/imager/work/data/muscle_imaging_integration_bagfiles',file_folder = 'toConvert',file_folder_out='hdf5_converted'):
        ## Find bag file
        os.chdir(data_loc)
        for file in os.listdir(file_folder):
            if file.endswith('.bag'):
                bag_path = data_loc+'/'+file_folder+'/'+file
                bag_file_name = bag_path[-26:]
                print ('running: ' +  bag_file_name)
                self.inbag = rosbag.Bag(bag_path)
                self.cv_bridge = CvBridge()
                ## Initialize hdf5 file
                hdf5_file_name = data_loc+'/'+file_folder_out+'/' + bag_file_name[:-4] + '.hdf5'
                self.h5f = h5py.File(hdf5_file_name,'w')
                ## Read topics and save to hdf5 file

                #read data
                self.read_kinefly_data('/kinefly/flystate')
                print('reading kinefly data')

                self.read_ledpanels_data('/ledpanels/command')
                print('reading ledpanels data')

                self.read_daq_phidget_data('/phidgets_daq/all_data')
                print('reading daq data')

                self.read_muscle_unmixer_data('/unmixer_left/b1')
                self.read_muscle_unmixer_data('/unmixer_left/b2')
                self.read_muscle_unmixer_data('/unmixer_left/b3')
                self.read_muscle_unmixer_data('/unmixer_left/bkg')
                self.read_muscle_unmixer_data('/unmixer_left/hg1')
                self.read_muscle_unmixer_data('/unmixer_left/hg2')
                self.read_muscle_unmixer_data('/unmixer_left/hg3')
                self.read_muscle_unmixer_data('/unmixer_left/hg4')
                self.read_muscle_unmixer_data('/unmixer_left/i1')
                self.read_muscle_unmixer_data('/unmixer_left/i2')
                self.read_muscle_unmixer_data('/unmixer_left/iii1')
                self.read_muscle_unmixer_data('/unmixer_left/iii24')
                self.read_muscle_unmixer_data('/unmixer_left/iii3')
                #self.read_muscle_unmixer_data('/unmixer_left/image_output')
                self.read_muscle_unmixer_data('/unmixer_left/nm')
                self.read_muscle_unmixer_data('/unmixer_left/pr')
                self.read_muscle_unmixer_data('/unmixer_left/tpd')
                self.read_muscle_unmixer_data('/unmixer_right/b1')
                self.read_muscle_unmixer_data('/unmixer_right/b2')
                self.read_muscle_unmixer_data('/unmixer_right/b3')
                self.read_muscle_unmixer_data('/unmixer_right/bkg')
                self.read_muscle_unmixer_data('/unmixer_right/hg1')
                self.read_muscle_unmixer_data('/unmixer_right/hg2')
                self.read_muscle_unmixer_data('/unmixer_right/hg3')
                self.read_muscle_unmixer_data('/unmixer_right/hg4')
                self.read_muscle_unmixer_data('/unmixer_right/i1')
                self.read_muscle_unmixer_data('/unmixer_right/i2')
                self.read_muscle_unmixer_data('/unmixer_right/iii1')
                self.read_muscle_unmixer_data('/unmixer_right/iii24')
                self.read_muscle_unmixer_data('/unmixer_right/iii3')
                #self.read_muscle_unmixer_data('/unmixer_right/image_output
                self.read_muscle_unmixer_data('/unmixer_right/nm')
                self.read_muscle_unmixer_data('/unmixer_right/pr')
                self.read_muscle_unmixer_data('/unmixer_right/tpd')

                self.h5f.close()
                print ' '

    def read_kinefly_data(self,topic_name):
        tstamps     = [msg[1].header.stamp.to_time() for msg in self.inbag.read_messages(topics = topic_name) if len(msg[1].left.angles)>0 and len(msg[1].right.angles)>0]
        seq         = [msg[1].header.seq             for msg in self.inbag.read_messages(topics = topic_name) if len(msg[1].left.angles)>0 and len(msg[1].right.angles)>0]
        langles=      [msg[1].left.angles[0]         for msg in self.inbag.read_messages(topics = topic_name) if len(msg[1].left.angles)>0 and len(msg[1].right.angles)>0]
        lgradients=   [msg[1].left.gradients[0]      for msg in self.inbag.read_messages(topics = topic_name) if len(msg[1].left.angles)>0 and len(msg[1].right.angles)>0]
        lintensity=   [msg[1].left.intensity         for msg in self.inbag.read_messages(topics = topic_name) if len(msg[1].left.angles)>0 and len(msg[1].right.angles)>0]
        rangles=      [msg[1].right.angles[0]        for msg in self.inbag.read_messages(topics = topic_name) if len(msg[1].left.angles)>0 and len(msg[1].right.angles)>0]
        rgradients=   [msg[1].right.gradients[0 ]    for msg in self.inbag.read_messages(topics = topic_name) if len(msg[1].left.angles)>0 and len(msg[1].right.angles)>0]
        rintensity=   [msg[1].right.intensity        for msg in self.inbag.read_messages(topics = topic_name) if len(msg[1].left.angles)>0 and len(msg[1].right.angles)>0]
        #count=        [msg[1].count                  for msg in self.inbag.read_messages(topics = topic_name) if len(msg[1].left.angles)>0 and len(msg[1].right.angles)>0]
        try:
            hangles=      [msg[1].head.angles[0]         for msg in self.inbag.read_messages(topics = topic_name) if len(msg[1].left.angles)>0 and len(msg[1].right.angles)>0]
        except:
            pass

        print 'writing the kinefly data to hdf5'
        self.h5f.create_dataset('flystate_tstamps',    data=tstamps)
        self.h5f.create_dataset('flystate_seq',    data=seq)
        self.h5f.create_dataset('flystate_langles',data=langles)
        self.h5f.create_dataset('flystate_lgrandients',data=lgradients)
        self.h5f.create_dataset('flystate_lintensity',data=lintensity)
        self.h5f.create_dataset('flystate_rangles',data=rangles)
        self.h5f.create_dataset('flystate_rgradients',data=rgradients)
        self.h5f.create_dataset('flystate_rintensity',data=rintensity)
        #self.h5f.create_dataset('flystate_count',data=count)
        try:
            self.h5f.create_dataset('flystate_hangles',data=hangles)
        except:
            pass

    def read_ledpanels_data(self, topic_name):
        ros_tstamps = [msg[2].to_time()     for msg in self.inbag.read_messages(topics = topic_name)]
        panels_command = [msg[1].command    for msg in self.inbag.read_messages(topics = topic_name)]
        panels_arg1 = [msg[1].arg1 for msg in self.inbag.read_messages(topics = topic_name)]
        panels_arg2 = [msg[1].arg2 for msg in self.inbag.read_messages(topics = topic_name)]
        panels_arg3 = [msg[1].arg3 for msg in self.inbag.read_messages(topics = topic_name)]
        panels_arg4 = [msg[1].arg4 for msg in self.inbag.read_messages(topics = topic_name)]
        panels_arg5 = [msg[1].arg5 for msg in self.inbag.read_messages(topics = topic_name)]
        panels_arg6 = [msg[1].arg6 for msg in self.inbag.read_messages(topics = topic_name)]
       
        print 'writing the ledpanels data to hdf5'
        self.h5f.create_dataset('ledpanels_ros_tstamps',    data=ros_tstamps)
        self.h5f.create_dataset('ledpanels_panels_command',data=panels_command)
        self.h5f.create_dataset('ledpanels_panels_arg1',data=panels_arg1)
        self.h5f.create_dataset('ledpanels_panels_arg2',data=panels_arg2)
        self.h5f.create_dataset('ledpanels_panels_arg3',data=panels_arg3)
        self.h5f.create_dataset('ledpanels_panels_arg4',data=panels_arg4)
        self.h5f.create_dataset('ledpanels_panels_arg5',data=panels_arg5)
        self.h5f.create_dataset('ledpanels_panels_arg6',data=panels_arg6)


    def read_muscle_unmixer_data(self, topic_name):
        ros_tstamps = [msg[2].to_time()     for msg in self.inbag.read_messages(topics = topic_name)]
        tstamps     = [msg[1].header.stamp.to_time() for msg in self.inbag.read_messages(topics = topic_name)]
        seq         = [msg[1].header.seq             for msg in self.inbag.read_messages(topics = topic_name)]
        value       = [msg[1].value         for msg in self.inbag.read_messages(topics = topic_name)]
        muscle      = [msg[1].muscle         for msg in self.inbag.read_messages(topics = topic_name)]

        print ('writing the muscle unmixer data to hdf5')
        self.h5f.create_dataset(str(topic_name[1:])+'/muscleunmixer_ros_tstamps',    data=ros_tstamps)
        self.h5f.create_dataset(str(topic_name[1:])+'/muscleunmixer_tstamps',    data=tstamps)
        self.h5f.create_dataset(str(topic_name[1:])+'/muscleunmixer_seq',    data=seq)
        self.h5f.create_dataset(str(topic_name[1:])+'/muscleunmixer_value',    data=value)
        self.h5f.create_dataset(str(topic_name[1:])+'/muscleunmixer_muscle',    data=muscle)


    def read_daq_phidget_data(self, topic_name):
        ros_tstamps = [msg[2].to_time()     for msg in self.inbag.read_messages(topics = topic_name)]
        time        = [msg[1].time          for msg in self.inbag.read_messages(topics = topic_name)]
        channels    = [msg[1].channels      for msg in self.inbag.read_messages(topics = topic_name)]
        values      = [msg[1].values        for msg in self.inbag.read_messages(topics = topic_name)]

        print 'writing the ledpanels data to hdf5'
        self.h5f.create_dataset('daq_ros_tstamps',    data=ros_tstamps)
        self.h5f.create_dataset('daq_time',    data=time)
        self.h5f.create_dataset('daq_channels',    data=channels)
        self.h5f.create_dataset('daq_value',    data=values)

bag2hdf5BR()
os.chdir('/home/imager/work/data') # change to script location
print 'Finished.'
