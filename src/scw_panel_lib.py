# !/usr/bin/env python
###########################################################################
# separate document to keep experiment functions a little cleaner
#
# can define some experiment params here as well(?)
###########################################################################
import time
import numpy as np

from ledpanels import display_ctrl

###########################################################################
#################### HELPER FUNCTIONS #####################################
###########################################################################
"""
Quick function to define a dictionary that takes pattern strings as inputs and outputs SD card filenames
...could be improved a lot (include gain/bias params?)

"""
def get_pattern_dict():
    pattern_dict = {'stripe'     : 'Pattern_bar.mat',
                    'yaw_right'  : 'Pattern_rot_axis_5.mat',
                    'yaw_left'   : 'Pattern_rot_axis_4.mat',
                    'pitch_up'   : 'Pattern_rot_axis_0.mat',
                    'pitch_down' : 'Pattern_rot_axis_2.mat',
                    'roll_left'  : 'Pattern_rot_axis_3.mat',
                    'roll_right' : 'Pattern_rot_axis_1.mat',
                    }

    return pattern_dict

# ------------------------------------------

"""
Turn panels ON for a fixed duration

 blk_pub.publish('all_on')
"""
def turn_on_panels(ctrl, duration=1):

    ctrl.stop()
    ctrl.all_on()
    time.sleep(duration)
    ctrl.stop()

# ------------------------------------------

"""
Turn panels OFF for a fixed duration

 blk_pub.publish('all_off')

"""
def turn_off_panels(ctrl, duration=1):

    ctrl.stop()
    ctrl.all_off()
    time.sleep(duration)
    ctrl.stop()

# ------------------------------------------

"""
Function to execute generic visual stimulus

INPUTS:
    - ctrl: LED panel display controller object
    - block_name: string describing the type of visual stimulus. should be of the form '(feedback type)_(rot axis)_(direction)' 
        i.e. 'ol_yaw_left' = open loop yaw left or 'cl_stripe' = closed loop stripe
        NB: do not include rep number here
    - duration: time to show stimulus in seconds
    - gain_x, gain_y: gains in x, y directions
    - bias_x, bias_y: bias in x, y directions
    - ch: boolean for using a Chrimson stimulus or not (under construction!)
    - x_init, y_init: initial position values for visual stim
"""

def exc_visual_stim(ctrl, block_name, duration, gain_x=0, gain_y=0, bias_x=0, bias_y=0, ch=0, x_init=None):
    
    # print stimulus name so we can see it
    # print block_name

    # try to extract some info from the string pattern
    block_name_split = block_name.split('_')

    # first see if we're doing open or closed loop 
    if block_name_split[0]=='cl':
        open_loop_flag = False
        xrate_fun = 'ch0'  # still not *sure* what this does, but i think it makes the x rate dependent on analog in
        y_idx = 0  # index in the "y" position (4th dim of pattern array)
        if not x_init:
            x_init = 64  # start value "x" position of pattern (3rd dim of pattery array) 

    elif block_name_split[0]=='ol':
        open_loop_flag = True
        xrate_fun = 'funcx'
        y_idx = 1  # index in the "y" position (4th dim of pattern array)
        if not x_init:
            x_init = np.random.randint(0,96)  # start value "x" position of pattern (3rd dim of pattery array) 

    else:
        raise Exception("Could not determine feedback type (open vs closed loop)")

    # next determine rotational axis/visual pattern. 
    # NB: this determines what to load off of SD card
    # pat_str = '_'.join(block_name_split[1:np.min(np.array([len(block_name_split),2]))])
    pat_str = '_'.join(block_name_split[1:])
    pattern_dict = get_pattern_dict()
    pattern_name = pattern_dict[pat_str]

    # intialize led panels
    ctrl.stop()
    ctrl.set_position_function_by_name('X', 'default')  # not sure what this does
    ctrl.set_pattern_by_name(pattern_name)  # set pattern 
    ctrl.set_position(x_init, y_idx)  # set initial position
    ctrl.set_mode('xrate=%s'%(xrate_fun),'yrate=funcy')  # not really sure what this does
    ctrl.send_gain_bias(gain_x=gain_x, gain_y=gain_y, bias_x=bias_x, bias_y=bias_y) # set gain and bias for panel
    
    # execute panel motion
    ctrl.start()
    time.sleep(duration)
    ctrl.stop()

# ------------------------------------------

"""
Function to read in experiment params from a .yaml file 

Want to make sure that this gets saved as well, and then I'll have a good way of keeping track of how each expt was done

"""
# def read_exp_params():
    
