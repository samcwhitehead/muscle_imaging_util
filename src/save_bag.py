#! /usr/bin/python

import os
import sys
import shutil
FLYDB_PATH = '/media/imager/FlyDataD/FlyDB'
BAG_PATH = '/home/imager/bagfiles'

def main():
	newest_bag = sorted(os.listdir(BAG_PATH))[-1]
	newest_bag = os.path.join(BAG_PATH,newest_bag)
	last_fly = sorted(os.listdir(FLYDB_PATH))[-1]
	last_fn = int(last_fly.split('Fly')[1])
	next_fly = os.path.join(FLYDB_PATH,'Fly%04d'%(last_fn+1))
	print('copying %s \n to %s'%(newest_bag,next_fly))
	os.mkdir(next_fly)
	shutil.copy2(newest_bag,next_fly)

if __name__ == '__main__':
	main()
