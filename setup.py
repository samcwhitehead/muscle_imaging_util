from setuptools import setup
from setuptools import find_packages
import os

setup(
    name='muscle_imaging_util',
    version='0.0.1',
    description='Python code for dealing with data from muscle/tendon imaging in Dickinson Lab.',
    long_description=__doc__,
    author='Sam Whitehead',
    author_email='swhitehe@caltech',
    license='MIT',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.11',
    ],

    packages=find_packages(exclude=['examples', 'scratch']),
)
