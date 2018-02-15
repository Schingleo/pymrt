# Copyright (c) 2015, Tinghui Wang <tinghui.wang@wsu.edu>
# All rights reserved.

from setuptools import setup, find_packages
import os

CLASSIFIERS = """\
Development Status :: 2 - Pre-Alpha
Intended Audience :: Developers
Intended Audience :: Science/Research
License :: OSI Approved :: MIT License
Operating System :: POSIX
Operating System :: POSIX :: Linux
Programming Language :: Python
Programming Language :: Python :: 3.6
Topic :: Home Automation
Topic :: Scientific/Engineering :: Artificial Intelligence
Topic :: Scientific/Engineering :: Information Analysis
""".splitlines()

NAME = "pymrt"
MAINTAINER = "Tinghui Wang (Steve)"
MAINTAINER_EMAIL = "tinghui.wang@wsu.edu"
DESCRIPTION = "Multi-Resident Tracking and Activity Recognition in Smart Home"
LONG_DESCRIPTION = DESCRIPTION
LICENSE = "BSD-3 Clause"
URL = "https://github.com/TinghuiWang/pymrt"
DOWNLOAD_URL = ""
AUTHOR = "Tinghui Wang (Steve)"
AUTHOR_EMAIL = "tinghui.wang@wsu.edu"
PLATFORMS = ["Linux", "Windows"]

# Get Version from pyActLearn.version
exec_results = {}
exec(open(os.path.join(os.path.dirname(__file__), 'pymrt/version.py')).read(),
     exec_results)
version = exec_results['version']

# Get Install Requirements
with open(os.path.join(os.path.dirname(__file__), 'requirements.txt'), 'r') as f:
    install_requires = f.read().splitlines()


def do_setup():
    setup(
        name=NAME,
        version=version,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        classifiers=CLASSIFIERS,
        platforms=PLATFORMS,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        url=URL,
        license=LICENSE,
        keywords=' '.join(['multi-resident', 'multi-target tracking',
                           'activity recognition', 'smart home']),
        packages=find_packages('.'),
        install_requires=install_requires,
    )


if __name__ == "__main__":
    do_setup()
