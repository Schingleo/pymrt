Python Environment Setup
========================

As the packages has a few package dependencies and get them set up properly
for your python environment may be tricky sometimes.
The following provides a general suggestion on how to set up each packages
for better performance on various operating systems.

Python 3.6 Environment
----------------------

On MacOS and Windows 10, I personally recommend `MiniConda`_ package
installer for python environment.
`MiniConda`_ provides most numerical calculation packages such as ``numpy``
and ``scipy``, pre-compiled for all operating systems with Intel Math kernel
Library - probably the best performance you can get as pre-compiled binaries.
It also includes virtual environment management so that you can have multiple
Python environments co-existing on your machine.
`MiniConda`_ is installed with minimal packages and you can add additional
packages that you need incrementally to keep a minimal footprint on your hard
drive.

Windows Setup
~~~~~~~~~~~~~

Setup `MiniConda`_ on windows, simply follow the link and download the
installer for you Windows OS.
Run the installer to install `MiniConda`_ to your machine.

Ubuntu 18.04 Setup
~~~~~~~~~~~~~~~~~~

Setup `MiniConda`_ on Linux, simple follow the link and download the bash
installer for Linux operationg system.
In terminal, make the downloaded bash script executable, and run it with
`sudo` command as follows.

.. code-block:: bash

    $ sudo ./Miniconda3-latest-Linux-x86_64.sh

In my case, I installed it under `/opt/miniconda3` so that it is accessible
for all users.

Add the path of `Miniconda` to `~/.bashrc`:

.. code-block:: bash

    export PATH="/opt/miniconda3/bin:$PATH"

MacOS Setup
~~~~~~~~~~~

In MacOS, I would recommend using `homebrew`_ as package manager to install
`MiniConda`_.

Install `homebrew`_ in terminal:

.. code-block:: bash

    $ /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

The command above fetches `homebrew`_ from GitHub and installs Xcode command
line tools as well.

Add `homebrew`_ to path

.. code-block:: bash

    $ echo export PATH='/usr/local/bin:$PATH' >> ~/.profile

Install basic `MiniConda`_ environment using `homebrew`_

.. code-block:: bash

    $ brew install wget
    $ wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
    $ bash Miniconda3-latest-MacOSX-x86_64.sh
    $ rm ~/Miniconda3-latest-MacOSX-x86_64.sh
    $ echo export PATH='~/miniconda3/bin:$PATH' >> ~/.profile
    $ source ~/.profile
    $ conda install anaconda
    $ conda update --all


Setup Tensorflow
----------------

For install ``tensorflow`` for your operating system, you can find
instruction on tensorflow document page `here <https://www.tensorflow
.org/install/>`_.

Note that on Windows with Conda environment, there is a chance that an out-dated
``html5lib`` package dependency may break the Conda setup.
As a walk around, you can run ``$pip install html5lib==1.0b10`` to correct it.
The fix has been merged into ``tensorflow`` source tree, but has not released
yet.

Setup Mayavi
------------

`Mayavi`_ library is used for 3D plot and visualization.
However, set it up properly takes quite some work.

Windows Setup
~~~~~~~~~~~~~

First, make sure that you have Visual Studio installed. In my case, I use
VS2017 Community for compilation.
Moreover, in VS2017, you need to enable ***Python Support*** and have
***Python Native Development Tools*** installed.

Start ***x64 Native Tools Command Prompt for VS2017*** in start menu and import
Conda Python environment scripts (usually named as ``activate.bat``).
The default one for base environment is at ``Scripts\activate.bat`` under
conda installation directory.
(Replace ``C:\Anaconda3`` in the following command with your installation
path of conda).

.. code-block:: bat

    > C:\Anaconda3\Scripts\activate.bat C:\Anaconda3


Install pyside 1.2.4

.. code-block:: bat

    > conda install -c conda-forge pyside=1.2.4

However, if you have ``pyqt`` package installed on your system, you may see
it fails with error complaining about version conflicts.
Remove ``pyqt`` first.

.. code-block:: bat

    > conda uninstall pyqt


Install VTK from clinicalgraphics

.. code-block:: bat

    > conda install -c clinicalgraphics vtk


Due to various bugs and compatibility issue, install mayavi, traits and
pyface from source (Github).

.. code-block:: bat

    > pip install git+https://github.com/enthought/envisage.git
    > pip install git+https://github.com/enthought/traitsui.git
    > pip install git+https://github.com/enthought/pyface.git
    > pip install git+https://github.com/enthought/mayavi.git

Ubuntu 18.04 Setup
~~~~~~~~~~~~~~~~~~

You can install the `mayavi` in the same way as in Windows.
In addition to the previous steps, you als need to install
`libcanberra-gtk-module` and `libcanberra-gtk3-module` using system package
manager.

.. code-block:: bash

    $ sudo apt install libcanberra-gtk-module libcanberra-gtk3-module


MacOS Setup
~~~~~~~~~~~

First, install VTK using `homebrew`_.

.. code-block:: bash

    $ brew install vtk --with-python3 --without-python --with-qt


Install pyside 1.2.4

.. code-block:: bash

    conda install pyside


Due to various bugs and compatibility issue, install mayavi, traits and
pyface from source (Github).
It takes a while to install and compile all of them from the source.

.. code-block:: bash

    $ pip install git+https://github.com/enthought/envisage.git
    $ pip install git+https://github.com/enthought/traitsui.git
    $ pip install git+https://github.com/enthought/pyface.git
    $ pip install git+https://github.com/enthought/mayavi.git


Known Issues
------------

mlab.axes() causes exception in Mayavi 4.5.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When use mlab.axes, the following exception message is observed in terminal::

    TypeError: SetInputData argument 1:method requires a vtkDataSet, a
    vtkPolyDataNormals was provided. (in _wrap_call)

    AttributeError: 'PolyDataNormals' object has no attribute 'bounds'

You can find fix on *mayavi* github page at `#474 <https://github
.com/enthought/mayavi/issues/474>`_.


UnicodeDecodeError while trying to close mayavi
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When you close ``mayavi`` window, you may saw the following error and the
window is not closed unless you kills it using processor manager.
The message may read::

    UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 3: invalid start byte

The issued is tracked on *mayavi* Github page at `#576 <https://github
.com/enthought/mayavi/issues/576>`_.

The fix for the issue is merged to master branch on Feb 14, 2018.


.. _`MiniConda`: https://conda.io/miniconda.html
.. _homebrew: https://brew.sh/
.. _Mayavi: https://github.com/enthought/mayavi
