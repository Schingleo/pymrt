""" Package pymrt.casas

This package contains classes and functions to interact with CASAS smart home
dataset.
"""

from .dataset import CASASDataset
from .site import CASASSite
from .sensor_type import CASASSensorType

global casas_sites_dir
casas_sites_dir = None
