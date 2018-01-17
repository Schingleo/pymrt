""" CASASSite class test

This is a simple test program of CASASSite class to load sensor information
and floorplan of a CASAS smart home site and plot in 2D using matplotlib
library.
"""

import logging
import argparse
import matplotlib.pyplot as plt
from pymrar.casas.site import CASASSite


def config_debug():
    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(asctime)s] %(name)s:%(levelname)s:%(message)s',
        handlers=[logging.StreamHandler()])


def parse_args():
    parser = argparse.ArgumentParser(description="Load and Plot Casas Site")
    parser.add_argument("site_dir",
                        help="Directory where CASAS site meta-data is stored")
    parser.add_argument("--plot2d", action='store_true',
                        help="Plot site in 2D using matplotlib")
    # parser.add_argument("--plot3d", action='store_true',
    #                     help="plot site in 3D using mayavi")
    return parser.parse_args()


if __name__ == "__main__":
    config_debug()
    args = parse_args()
    site = CASASSite(directory=args.site_dir)
    site.summary()
    print("All sensors: \n%s\n" % '\n'.join(site.get_all_sensor_names()))
    print("All sensor types: \n%s\n" % '\n'.join(site.get_all_sensor_types()))
    site.sensor_type_summary()
    if args.plot2d:
        site.draw_floorplan()
        plt.show()
