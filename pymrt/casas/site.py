import os
import json
import logging
import numpy as np
import matplotlib.image as mimg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from ..utils.plot import LabeledLine
from .sensor_type import CASASSensorType


logger = logging.getLogger(__name__)


class CASASSite:
    """CASAS Site Class

    This class parses the metadata for a smart home site and loaded it into
    prameter ``data_dict``.
    Usually, the description of a site is composed of following keys:

    * ``name``: Site (Testbed) name.
    * ``floorplan``: Relative path to the floorplan image (usually \*.png).
    * ``sensors``: List of sensors populated in the smarthome.
    * ``timezone``: Local timezone of the site (IANA string).

    Each sensor in the metadata is a dictionary composed of following keys:

    * ``name``: Target name of the sensor.
    * ``types``: List of sensor types.
    * ``locX``: x position relative to the width of floorplan (between 0 and 1)
    * ``locY``: y position relative to the height of floorplan (between 0 and 1)
    * ``sizeX``: depricated.
    * ``sizeY``: depricated.
    * ``description``: custom description added to the sensor.
    * ``tag``: tag string of the sensor.
    * ``serial``: list of serial number for the tag.

    Note that the name and types are the two that matter in multi-resident
    activity recognition implementation.

    Attributes:
        data_dict (:obj:`dict`): A dictionary contains information about smart
            home.
        directory (:obj:`str`): Directory that stores CASAS smart home site data

    Parameters:
        directory (:obj:`str`): Directory where the meta-data describing a
            CASAS smart home site is stored.
    """
    def __init__(self, directory):
        if os.path.isdir(directory):
            self.directory = directory
            site_json_fname = os.path.join(directory, 'site.json')
            if os.path.exists(site_json_fname):
                f = open(site_json_fname, 'r')
                self.data_dict = json.load(f)
                self.sensor_list = self.data_dict['sensors']
                # Populate sensor_dict (dict type) for faster sensor
                # information lookup.
                self.sensor_dict = self._populate_sensor_lookup_dict()
            else:
                logger.error('Smart home metadata file %s does not exist. '
                             'Create an empty CASASHome Structure'
                             % site_json_fname)
                raise FileNotFoundError('File %s not found.' % site_json_fname)

    def _populate_sensor_lookup_dict(self):
        """ Populate a dictionary structure to find each sensor by target name.

        Returns:
            :obj:`dict`: A dictionary indexed by sensor target name (:obj:`str`)
                with a dictionary structure detailing the information of the
                sensor (:obj:`dict`).
        """
        sensor_dict = {}
        for sensor in self.sensor_list:
            sensor_dict[sensor['name']] = sensor
        return sensor_dict

    def get_name(self):
        """Get the smart home name

        Returns:
            :obj:`str`: smart home name
        """
        return self.data_dict['name']

    def get_sensor(self, name):
        """Get the information about the sensor

        Parameters:
            name (:obj:`str`): name of the sensor

        Returns:
            :obj:`dict`: A dictionary that stores sensor information
        """
        for sensor in self.data_dict['sensors']:
            if sensor['name'] == name:
                return sensor
        return None

    def get_all_sensor_names(self):
        """Get All Sensor Names

        Returns:
            :obj:`list` of :obj:`str`: a list of sensor names.
        """
        names = [sensor['name'] for sensor in self.sensor_list]
        return names

    def get_all_sensor_types(self):
        """ Get All Sensor Types

        Returns
            :obj:`dict`: A dictionary indexed by sensor type (:obj:`str`)
                with the value of a list of sensor target names that belongs
                to the index type (:obj:`list` of :obj:`str`).
        """
        sensor_types_dict = {}
        for sensor in self.sensor_list:
            for sensor_type in sensor['types']:
                if sensor_type not in sensor_types_dict:
                    sensor_types_dict[sensor_type] = [sensor['name']]
                elif sensor['name'] not in sensor_types_dict[sensor_type]:
                    sensor_types_dict[sensor_type].append(sensor['name'])
        return sensor_types_dict

    def sensor_type_summary(self):
        """Print summary on all types of sensors on site
        """
        sensor_types_dict = self.get_all_sensor_types()
        print('%d types of sensors on site' % len(sensor_types_dict))
        for sensor_type, sensor_list in sensor_types_dict.items():
            print('    %s:\n        %s' %
                  (sensor_type, ",\n        ".join(sensor_list)))

    def prepare_floorplan(self):
        """Prepare the floorplan for drawing

        This internal function generates a dictionary of elements to be ploted
        with matplotlib.
        The return is a dictionary composed of following keys:

        * ``img``: floor plan image loaded in :obj:`mimg` class for plotting
        * ``width``: actual width of the image
        * ``height``: actual height of the image
        * ``sensor_centers``: list of the centers to plot each sensor text.
        * ``sensor_boxes``: list of rectangles to show the location of sensor
          (:obj:`patches.Rectangle`).
        * ``sensor_texts``: list of sensor names

        Returns:
            :obj:`dict`: A dictionary contains all the pieces needed to draw
                the floorplan
        """
        floorplan_dict = {}
        img = mimg.imread(os.path.join(self.directory, self.data_dict['floorplan']))
        img_x = img.shape[1]
        img_y = img.shape[0]
        # Create Sensor List/Patches
        sensor_boxes = {}
        sensor_texts = {}
        sensor_centers = {}
        # Check Bias
        for sensor in self.data_dict['sensors']:
            loc_x = sensor['locX'] * img_x
            loc_y = sensor['locY'] * img_y
            width_x = 0.01*img_x
            width_y = 0.01*img_y
            sensor_category = \
                CASASSensorType.get_best_category_for_sensor(sensor['types'])
            sensor_color = CASASSensorType.get_category_color(sensor_category)
            sensor_boxes[sensor['name']] = \
                patches.Rectangle((loc_x - width_x / 2, loc_y - width_y / 2),
                                  width_x, width_y,
                                  edgecolor='grey',
                                  facecolor=sensor_color,
                                  linewidth=1,
                                  zorder=2)
            sensor_texts[sensor['name']] = (loc_x, loc_y + width_y + 1,
                                            sensor['name'])
            sensor_centers[sensor['name']] = (loc_x, loc_y)
        # Populate dictionary
        floorplan_dict['img'] = img
        floorplan_dict['width'] = img_x
        floorplan_dict['height'] = img_y
        floorplan_dict['sensor_centers'] = sensor_centers
        floorplan_dict['sensor_boxes'] = sensor_boxes
        floorplan_dict['sensor_texts'] = sensor_texts
        return floorplan_dict

    def draw_floorplan(self, filename=None):
        """Draw the floorplan of the house, save it to file or display it on screen

        Args:
            filename (:obj:`str`): Name of the file to save the floorplan to
        """
        floorplan_dict = self.prepare_floorplan()
        self._plot_floorplan(floorplan_dict, filename)

    @staticmethod
    def _plot_floorplan(floorplan_dict, filename=None):
        fig, (ax) = plt.subplots(1, 1)
        fig.set_size_inches(18, 18)
        ax.imshow(floorplan_dict['img'])
        # Draw Sensor block patches
        for key, patch in floorplan_dict['sensor_boxes'].items():
            ax.add_patch(patch)
        # Draw Sensor name
        for key, text in floorplan_dict['sensor_texts'].items():
            ax.text(*text, color='black', backgroundcolor=mcolors.colorConverter.to_rgba('#D3D3D3', 0.7),
                    horizontalalignment='center', verticalalignment='top',
                    zorder=3)
        if floorplan_dict.get('sensor_lines', None) is not None:
            for key, line in floorplan_dict['sensor_lines'].items():
                ax.add_line(line)
        if filename is None:
            # Show image
            fig.show()
        else:
            fig.savefig(filename)
            plt.close(fig)

    def plot_sensor_distance(self, sensor_name, distance_matrix, max_sensors=None, filename=None):
        """Plot distance in distance_matrix
        """
        sensor_index = self.get_all_sensors().index(sensor_name)
        num_sensors = len(self.data_dict['sensors'])
        floorplan_dict = self.prepare_floorplan()
        x1 = floorplan_dict['sensor_centers'][sensor_name][0]
        y1 = floorplan_dict['sensor_centers'][sensor_name][1]
        # Draw Lines, and Set alpha for each sensor box
        sensor_lines ={}
        for i in range(num_sensors):
            sensor = self.data_dict['sensors'][i]
            if sensor_name != sensor['name']:
                x2 = floorplan_dict['sensor_centers'][sensor['name']][0]
                y2 = floorplan_dict['sensor_centers'][sensor['name']][1]
                line = LabeledLine([x1, x2], [y1, y2], linewidth=1,
                                   linestyle='--', color='b', zorder=10,
                                   label='%.5f' % distance_matrix[sensor_index, i],
                                   alpha=(1 - distance_matrix[sensor_index, i]) * 0.9 + 0.1)
                sensor_lines[sensor['name']] = line
                floorplan_dict['sensor_boxes'][sensor['name']].set_alpha(1 - distance_matrix[sensor_index, i])
        # Only show up to `max_lines` of sensors
        if max_sensors is not None and max_sensors < num_sensors:
            sorted_index = np.argsort(distance_matrix[sensor_index, :])
            for i in range(max_sensors + 1, num_sensors):
                sensor_lines.pop(self.data_dict['sensors'][sorted_index[i]]['name'], None)
        floorplan_dict['sensor_lines'] = sensor_lines
        self._plot_floorplan(floorplan_dict, filename)

    def summary(self):
        """Print brief site summary
        """
        print('[%s]: %s' % (self.get_name(), self.directory))
        print('\t sensors %d' % len(self.sensor_list))
        print('\t floorplan %s\n' % os.path.join(self.directory, self.data_dict['floorplan']))

    @staticmethod
    def load_all_sites(sites_dir=None):
        """Load all CASAS sites from data folder

        Returns:
            :obj:`dict` of :obj:`CASASSite`: indexed by the name (:obj:`str`) of each site.
        """
        if sites_dir is None:
            from . import SITES_DIR
            sites_dir = SITES_DIR
        sites_list = {}
        if os.path.isdir(sites_dir):
            site_names = os.listdir(sites_dir)
            for site_name in site_names:
                site_dirname = os.path.join(sites_dir, site_name)
                if os.path.isdir(site_dirname):
                    cur_site = CASASSite(site_dirname)
                    sites_list[cur_site.get_name()] = cur_site
        return sites_list

