import os
import sys
import json
import humanize
import logging
import copy
import numpy as np
import dateutil.parser
from .site import CASASSite

logger = logging.getLogger(__name__)


class CASASDataset:
    r"""CASAS Dataset Class

    CASAS Dataset Class stores and processes all information regarding a
    recorded smart home dataset.

    A dataset is usually composed of a meta-data json file and a list of .csv
    file containing recorded sensor events.
    The meta-data about the dataset (dataset.json) is parsed and loaded into
    the ``data_dict`` attribute.
    Usually, it is a dictionary consisted of the following keys:

    * ``name``: name of the dataset.
    * ``activities``: list of activities (dictionary structure) tagged in the
      dataset.
    * ``residents``: list of residents (dictionary structure) tagged in the
      dataset.
    * ``site``: name of the site where this dataset is recorded.

    ``events.csv`` file within the dataset directory contains binary sensor
    events triggered by motion sensors, area motion sensors, item sensors,
    and door sensors. Actually, any sensor that reports "ON", "OFF", "OPEN",
    "CLOSE", "ABSENT" or "PRESENT" as their message are considered binary event
    sensors and are recorded sequentially within the file.
    Most of the algorithm and methods implemented in this library consume these
    binary sensor events to infer the possible trace of each resident and infer
    the activity they are performing at a given time slot.

    The ``event.csv`` file is composed of 6 columns separated by comma.
    Here are those columns:

    * **time tag**: MM/DD/YYYY HH:mm:ss with time zone information represents
      unambiguously a specific time spot where the sensor event
      occurs, e.g. 12/01/2016 00:01:41 -08:00
    * **sensor name**: the target name of the sensor that triggers the event.
    * **sensor message**: the message that the sensor reports. For motion
      sensor, it is either "ON" or "OFF". Door sensor sends "OPEN" or
      "CLOSE", while item sensor sends "PRESENT" or "ABSENT".
    * **resident names**: The name/alias of the resident that triggers the
      sensor event and message. If multiple residents are the cause of the
      sensor event, they are separated by ``;``.
    * **activity name**: The name/label of the activity that the resident who
      triggers the sensor event is performing. If the sensor is triggered by
      multiple residents and they are performing different activities,
      the activities labels are separated by ``;``.
    * **comments**: Extra comments about this sensor event. Usually it contains
      the sensor type there.

    The ``event.csv`` file is loaded with :meth:`load_event` function. All the
    loaded events are stored in ``event_list`` attributes. Each entry in the
    list is a dictionary composed of the following key-value pairs:

    * ``datetime``: a datetime structure representing a specific time spot.
    * ``sensor``: the target name of the sensor.
    * ``message``: the message that the sensor sends.
    * ``resident``: the name/alias of the resident if tagged.
    * ``activity``: the name/label of the acitivity if tagged.

    Attributes:
        data_dict (:obj:`dict`):
            A dictionary contains information about smart home.
        directory (:obj:`str`): Path to the directory that stores CASAS smart
            home data.
        site (:obj:`pymrt.casas.site.CASASSite`):
            The :obj:`CASASSite` class representing the smart home site
            information.
        activity_dict (:obj:`dict`): A dictionary indexed by the name of
            activity for faster activity lookup. Each activity is a
            :obj:`dict` structure composed of key ``name`` and ``color``.
        resident_dict (:obj:`dict`): A dictionary indexed by the name of
            resident living in the smart home for faster resident lookup.
            Each resident is a :obj:`dict` structure composed of key ``name``
            and ``color``.
        activity_indices_dict (:obj:`dict`): A dictionary indexed by the
            activity name where the activity is assigned an index value to be
            used in activity recognition algorithm.
        enabled_sensor_types (:obj:`dict`): A dictionary indexed by enabled
            sensor types. The corresponding value is a :obj:`list` of sensor
            names that belongs to the sensor type index. It is used in
            load_event function to provide the functionality of enable or
            disable specific sensors.
        sensor_list (:obj:`list`): List of sensors enabled. It will be
            populated after the dataset is loaded.
        sensor_indices_dict (:obj:`dict`): A dictionary indexed by sensor name
            with the index of the corresponding sensor as its value. It is
            used for fast look-up of sensor index during data pre-processing and
            post-processing for activity recognition and multi-resident
            tracking.
        event_list (:obj:`list`): List of loaded events.
        stats (:obj:`dict`): Dictionary containing simple statistics about
            the dataset.

    Parameters:
        directory (:obj:`str`): Directory that stores CASAS smart home data
        site_dir (:obj:`str`): Parent directory that contains the information
            about the site where this dataset is recorded.
        site (:obj:`CASASSite`): :obj:`CASASSite` structure that corresponds to
            the site.
    """
    def __init__(self, directory, site_dir=None, site=None):
        # Sanity checks
        if not os.path.isdir(directory):
            raise NotADirectoryError('%s is not a directory.' % directory)
        dataset_json_fname = os.path.join(directory, 'dataset.json')
        if not os.path.exists(dataset_json_fname):
            raise FileNotFoundError('CASAS dataset meta-data file '
                                    '\'data.json\' is not found under '
                                    'directory %s. Please check if the '
                                    'directory provided is correct.' %
                                    directory)
        # Finished check - Start loading data from CASAS dataset
        self.directory = directory
        f = open(dataset_json_fname, 'r')
        self.data_dict = json.load(f)
        self.site = self._get_site(site_dir=site_dir, site=site)
        self.activity_dict = {activity['name']: activity for activity in
                              self.data_dict['activities']}
        self.resident_dict = self._get_residents()
        # Generate indices for each activity
        self.activity_indices_dict = self._get_activity_indices()
        self.enabled_sensor_types = self.site.get_all_sensor_types()
        # Sensor list and indices dict populated after dataset is loaded
        self.sensor_list = []
        self.sensor_indices_dict = {}
        self.event_list = []
        # Simple statistics
        self.stats = {}
        # Additional information about the dataset
        self.description = self.data_dict[
            'description'] if 'description' in self.data_dict else ""

    def enable_sensor_with_types(self, type_array):
        """Load only certain type of sensors in the dataset
        """
        all_types = self.site.get_all_sensor_types()
        enabled_sensor_types = {}
        num_errors = 0
        for sensor_type in type_array:
            if sensor_type in all_types:
                enabled_sensor_types[sensor_type] = all_types[sensor_type]
            else:
                logger.error('Sensor type %s not found in site %s' % (
                sensor_type, self.site.get_name()))
                num_errors += 1
        if num_errors == 0:
            self.enabled_sensor_types = enabled_sensor_types

    def _get_residents(self):
        """Analyze residents information from loaded meta-data.

        Note that self.data_dict needs to be populated from the json file first
        before this method is called.
        """
        resident_dict = {}
        multi_resident_names = []
        for resident in self.data_dict['residents']:
            if ';' in resident['name']:
                multi_resident_names.append(resident['name'])
            else:
                resident_dict[resident['name']] = resident
        # Check and see if multi-resident names are composed of residents listed
        # in the meta-data
        for multi_resident_name in multi_resident_names:
            name_list = multi_resident_name.split(';')
            for name in name_list:
                if name not in resident_dict:
                    logger.warning('Resident %s in name %s does not exist in '
                                   'the meta-data. Please check the data '
                                   'integrity of the annotated dataset.' %
                                   (name, multi_resident_names))
        # Return the dictionary
        return resident_dict

    def _get_site(self, site_dir=None, site=None):
        """Returns the smart home site

        Args:
            site_dir (:obj:`str`): Path to the CASAS site directory where this
                dataset is recorded.
            site (:obj:`CASASSite`): The CASASSite structure containing the
                information about this dataset.
        """
        if type(site) is CASASSite:
            if self.data_dict['site'] == site.get_name():
                return site
        if site_dir is not None and os.path.isdir(site_dir):
            possible_site_path = site_dir
        else:
            from . import casas_sites_dir
            if casas_sites_dir is not None:
                possible_site_path = os.path.join(casas_sites_dir,
                                                  self.data_dict['site'])
            else:
                raise RuntimeError("Cannot load site specified in the dataset.")
        return CASASSite(directory=possible_site_path)

    def get_name(self):
        """Get the smart home name

        Returns:
            :obj:`str`: smart home name
        """
        return self.data_dict['name']

    def get_all_activity_names(self):
        """Get All Activities

        Returns:
            :obj:`list` of :obj:`str`: list of activity names
        """
        names = [activity['name'] for activity in self.data_dict['activities']]
        return names

    def get_activity(self, label):
        """Find the information about the activity

        Parameters:
            label (:obj:`str`): activity label

        Returns:
            :obj:`dict`: A dictionary containing activity information
        """
        return self.activity_dict.get(label, None)

    def get_activity_color(self, label):
        """Find the color string of the activity

        Parameters:
            label (:obj:`str`): activity label

        Returns:
            :obj:`str`: RGB color string
        """
        activity = self.get_activity(label)
        if activity is not None:
            return "#" + activity['color'][3:9]
        else:
            raise ValueError('Activity %s Not Found' % label)

    def get_resident(self, name):
        """Get Information about the resident

        Parameters:
            name (:obj:`str`): name of the resident

        Returns:
            :obj:`dict`: A Dictionary that stores resident information
        """
        return self.resident_dict.get(name, None)

    def get_resident_color(self, name):
        """Get the color string for the resident

        Parameters:
            name (:obj:`str`): name of the resident

        Returns:
            :obj:`str`: RGB color string representing the resident
        """
        resident = self.get_resident(name)
        if resident is not None:
            return "#" + resident['color'][3:9]
        else:
            raise ValueError('Resident %s Not Found' % name)

    def get_all_resident_names(self):
        """Get All Resident Names

        Returns:
            :obj:`list` of :obj:`str`: A list of resident names
        """
        names = [resident['name'] for resident in self.data_dict['residents']]
        return names

    def _get_sensor_indices(self):
        """Get a dictionary for sensor id look-up.
        """
        self.sensor_indices_dict = {}
        for key, sensor in enumerate(self.sensor_list):
            self.sensor_indices_dict[sensor['name']] = key

    def _get_activity_indices(self):
        """Get a dictionary for activity id look-up
        """
        activity_indices_dict = {}
        for key, activity in enumerate(self.data_dict['activities']):
            activity_indices_dict[activity['name']] = key
        return activity_indices_dict

    def _get_resident_indices(self):
        """Get a dictionary for residents id look-up
        """
        resident_indices_dict = {}
        index = 0
        for resident_name in self.resident_dict:
            resident_indices_dict[resident_name] = index
            index += 1
        return resident_indices_dict

    def load_events(self, show_progress=True):
        """Load events from CASAS event.csv file

        Args:
            show_progress (:obj:`bool`): Show progress of event loading
        """
        # Sanity check
        events_fname = os.path.join(self.directory, 'events.csv')
        if not os.path.isfile(events_fname):
            raise FileNotFoundError('Sensor event records not found in CASAS '
                                    'dataset %s. Check if event.csv exists '
                                    'under directory %s.' % (self.get_name(),
                                                             self.directory))
        if show_progress:
            file_size = os.path.getsize(events_fname)
            sys.stdout.write('Loading events from events.csv. Total size: ' +
                             humanize.naturalsize(file_size, True) + '\n')
            chunk_size = file_size / 100
            size_loaded = 0
            loaded_percentage = 0
        # Clear event_list
        self.event_list = []
        # Initialize supporting data structure
        # Record all sensor names that is valid and presented in the event file
        #  loaded. Contain all valid sensor names from the site, use dict for
        #  faster search.
        valid_sensor_names = {key: None for key in
                              self.site.get_all_sensor_names()}
        # List of sensors that appeared in sensor events but are not found in
        #  valid sensor names dictionary
        sensors_notfound_list = {}
        # Records all the enabled valid sensors that has one or more than one
        #  record in the sensor event file.
        logged_sensor_names = {}
        # A set contains enabled sensor names.
        enabled_sensor_names = set()
        for sensor_type, sensor_names in self.enabled_sensor_types.items():
            enabled_sensor_names = enabled_sensor_names.union(sensor_names)
        # Residents that are tagged in the dataset
        logged_residents = {}
        residents_notfound_list = {}
        # Activities that are tagged in the dataset
        logged_activities = {}
        activities_notfound_list = {}

        # Start reading event file. The process loads all sensor events into
        #  self.event_list. If the sensor reporting the event is enabled,
        #  the event is appended to the list. If it is disabled, the event is
        #  skipped. If the sensor cannot be found in the meta-data of smart
        #  home site, an error is reported and the event is skipped.
        f = open(events_fname, 'r')
        line_number = 0
        for line in f:
            line_number += 1
            word_list = str(str(line).strip()).split(',')
            if len(word_list) < 3:
                # If not enough item (at least 3) in the entry for a sensor
                #  event, report error and continue.
                logger.error(
                    'Error parsing %s:%d' % (events_fname, line_number))
                logger.error('  %s' % line)
                continue
            # Parse datetime
            event_time = dateutil.parser.parse(word_list[0])
            # Check sensor name
            event_sensor = word_list[1]
            if event_sensor not in valid_sensor_names:
                # If event sensor is not found in the site information,
                # record the issue and continue.
                if event_sensor not in sensors_notfound_list:
                    sensors_notfound_list[event_sensor] = 1
                    logger.warning(
                        'Sensor name %s not found in home metadata' %
                        event_sensor)
                sensors_notfound_list[event_sensor] += 1
                continue
            # If sensor name is found, and its type is enabled, log its name.
            if event_sensor in enabled_sensor_names:
                if event_sensor not in logged_sensor_names:
                    logged_sensor_names[event_sensor] = 0
                else:
                    logged_sensor_names[event_sensor] += 1
            else:
                # The sensor is disabled, skip to next event.
                continue
            event_message = word_list[2]
            # Parse resident name and activity label
            # In order to accommodate multi-resident scenario, resident and
            # activities can both be an array.
            if len(word_list) > 3:
                # Parse residents
                residents = word_list[3] if word_list[3] != "" else None
                # Check if the resident name is legit
                if residents is not None:
                    residents = residents.split(';')
                    # Check if all residents are valid
                    for i in range(len(residents) - 1, -1, -1):
                        if residents[i] not in self.resident_dict:
                            resident = residents.pop(i)
                            if resident not in residents_notfound_list:
                                residents_notfound_list[resident] = 0
                                logger.warning(
                                    "Resident %s is not found in resident list."
                                    % resident)
                            else:
                                residents_notfound_list[resident] += 1
                        else:
                            resident = residents[i]
                            if resident in logged_residents:
                                logged_residents[resident] += 1
                            else:
                                logged_residents[resident] = 0
                else:
                    residents = []
            else:
                residents = []

            if len(word_list) > 4:
                # Activities list
                activities = word_list[4] if word_list[4] != "" else None
                # Check if the resident name is legit
                if activities is not None:
                    activities = activities.split(';')
                    # Check if all residents are valid
                    for i in range(len(activities) - 1, -1, -1):
                        if activities[i] not in self.resident_dict:
                            activity = activities.pop(i)
                            if activity not in activities_notfound_list:
                                activities_notfound_list[activity] = 0
                                logger.warning(
                                    "Activity %s is not found in activity list."
                                    % activity)
                            else:
                                activities_notfound_list[activity] += 1
                        else:
                            if activity in logged_activities:
                                logged_activities[activity] += 1
                            else:
                                logged_activities[activity] = 0
                else:
                    activities = []
            else:
                activities = []

            cur_data_dict = {
                'datetime': event_time,
                'sensor': event_sensor,
                'message': event_message,
                'resident': residents,
                'activity': activities
            }

            # Add Corresponding Labels
            self.event_list.append(cur_data_dict)
            if show_progress:
                # Figure out a way of showing progress
                size_loaded += len(line)
                if size_loaded > chunk_size:
                    loaded_percentage += int(size_loaded / chunk_size)
                    size_loaded = size_loaded % chunk_size
                    sys.stdout.write("\rProgress: %d%%" %
                                     int(loaded_percentage))
        if show_progress:
            sys.stdout.write("\rProgress: 100%%\n")
        # Finished reading the whole file, create sensor list
        for sensor in self.site.sensor_list:
            if sensor['name'] in logged_sensor_names:
                self.sensor_list.append(sensor)
        self._get_sensor_indices()

    def to_sensor_sequence(self, ignore_off=True):
        """Change the loaded events into sensor sequence (timetag ignored).
        """
        sensor_seq = []
        for event in self.event_list:
            if ignore_off and \
                    (event['message'] == "OFF" or
                     event['message'] == 'ABSENT' or
                     event['message'] == 'CLOSE'):
                continue
            else:
                sensor_seq.append(self.sensor_indices_dict[event['sensor']])
        return sensor_seq

    def to_observation_track(self, show_progress=True,
                             sensor_check=False,
                             default_off_interval=5,
                             default_threshold=3600,
                             sensor_vector_mapping=None):
        """Acknowledge ON/OFF events and output an array of observations taken
           at current timestep

        Args:
            default_off_interval (:obj:`int`): Default turn-off interval if
                "OFF" event is missing.
            default_threshold (:obj:`int`): If we do not find a matching tag
                after the threshold, we enable automatic sensor shutdown
                interval.
            sensor_vector_mapping (:obj:`np.ndarray`): Mapping matrix between
                sensor id and vector embedding.
            show_progress (:obj:`bool`): Show observation track generation
                progress.
            sensor_check (:obj:`bool`): Whether to check matching sensor
                messages. For example, whether a 'CLOSE' is followed by 'OPEN'.
        """
        if show_progress:
            sys.stdout.write('Generate Observations from event list.\n')
            num_events = len(self.event_list)
            sys.stdout.write('Total events: %d\n' % num_events)
            num_event_chunk = num_events / 100
            num_events_processed = 0
            percentage_processed = 0

        observation_track = []  # to hold observations
        # dictionary to store current states of all sensors
        sensor_status = {sensor['name']: 0 for sensor in self.sensor_list}

        # TODO: Sensor check needs further testing
        if sensor_check:
            auto_shutdown = {}  # dictionary store shutdown time
            auto_shutdown_summary = {}
            for sensor in self.sensor_list:
                sensor_status[sensor['name']] = 0
                auto_shutdown[sensor['name']] = None
                auto_shutdown_summary[sensor['name']] = 0
        # Go through all events
        for i, event in enumerate(self.event_list):
            if event['message'] == "OFF" or event['message'] == 'ABSENT' or \
                    event['message'] == 'CLOSE':
                sensor_status[event['sensor']] = 0
                # Record sensor in shutdown list
                if sensor_check:
                    auto_shutdown[event['sensor']] = 0
            else:
                current_observation = []
                sensor_status[event['sensor']] = 1

                # Check auto-shutdown
                if sensor_check:
                    for j in range(i, len(self.event_list)):
                        if (self.event_list[j]['datetime'] - event['datetime'])\
                                .total_seconds() > default_threshold:
                            auto_shutdown[event['sensor']] = event['datetime']
                            if auto_shutdown_summary[event['sensor']] == 0:
                                logger.debug(
                                    "debug_warn: %s at %s does not have a "
                                    "closing tag matched within %d seconds" %
                                    (event['sensor'], event['datetime'],
                                     default_threshold))
                            auto_shutdown_summary[event['sensor']] += 1

                # Check for auto-shutdown
                for sensor in self.sensor_list:
                    sensor_name = sensor['name']
                    if sensor_status[sensor_name] == 1:
                        # Add auto-shutdown if sensor check is enabled
                        if sensor_check:
                            if auto_shutdown[sensor_name] == 1:
                                # Check time to see if it is down now.
                                if (event['datetime'] -
                                    auto_shutdown[sensor_name]).\
                                        total_seconds() > default_off_interval:
                                    auto_shutdown[sensor_name] = None
                                    sensor_status[sensor_name] = 0
                        # Append sensor
                        if sensor_status[sensor_name] == 1:
                            if sensor_vector_mapping is not None:
                                current_observation.append(
                                    sensor_vector_mapping[
                                        self.sensor_indices_dict[sensor_name], :
                                    ].T
                                )
                            else:
                                current_observation.append(
                                    self.sensor_indices_dict[sensor_name])
                observation_track.append(current_observation)
                if show_progress:
                    num_events_processed += 1
                    if num_events_processed > num_event_chunk:
                        percentage_processed += \
                            int(num_events_processed / num_event_chunk)
                        num_events_processed = num_events_processed % \
                                               num_event_chunk
                        sys.stdout.write('\rprogress: %d %%' %
                                         percentage_processed)
                logger.debug("Obs %5d: %s" % (i, str(current_observation)))
        if show_progress:
            sys.stdout.write('progress: 100%%\n')
        return observation_track

    def summary(self):
        """Print Summary of the dataset class
        """
        print('==============================')
        print('Dataset: %s' % self.get_name())
        print('Location: %s' % self.directory)
        print('==============================')
        print('\t Sensor Types enabled: %s' % str(
            list(self.enabled_sensor_types.keys())))
        if len(self.event_list) == 0:
            print('\t Events not loaded')
        else:
            print('\t %d events loaded.' % len(self.event_list))
            print('\t %d sensors presented in the dataset.' % len(
                self.sensor_list))

    def check_event_list(self, xlsx_fname=None):
        """Check event list for issues

        Common issues are:

        1. Motion sensor, Missing ON event (two consecutive OFF)
        2. Motion sensor, Missing OFF event (two consecutive ON)
        3. Item sensor, Missing ABSENT (two consecutive PRESENT)
        4. Item sensor, Missing PRESENT (two consecutive ABSENT)
        5. Door sensor, Missing OPEN (two consecutive CLOSE)
        6. Door sensor, Missing CLOSE (two consecutive OPEN)

        Args:
            xlsx_fname (:obj:`str`): XLSX file path to store the event checking
                result. Print the result in console if it is set to None.
        """
        # dict[sensor, state] that stores last state of a sensor
        prev_state = {sensor['name']: '' for sensor in self.sensor_list}
        # dict[sensor, int] that stores the occurrence of each sensor
        total_events = {sensor['name']: 0 for sensor in self.sensor_list}
        # dict[sensor, dict[state, int]] that stores the occurrence of each faults
        issues_summary = {sensor['name']: {} for sensor in self.sensor_list}
        for event in self.event_list:
            if event['message'] not in issues_summary[event['sensor']]:
                issues_summary[event['sensor']][event['message']] = 0.
            if prev_state[event['sensor']] == event['message']:
                issues_summary[event['sensor']][event['message']] += 1
                total_events[event['sensor']] += 1
            else:
                total_events[event['sensor']] += 0.5
            prev_state[event['sensor']] = event['message']
        # Report the result
        if xlsx_fname is None:
            # print on screen
            print('-------------')
            print('Issues Report')
            print('-------------')
            for sensor_name, issue_dict in issues_summary.items():
                issues = list(issue_dict.items())
                if total_events[sensor_name] < 2:
                    # Only occurred once in the dataset
                    logger.warning(
                        'Sensor %s only occurred once/twice in the whole '
                        'dataset.' % sensor_name)
                    continue
                if len(issues) < 2:
                    logger.warning(
                        'Sensor %s does not have two states in the file.' %
                        sensor_name)
                    logger.warning('Here is the issue dict:')
                    logger.warning(str(issues))
                print(
                    '[%s]: total %d, %s error %d (%.3f%%), %s error %d (%.3f%%)'
                    % (sensor_name, total_events[sensor_name],
                       issues[0][0], issues[0][1],
                       issues[0][1] * 100. / float(total_events[sensor_name]),
                       issues[1][0], issues[1][1],
                       issues[1][1] * 100. / float(total_events[sensor_name])))
        else:
            import xlsxwriter
            workbook = xlsxwriter.Workbook(xlsx_fname)
            summary_sheet = workbook.add_worksheet('Summary')
            for i, text in enumerate(
                    ['Sensors', 'Total Events', 'Issues', 'Events',
                     'Percentage']):
                summary_sheet.write(0, i, text)
            for i, name in enumerate(list(issues_summary.keys())):
                summary_sheet.merge_range(1 + 2 * i, 0, 2 + 2 * i, 0, name)
                summary_sheet.merge_range(1 + 2 * i, 1, 2 + 2 * i, 1,
                                          total_events[name])
                if total_events[name] < 2:
                    # Only occurred once in the dataset
                    logger.warning(
                        'Sensor %s only occurred once/twice in the whole '
                        'dataset.' % name)
                    # continue
                for j, issue_label in enumerate(
                        list(issues_summary[name].keys())):
                    summary_sheet.write(1 + j + 2 * i, 2, issue_label)
                    summary_sheet.write(1 + j + 2 * i, 3,
                                        issues_summary[name][issue_label])
                    summary_sheet.write(1 + j + 2 * i, 4, issues_summary[name][
                        issue_label] / float(total_events[name]))
            workbook.close()

    def check_sensor_activation_duration(self):
        """Summary of the duration of sensor activation
        """
        start_status = ['ON', 'PRESENT', 'OPEN']
        stop_status = ['OFF', 'ABSENT', 'CLOSE']
        # dict[sensor, state] that stores last state of a sensor
        prev_state = {sensor['name']: '' for sensor in self.sensor_list}
        # dict[sensor, datetime] that stores the time of last sensor state
        # report
        prev_time = {sensor['name']: None for sensor in self.sensor_list}
        # dict[sensor, list of int] that stores the duration summary for each
        # sensor firing (between ON and OFF)
        sensor_duration = {sensor['name']: [] for sensor in self.sensor_list}
        # dict[sensor, list of int] that stores the interval between sensor
        # deactivation to next activation
        sensor_interval = {sensor['name']: [] for sensor in self.sensor_list}
        for event in self.event_list:
            if prev_time[event['sensor_id']] is not None:
                if prev_state[event['sensor_id']] != event['sensor_status']:
                    if prev_state[event['sensor_id']] in start_status:
                        sensor_duration[event['sensor_id']].append(
                            (event['datetime'] - prev_time[
                                event['sensor_id']]).total_seconds())
                    else:
                        sensor_interval[event['sensor_id']].append(
                            (event['datetime'] - prev_time[
                                event['sensor_id']]).total_seconds())
            # Update previous state and time
            prev_state[event['sensor_id']] = event['sensor_status']
            prev_time[event['sensor_id']] = event['datetime']
        for sensor in sensor_duration.keys():
            sensor_duration[sensor] = sorted(sensor_duration[sensor],
                                             reverse=True)
        for sensor in sensor_interval.keys():
            sensor_interval[sensor] = sorted(sensor_interval[sensor],
                                             reverse=True)
        for sensor in sensor_duration.keys():
            print('%s: duration mid %f, mean %f, max %f, min %f' %
                  (sensor, np.median(sensor_duration[sensor]),
                   np.mean(sensor_duration[sensor]),
                   sensor_duration[sensor][0], sensor_duration[sensor][-1]))

    def save(self, fname, description=None):
        """Pickle the dataset structure for faster load and store

        Args:
            fname (:obj:`str`): The path to the pickle file which the dataset is
                saved to.
            description (:obj:`str`): Extra description of the dataset to be
                stored.
        """
        if description is not None:
            self.description = description
        import pickle
        fp = open(fname, 'wb')
        pickle.dump(self, fp, protocol=-1)
        fp.close()

    @staticmethod
    def load(fname):
        """Load from file where dataset is stored in pickle format.
        """
        if not os.path.isfile(fname):
            raise FileNotFoundError("File %s not found." % fname)
        fp = open(fname, 'rb')
        import pickle
        dataset = pickle.load(fp)
        assert (isinstance(dataset, CASASDataset))
        print(
            "Dataset %s is loaded from file successfully." % dataset.data_dict[
                'name'])
        print(dataset.description)
        fp.close()
        return dataset

    def annotate_mrt(self, track_association, dir_path, dataset_name,
                     description, residents=None):
        """
        """
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        # Get Resident Info
        if residents is None or len(residents) == 0:
            if self.data_dict['residents'] is not None and len(
                    self.data_dict['residents']) > 0:
                residents = self.data_dict['residents']
            else:
                residents = []
        # Copy metadata
        metadata_filename = os.path.join(dir_path, "dataset.json")
        fp = open(metadata_filename, 'w')
        metadata_dict = copy.deepcopy(self.data_dict)
        metadata_dict['name'] = dataset_name
        metadata_dict['description'] = description
        if len(metadata_dict['residents']) == 0:
            metadata_dict['residents'] = residents
            for resident_dict in metadata_dict['residents']:
                if 'color' not in resident_dict:
                    resident_dict['color'] = "#FF000000"
        json.dump(metadata_dict, fp, indent=4)
        fp.close()
        # Temporary dictionary keep previous status of sensor
        dict_sensor_association = {sensor['name']: None for sensor in
                                   self.sensor_list}
        # go through event list and create new tagged event list
        i = 0
        event_filename = os.path.join(dir_path, "events.csv")
        fp = open(event_filename, 'w+')
        for event in self.event_list:
            resident_tag = None
            if event['message'] == "OFF" or \
                    event['message'] == 'ABSENT' or \
                    event['message'] == 'CLOSE':
                resident_tag = dict_sensor_association[event['sensor']]
            else:
                resident_tag = track_association[i][
                    self.sensor_indices_dict[event['sensor']]]
                dict_sensor_association[event['sensor']] = resident_tag
                i += 1
            event_time = event['datetime']
            event_time_string = event_time.strftime("%m/%d/%Y %H:%M:%S ") + \
                event_time.strftime("%z")[:3] + ":" + \
                event_time.strftime("%z")[3:]
            if resident_tag == -1 or resident_tag is None:
                resident_string = ""
            else:
                if len(residents) > 0:
                    resident_string = residents[resident_tag]['name']
                else:
                    resident_string = "R" + str(resident_tag)

            event_string = "%s,%s,%s,%s,,\n" % (event_time_string,
                                                event['sensor'],
                                                event['message'],
                                                resident_string)
            fp.write(event_string)
        fp.close()
