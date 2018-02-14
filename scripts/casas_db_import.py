""" Script: Import dataset from CASAS Database
"""
import os
import sys
import psycopg2
import pytz
import json
import dateutil.parser
import argparse

conn = None


def DbQuerySiteInfo(tbname):
    """
    Args:
        tbname (:obj:`str`): Testbed name.

    Return:
        string, string, bool, datetime, string for name, description,
        activeness, created datetime and timezone.
    """
    cur = conn.cursor()
    cur.execute(
        "select tbname, description, active, created_on, timezone from "
        "testbed where tbname='%s';" % tbname)
    rowlist = cur.fetchall()
    if len(rowlist) == 0:
        raise ValueError("No testbed named %s" % tbname)
    return rowlist[0]


def DbQuerySiteSensors(tbname):
    """
    Args:
        tbname (:obj:`str`): Testbed name.

    Return:
         Array of sensor dictionary.
    """
    sensor_lookup = {}
    sensor_list = []
    cur = conn.cursor()
    cur.execute(
        "select target, sensor_type, serial from detailed_all_sensors "
        "where tbname='%s';" % tbname)
    rowlist = cur.fetchall()
    for row in rowlist:
        if row[0] in sensor_lookup:
            sensor_lookup[row[0]]["types"].append(row[1])
            sensor_lookup[row[0]]["serial"].append(row[2])
        else:
            sensor = {
                "name": row[0],
                "types": [row[1]],
                "locX": 0.,
                "locY": 0.,
                "sizeX": 0.05,
                "sizeY": 0.02,
                "description": "",
                "tag": "",
                "serial": [row[2]]
            }
            sensor_list.append(sensor)
            sensor_lookup[row[0]] = sensor
    return sensor_list


def get_category(type_name):
    if 'Battery' in type_name:
        return 'Battery'
    if 'Radio' in type_name or 'Zigbee' in type_name:
        return 'Radio'
    if 'Motion' in type_name:
        return 'Motion'
    if 'Door' in type_name:
        return 'Door'
    if 'LightSwitch' in type_name:
        return 'LightSwitch'
    if 'Light' in type_name:
        return 'Light'
    if 'Temperature' in type_name or 'Thermostat' in type_name:
        return 'Temperature'
    if 'Item' in type_name:
        return 'Item'
    return 'Other'


def DbQuerySensorEvents(tbname, start_stamp, stop_stamp, site_timezone):
    """
    Args:
        tbname (:obj:`name`): Testbed name.
        start_stamp (:obj:`DateTime`): Start time stamp with timezone info.
        stop_stamp (:obj:`DateTime`): Stop time stamp with timezone info.
        site_timezone (:obj:`tzinfo`): Timezone information of the site.

    Returns:
        :obj:`tuple` of :obj:`list` of :obj:`str` that contains binary
            event, temperature events, light sensor events, radio events and
            other events.
    """
    bin_events = []
    temp_events = []
    light_events = []
    radio_events = []
    other_events = []
    cur = conn.cursor()
    cur.execute(
        "select stamp, target, message, sensor_type from detailed_all_events "
        "where tbname='%s' and stamp > '%s' and stamp < '%s';" % (
            tbname, str(start_stamp), str(stop_stamp)
        )
    )
    eventList = cur.fetchall()
    for event in eventList:
        eventTime = event[0].astimezone(site_timezone)
        sensor_name = event[1]
        sensor_message = event[2]
        sensor_type = event[3]
        event_time_string = eventTime.strftime(
            "%m/%d/%Y %H:%M:%S ") + eventTime.strftime("%z")[
                                    :3] + ":" + eventTime.strftime("%z")[3:]
        eventString = "%s,%s,%s,,,%s," % (
        event_time_string, sensor_name, sensor_message, sensor_type)
        if sensor_message in ['ON', 'OFF', 'ABSENT', 'PRESENT', 'OPEN',
                              'CLOSE']:
            bin_events.append(eventString)
        else:
            category = get_category(sensor_type)
            if category == "Temperature":
                temp_events.append(eventString)
            elif category == "Light":
                light_events.append(eventString)
            elif category == "Radio":
                radio_events.append(eventString)
            else:
                other_events.append(eventString)
    return bin_events, temp_events, light_events, radio_events, other_events


def SaveSite(site, sites_folder):
    site_dir = os.path.join(sites_folder, site["name"])
    os.makedirs(site_dir, exist_ok=True)
    site_file = os.path.join(site_dir, "site.json")
    fp = open(site_file, "w+")
    json.dump(site, fp, indent=4)
    fp.close()


def ImportDataset(name, tbname):
    dataset = {
        "name": name,
        "activities": [],
        "residents": [],
        "site": tbname,
    }
    return dataset


def SaveDataset(dataset, dataset_dir, bin_events, temp_events, light_events,
                radio_events, other_events):
    dataset_dir = os.path.join(dataset_dir, dataset["name"])
    os.makedirs(dataset_dir, exist_ok=True)
    dataset_file = os.path.join(dataset_dir, "dataset.json")
    fp = open(dataset_file, "w+")
    json.dump(dataset, fp, indent=4)
    fp.close()
    event_file = os.path.join(dataset_dir, "events.csv")
    fp = open(event_file, "w+")
    fp.write('\n'.join(bin_events))
    fp.close()
    event_file = os.path.join(dataset_dir, "temperature.csv")
    fp = open(event_file, "w+")
    fp.write('\n'.join(temp_events))
    fp.close()
    event_file = os.path.join(dataset_dir, "light.csv")
    fp = open(event_file, "w+")
    fp.write('\n'.join(light_events))
    fp.close()
    event_file = os.path.join(dataset_dir, "radio.csv")
    fp = open(event_file, "w+")
    fp.write('\n'.join(radio_events))
    fp.close()
    event_file = os.path.join(dataset_dir, "other.csv")
    fp = open(event_file, "w+")
    fp.write('\n'.join(other_events))
    fp.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load dataset from CASAS Database.")
    parser.add_argument('dbHost', type=str,
                        help='CASAS database server address (server:port)')
    parser.add_argument('-u', '--user', type=str,
                        help='CASAS database username')
    parser.add_argument('-p', '--password', type=str,
                        help='CASAS database password')
    parser.add_argument('site', type=str, help='CASAS testbed name to import')
    parser.add_argument('--dataonly', action='store_true',
                        help='Create dataset only')
    parser.add_argument('--siteonly', action='store_true',
                        help='Create site only')
    parser.add_argument('--start_date', type=str,
                        help='Start date (local) in format YYYY-MM-DD')
    parser.add_argument('--stop_date', type=str,
                        help='Stop date (local) in format YYYY-MM-DD')
    parser.add_argument('output_dir',
                        help='Folder to save the site and dataset')
    args = parser.parse_args();
    # Parse Host and Port
    if ':' in args.dbHost:
        host, port = args.dbHost.split(':')
    else:
        host = args.dbHost
        port = '5432'
    tbname = args.site
    user = args.user
    password = args.password
    # Connect to Database
    if user is None:
        print("Connect to CASAS database at %s:%s" % (host, port))
        conn = psycopg2.connect(host=host, port=port, database="smarthomedata")
    else:
        print("Connect to CASAS database at %s:%s as user %s" %
              (host, port, user)
              )
        conn = psycopg2.connect(host=host, port=port, database="smarthomedata",
                                user=user, password=password)

    try:
        print("Lookup site %s on database..." % tbname)
        site_info = DbQuerySiteInfo(tbname)

        if not args.dataonly:
            print("Import site %s..." % tbname)
            # import site
            sensor_info = DbQuerySiteSensors(tbname)
            site = {
                "name": tbname,
                "floorplan": "placeholder.png",
                "sensors": sensor_info,
                "timezone": site_info[4],
            }
            site_final_path = os.path.join(args.output_dir, "site")
            print("Save site %s to %s ..." % (tbname, site_final_path))
            SaveSite(site, site_final_path)

        # Get Dataset
        site_timezone = pytz.timezone(site_info[4])
        if not args.siteonly:
            start_date = site_timezone.localize(
                dateutil.parser.parse(args.start_date))
            stop_date = site_timezone.localize(
                dateutil.parser.parse(args.stop_date))
            print("Import Data between %s and %s" % (
            start_date.isoformat(), stop_date.isoformat()))
            bin_events, temp_events, light_events, radio_events, other_events \
                = DbQuerySensorEvents(
                    tbname, start_date, stop_date, site_timezone
                    )
            dataset = {
                "name": tbname + "_" + start_date.strftime(
                    "%y%m%d") + "_" + stop_date.strftime("%y%m%d"),
                "activities": [],
                "residents": [],
                "site": tbname,
            }
            dataset_final_path = os.path.join(args.output_dir, "data")
            print(
                "Save dataset %s to %s" % (dataset["name"], dataset_final_path))
            SaveDataset(dataset, dataset_final_path, bin_events, temp_events,
                        light_events, radio_events, other_events)
    except:
        print("Unexpected error:" + sys.exc_info()[0])
    conn.close()
