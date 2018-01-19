"""Test 2D/3D data animation
"""

import os
import logging
import argparse
import pickle
from pymrt.casas.dataset import CASASDataset
from pymrt.utils.visualizer import *


def config_debug():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(name)s:%(levelname)s:%(message)s',
        handlers=[logging.StreamHandler()]
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Load and Plot Casas Site")
    parser.add_argument("dataset_dir",
                        help="Directory where CASAS dataset is stored")
    parser.add_argument("--site",
                        help="Directory where CASAS site meta-data is stored")
    return parser.parse_args()


def observation_summary(observations, sensor_list):
    """Statistical summary of the observation

    Args:
        observations (:obj:`list`):
        sensor_list (:obj:`list`):
    """
    print("Length of observation list: %d" % len(observations))
    # Average number of sensors that are active given a time step.
    avg_num_observation = 0.
    # How many consecutive steps does one sensor belongs to
    temp_sensor_count = {sensor['name']: 0 for sensor in sensor_list}
    sensor_duration = {sensor['name']: [] for sensor in sensor_list}
    for observation in observations:
        avg_num_observation += len(observation)
        activated_sensor_names = []
        for sensor_activated in observation:
            if sensor_activated is str:
                temp_sensor_count[sensor_activated] += 1
                activated_sensor_names.append(sensor_activated)
            else:
                temp_sensor_count[sensor_list[sensor_activated]['name']] += 1
                activated_sensor_names.append(
                    sensor_list[sensor_activated]['name']
                )
        for sensor_name, count in temp_sensor_count.items():
            if temp_sensor_count[sensor_name] != 0 and \
                    sensor_name not in activated_sensor_names:
                sensor_duration[sensor_name].append(
                    temp_sensor_count[sensor_name]
                )
                temp_sensor_count[sensor_name] = 0
    # Calculate statistical parameters
    avg_num_observation /= len(observations)
    print("Average activated sensor per observation: %.3f" %
          avg_num_observation)
    for sensor_name, count_list in sensor_duration.items():
        min_count = 0 if len(count_list) == 0 else np.min(count_list)
        max_count = 0 if len(count_list) == 0 else np.max(count_list)
        mean_count = 0 if len(count_list) == 0 else np.mean(count_list)
        print("    [%s] %.3f, ranging from %2d to %2d" %
              (sensor_name, mean_count, min_count, max_count))


chkpt_dir = '../../result/test/visualzier'


def load_dataset(args, site_dir):
    dataset = CASASDataset(directory=args.dataset_dir, site_dir=site_dir)
    dataset_chkpt_file = os.path.join(chkpt_dir,
                                      dataset.data_dict['name'] + ".pkl")
    if os.path.exists(dataset_chkpt_file):
        dataset = CASASDataset.load(dataset_chkpt_file)
    else:
        dataset.load_events(show_progress = True)
        dataset.save(dataset_chkpt_file)
    dataset.summary()
    return dataset


def get_observations(dataset):
    observation_chkpt_file = os.path.join(
        chkpt_dir,
        dataset.data_dict['name'] + "_observations.pkl")
    if os.path.exists(observation_chkpt_file):
        fp = open(observation_chkpt_file, "rb")
        observations = pickle.load(fp)
        fp.close()
    else:
        observations = dataset.to_observation_track()
        fp = open(observation_chkpt_file, "wb")
        pickle.dump(observations, fp, protocol=-1)
        fp.close()
    return observations


def get_sequence(dataset):
    sequence_chkpt_file = os.path.join(
        chkpt_dir,
        dataset.data_dict['name'] + "_sequence.pkl")
    if os.path.exists(sequence_chkpt_file):
        fp = open(sequence_chkpt_file, "rb")
        sequence = pickle.load(fp)
        fp.close()
    else:
        sequence = dataset.to_sensor_sequence()
        fp = open(sequence_chkpt_file, "wb")
        pickle.dump(sequence, fp, protocol=-1)
        fp.close()
    return sequence


if __name__ == "__main__":
    config_debug()
    args = parse_args()
    if args.site is not None:
        site_dir = args.site
    else:
        site_dir = None
    os.makedirs(chkpt_dir, exist_ok=True)
    dataset = load_dataset(args, site_dir)
    observations = get_observations(dataset)
    sequence = get_sequence(dataset)
    observation_summary(observations, dataset.sensor_list)
    print("Length of sequence: %d" % len(sequence))
    animate_2d_mrt_observations_onsite(dataset.site, observations,
                                       dataset.sensor_list)

    animate_2d_mrt_sequence_onsite(dataset.site, sequence, dataset.sensor_list)
    # sequence = dataset.to_sensor_sequence()
