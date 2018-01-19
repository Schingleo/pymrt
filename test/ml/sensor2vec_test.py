""" Test sensor to vector embedding
"""

import os
import logging
import argparse
import pickle
from pymrt.casas.dataset import CASASDataset
from pymrt.ml.sensor2vec import *
from pymrt.utils.visualizer import *


def config_debug():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(name)s:%(levelname)s:%(message)s',
        handlers=[logging.StreamHandler()])


def parse_args():
    parser = argparse.ArgumentParser(description="Load and Plot Casas Site")
    parser.add_argument("dataset_dir",
                        help="Directory where CASAS dataset is stored")
    parser.add_argument("--site",
                        help="Directory where CASAS site meta-data is stored")
    return parser.parse_args()


# Check point
chkpt_dir = '../../result/test/ml'


def load_dataset(args, site_dir):
    dataset = CASASDataset(directory=args.dataset_dir, site_dir=site_dir)
    dataset_chkpt_file = os.path.join(chkpt_dir,
                                      dataset.data_dict['name'] + ".pkl")
    if os.path.exists(dataset_chkpt_file):
        dataset = CASASDataset.load(dataset_chkpt_file)
    else:
        dataset.load_events(show_progress=True)
        dataset.save(dataset_chkpt_file)
    dataset.summary()
    return dataset


def get_sequence(dataset):
    sequence_chkpt_file = os.path.join(
        chkpt_dir,
        dataset.data_dict['name'] + "_sequence.pkl"
    )
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


def get_observations(dataset):
    observation_chkpt_file = os.path.join(
        chkpt_dir,
        dataset.data_dict['name'] + "_observations.pkl"
    )
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


def get_sensor_embedding(dataset, sequence):
    embedding_chkpt_file = os.path.join(
        chkpt_dir,
        dataset.data_dict['name'] + "_embedding.pkl"
    )
    if os.path.exists(embedding_chkpt_file):
        fp = open(embedding_chkpt_file, 'rb')
        embeddings, distance_matrix = pickle.load(fp)
        fp.close()
    else:
        embeddings, distance_matrix = sensor2vec(
            len(dataset.sensor_list), sequence,
            embedding_size=3, skip_window=5, num_skips=8, learning_rate=0.05,
            num_steps=100000)
        fp = open(embedding_chkpt_file, 'wb+')
        pickle.dump((embeddings, distance_matrix), fp)
        fp.close()
    return embeddings, distance_matrix


if __name__ == "__main__":
    config_debug()
    args = parse_args()
    if args.site is not None:
        site_dir = args.site
    else:
        site_dir = None
    os.makedirs(chkpt_dir, exist_ok=True)
    dataset = load_dataset(args, site_dir)
    sequence = get_sequence(dataset)
    observations = get_observations(dataset)
    embeddings, distance_matrix = get_sensor_embedding(dataset, sequence)
    plot_3d_sensor_vector(dataset, embeddings)
    animate_3d_mrt_observations(dataset, embeddings, observations)
