""" CASASDataset test

This test file loads a recorded sensor events from a CASAS smart home site,
prints out the summary and detects the abnormality of the sensor events.
The abnormalities in the dataset is exported to an Microsoft Excel (xlsx) file.
"""

import os
import logging
import argparse
from pymrt.casas import CASASDataset

result_dir = '../../result/test/casas'


def config_debug():
    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(asctime)s] %(name)s:%(levelname)s:%(message)s',
        handlers=[logging.StreamHandler()])


def parse_args():
    parser = argparse.ArgumentParser(description="Load and Plot Casas Site")
    parser.add_argument("dataset_dir",
                        help="Directory where CASAS dataset is stored")
    parser.add_argument("--site",
                        help="Directory where CASAS site meta-data is stored")
    return parser.parse_args()


if __name__ == "__main__":
    config_debug()
    args = parse_args()
    if args.site is not None:
        site_dir = args.site
    else:
        site_dir = None
    dataset = CASASDataset(directory=args.dataset_dir, site_dir=site_dir)
    dataset.load_events(show_progress=True)
    dataset.summary()
    os.makedirs(result_dir, exist_ok=True)
    dataset.check_event_list(
        xlsx_fname=os.path.join(result_dir,
                                dataset.get_name() + '_report.xlsx')
    )
