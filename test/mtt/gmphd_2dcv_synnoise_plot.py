import os
import pickle
import logging
from pymrt.visualizer.gmphd import plot2d_gmphd_track

# Check point
chkpt_dir = '../../result/test/phd/2dcv_sd'


def config_debug():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(name)s:%(levelname)s:%(message)s',
        handlers=[logging.StreamHandler()])


def load_result():
    result_filename = os.path.join(chkpt_dir, 'gmphd_2dcv_noise_chkpt.pkl')
    result_fp = open(result_filename, 'rb')
    result_dict = pickle.load(result_fp)
    result_fp.close()
    return result_dict


if __name__ == "__main__":
    result_dict = load_result()
    gmphd_phd = result_dict['phd']
    gmphd_grid = (gmphd_phd['x'], gmphd_phd['y'])
    gmphd_prediction = result_dict['prediction']
    gmphd_truth = result_dict['truth']
    gmphd_observation = result_dict['observations']
    gmphd_model = result_dict['model']
    plot2d_gmphd_track(
        model=gmphd_model,
        grid=gmphd_grid,
        gm_s_list=gmphd_phd['s'],
        observation_list=gmphd_observation,
        prediction_list=gmphd_prediction,
        truth=gmphd_truth,
        title='2D GM-PHD Filter with synthetic data',
        contours=10,
        log_plot=True
    )