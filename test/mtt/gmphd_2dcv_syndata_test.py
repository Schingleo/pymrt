import os
import sys
import copy
import pickle
import logging
import numpy as np

from pymrt.tracking.models import GmPhdCvModel, CubicUniformPoissonClutter
from pymrt.tracking.models import sd_generate_state, sd_generate_observation
from pymrt.tracking.gmphd import gmphd_predictor, gmphd_corrector, \
    gm_pruning, gm_estimator
from pymrt.tracking.utils import gm_calculate
from mtt2dcv_bd_init import *

# Check point
chkpt_dir = '../../result/test/phd/2dcv_sd'


def config_debug():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(name)s:%(levelname)s:%(message)s',
        handlers=[logging.StreamHandler()])


def generate_2dcv_model():
    # Create 2D CV Model
    model = GmPhdCvModel(n=n)
    # Initialize Generator for state updates
    model.sigma_w = w_cov_std
    # Initialize Generator for measurement updates
    model.sigma_r = r_cov_std
    # Initialize Clutter parameters
    model.lambda_c = poisson_lambda
    model._cz = 1/(2000*2000)
    # Birth
    model.birth = model_birth
    return model


def generate_2dcv_states(model):
    return sd_generate_state(
        model=model, n_steps=n_steps, n_targets=n_targets,
        init_state=x_init, birth=birth, death=death,
        noise=False
    )


def load_2dcv_states(model):
    states_chk_point = os.path.join(chkpt_dir, "2dcv_syn_states.pkl")
    if os.path.exists(states_chk_point):
        chk_point_file = open(states_chk_point, 'rb')
        states = pickle.load(chk_point_file)
    else:
        states = generate_2dcv_states(model=model)
        chk_point_file = open(states_chk_point, 'wb')
        pickle.dump(states, chk_point_file, protocol=pickle.HIGHEST_PROTOCOL)
    chk_point_file.close()
    return states


def generate_2dcv_observation(model, truth):
    return sd_generate_observation(
        model=model, truth=truth, noise=False,
        clutter_generator=CubicUniformPoissonClutter(
            lam=poisson_lambda,
            low=-1000,
            high=1000,
            n=n
        ),
        p_d=1.0
    )


def load_2dcv_observation(model, truth):
    observation_chk_point = os.path.join(chkpt_dir, "2dcv_syn_observation.pkl")
    if os.path.exists(observation_chk_point):
        chk_point_file = open(observation_chk_point, 'rb')
        observation = pickle.load(chk_point_file)
    else:
        observation = generate_2dcv_observation(model=model, truth=truth)
        chk_point_file = open(observation_chk_point, 'wb')
        pickle.dump(observation, chk_point_file,
                    protocol=pickle.HIGHEST_PROTOCOL)
    chk_point_file.close()
    return observation


if __name__ == "__main__":
    config_debug()
    os.makedirs(chkpt_dir, exist_ok=True)
    gmphd_model = generate_2dcv_model()
    gmphd_states = load_2dcv_states(gmphd_model)
    gmphd_observation = load_2dcv_observation(model=gmphd_model,
                                              truth=gmphd_states)
    #
    max_x = np.max([
        np.max([x[0, 0] for x in gmphd_observation[j]])
        for j in range(len(gmphd_observation))
    ])
    min_x = np.min([
        np.min([x[0, 0] for x in gmphd_observation[j]])
        for j in range(len(gmphd_observation))
    ])
    max_y = np.max([
        np.max([x[1, 0] for x in gmphd_observation[j]])
        for j in range(len(gmphd_observation))
    ])
    min_y = np.min([
        np.min([x[1, 0] for x in gmphd_observation[j]])
        for j in range(len(gmphd_observation))
    ])
    # GMPHD 3D plot grid
    grid_x, grid_y = np.mgrid[min_x:max_x:20, min_y:max_y:20]
    # Current step
    gmphd_step = 0
    # Total steps
    gmphd_num_steps = n_steps
    # Current GM list
    gmphd_gm_list = []
    # Prediction
    gmphd_prediction = []
    # Calculated PHD in space
    gmphd_phd = {
        'x': grid_x,
        'y': grid_y,
        's': [],
        'gms': []
    }

    # Start GMPHD
    for i in range(gmphd_num_steps):
        sys.stdout.write('Step %d\r\n' % i)
        gm_list = gmphd_predictor(model=gmphd_model,
                                  gm_list=gmphd_gm_list)
        gm_list = gmphd_corrector(model=gmphd_model,
                                  gm_list=gm_list,
                                  observation=gmphd_observation[i])
        gm_list = gm_pruning(gm_list=gm_list,
                             T=gmphd_model.gm_T,
                             U=gmphd_model.gm_U,
                             C=gmphd_model.gm_Jmax)
        # Get Prediction
        gmphd_prediction.append(gm_estimator(gm_list))

        # Save temporary result
        gmphd_phd['s'].append(gm_calculate(
            gm_list=gm_list,
            grid=(grid_x, grid_y)
        ))

        gmphd_phd['gms'] = copy.deepcopy(gm_list)

        # Update GM_LIST
        gmphd_gm_list = gm_list

        print('Target spotted %d' % len(gmphd_prediction[i]))

    result_filename = os.path.join(chkpt_dir, 'gmphd_2dcv_chkpt.pkl')
    result_fp = open(result_filename, 'wb')
    pickle.dump({
        'model': gmphd_model,
        'phd': gmphd_phd,
        'prediction': gmphd_prediction,
        'truth': gmphd_states,
        'observations': gmphd_observation
    }, result_fp, protocol=pickle.HIGHEST_PROTOCOL)
    result_fp.close()
