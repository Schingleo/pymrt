import copy
import logging
import numpy as np
import scipy.stats
from .utils import GaussianComponent


logger = logging.getLogger(__name__)


def gmphd_predictor(model, gm_list):
    """GM-PHD Predictor

    This GM-PHD predictor process updates the list of GMs at time k based on
    dynamic models described in `model`.
    During this process, the number of GMs will not change. So, to save the
    time and space, GMs are updated in place.
    At the end of predictor, the birth-term is added to the front of gm_list.

    Args:
        model (:obj:`~pymrt.tracking.models.GmPhdCvModel`): Multi-target Model
            with GM representations.
        gm_list (:obj:`list` of :obj:`GaussianMixtureComponents`): Group of
            GM Components representing the PHD function of the multi-target
            system before predictor.

    Returns:
        gm_list (:obj:`list`): Group of GM Components
            (:obj:`~pymrt.tracking.utils.GaussianComponent`) representing the
            PHD function of the multi-target system after predictor.
    """
    for gm in gm_list:
        gm.kalman_update(model._F, model._Q)
        gm.weight = gm.weight * model.ps
    gm_list = copy.deepcopy(model.birth) + gm_list
    return gm_list


def gmphd_corrector(model, gm_list, observation):
    """GM-PHD Corrector

    This GM-PHD corrector process updates the PHD of the multi-target system
    (after predictor) based on the measurement model described in ``model``.
    At first, the weights of existing GMs are adjusted based on detection
    probability.
    Then, according to mathematical equations, there are substantial growth
    to the number of GMs.
    List of GMs will be recreated.

    Args:
        model (:obj:`pymrt.tracking.models.GmPhdCvModel`): Multi-target Model
            with GM representations.
        gm_list (:obj:`list` of :obj:`GaussianMixtureComponents`): Group of GM
            Components representing the PHD function of the multi-target
            system after predictor before corrector.
        observation (:obj:`list` of :obj:`np.ndarray`): List of measurements
            observed at current time step.

    Returns:
        gm_list (:obj:`list` of :obj:`GaussianMixtureComponents`): Group of GM
            Components representing the PHD function of the multi-target
            system after corrector.
    """
    # According to GM-PHD Corrector Derivation
    #
    # .. math::
    #     D_{k|k}(x) = (1 - p_D) v_{k|k-1}(x) +
    #     \sum_{z \in Z_k} \sum_{j=1}^{J_{k|k-1}} w_{k|k}^{(j)}
    #     \mathcal{N}\left(x; m_{k|k}^{(j)}, P_{k|k}^{(j)}\right)
    #
    # Term :math:`(1 - p_D) v_{k|k-1}(x)` is implemented by adjusting the
    # weight of GMs in the ``gm_list`` directly.
    new_gm_list = []
    if len(observation) > 0:
        # The second term is quite complicated to implement.
        # First, it tries to create temporary values needed.
        gm_len = len(gm_list)                 # :math:`J_{k|k-1}`
        z_len = len(observation)              # :math:`\left|Z_k\right|`
        qz = np.zeros((gm_len, z_len))        # Stores :math:`q_k^{(j)}(z)`
        m_k = np.zeros((gm_len, z_len, model.x_dim, 1))
        P_k = np.zeros((gm_len, model.x_dim, model.x_dim))
        w_qz_int = np.zeros((z_len,))
        w_k = np.zeros((gm_len, z_len))
        for j in range(gm_len):
            mk1_j = gm_list[j].mean    # :math:`m_{k|k-1}^{(j)}`
            Pk1_j = gm_list[j].cov     # :math:`P_{k|k-1}^{(j)}`
            # :math:`H_k m_{k|k-1}^{(j)}`
            hm = np.dot(model._H, mk1_j)
            # :math:`H_k P_{k|k-1}^{(j)}H_k^T + R_{k}`
            r_hph = model._R + np.dot(model._H,
                                      np.dot(Pk1_j, model._H.T))
            K_k = np.dot(Pk1_j, np.dot(model._H.T,
                                                np.linalg.inv(r_hph)))
            # Implementing w_k
            P_k[j, :, :] = np.dot(np.eye(model.x_dim) - np.dot(K_k, model._H),
                                  Pk1_j)
            # Vs is the upper triangle matrix of Cholesky decomposition
            # According to definition, :math:`S = Vs^T \cdot Vs`
            # :math:`S^{-1} = Vs^{-1} \cdot (Vs^{-1})^T`
            # Vs = np.linalg.cholesky(S)
            # det_S = np.prod(np.diag(Vs))
            # inv_Vs = np.linalg.inv(Vs)
            # inv_S = np.dot(inv_Vs, inv_Vs.T)
            # inv_S = np.linalg.inv(S)
            # :math:`K = P \cdot H \cdot (H \cdot P \cdot H^T + R) ^ {-1}
            # K = np.dot(cur_gm.cov, np.dot(model.H.T, inv_S))
            # Calculate new P
            for i in range(z_len):
                z = observation[i]      # :math:`z`
                m_k[j, i, :, :] = mk1_j + \
                                  np.dot(K_k, z - hm)
                qz[j, i] = scipy.stats.multivariate_normal.pdf(
                    x=z.flatten(),
                    mean=hm.flatten(),
                    cov=r_hph.T)
        for i in range(z_len):
            z = observation[i]  # :math:`z`
            # Clutter constant
            clutter_k = model.lambda_c * model.c(z)
            for j in range(gm_len):
                w_qz_int[i] += qz[j, i] * gm_list[j].weight
            for j in range(gm_len):
                wk1_j = gm_list[j].weight
                gm_w_numerator = model.pd * wk1_j * qz[j, i]
                gm_w = gm_w_numerator / (clutter_k + model.pd * w_qz_int[i])
                new_gm = GaussianComponent(
                    mean=m_k[j, i, :, :],
                    cov=P_k[j, :, :],
                    weight=gm_w,
                    n=model.x_dim
                )
                new_gm_list.append(new_gm)
            logger.debug('Update GMs for measurement %d: %s' %
                         (i, str(observation[i].tolist())))

    # The not updated part.
    for gm in gm_list:
        gm.weight = gm.weight * (1-model.pd)
    return gm_list + new_gm_list


def gm_pruning(gm_list, T, U, C):
    """ Prunning List of Gaussian Mixture

    Args:
        gm_list:
        T (:obj:`float`): Weight Threshold.
        U (:obj:`float`): Merge Threshold.
        C (:obj:`float`): Count Threshold.
    """
    # Sort according to weight
    sorted_gms = sorted(gm_list, key=lambda gm: gm.weight, reverse=True)
    len_gms = len(sorted_gms)
    handled_indices = {}
    # Truncating based on weight threshold.
    truncated_gms = []
    for i in range(len_gms):
        if sorted_gms[i].weight > T:
            truncated_gms.append(sorted_gms[i])
    sorted_gms = truncated_gms
    # Merge if distance between two Gaussian Component is smaller than :math:`U`
    merged_gms = []
    len_gms = len(sorted_gms)
    for i in range(len_gms):
        if i not in handled_indices:
            merge_list = []
            for j in range(i+1, len_gms):
                 # Find the closest and merge
                if j not in handled_indices:
                    if gmDistance(sorted_gms[i], sorted_gms[j]) < U:
                        merge_list.append(j)
                        handled_indices[j] = True
            # Now Merge
            if len(merge_list) > 0:
                new_weight = sorted_gms[i].weight
                new_mean = sorted_gms[i].mean * sorted_gms[i].weight
                new_cov = sorted_gms[i].cov * sorted_gms[i].weight
                for j in merge_list:
                    new_weight += sorted_gms[j].weight
                    new_mean += sorted_gms[j].mean * sorted_gms[j].weight
                    mean_diff = sorted_gms[i].mean - sorted_gms[j].mean
                    new_cov += sorted_gms[j].weight * np.dot(mean_diff, mean_diff.T)
                new_mean = new_mean / new_weight
                new_cov = new_cov / new_weight
                merged_gms.append(GaussianComponent(
                    mean=new_mean,
                    weight=new_weight,
                    cov=new_cov,
                    n=sorted_gms[i].n
                ))
            else:
                merged_gms.append(sorted_gms[i])
            handled_indices[i] = True
    # Prunning based on the maximum length of GMs to keep
    sorted_gms = sorted(merged_gms, key=lambda gm: gm.weight, reverse=True)
    if len(truncated_gms) > C:
        return sorted_gms[:C]
    else:
        return sorted_gms


def gmDistance(gm1, gm2):
    """ Distance between two Gaussian Mixture Component
    """
    mean_diff = gm1.mean - gm2.mean
    return np.dot(mean_diff.T, np.dot(np.linalg.inv(gm1.cov), mean_diff))


def gm_estimator(gm_list):
    """ GM-PHD State Estimator

    Args:
        gm_list (:obj:`list`): List of Gaussian components representing
            posterior probability hypothesis density of multiple targets.

    Returns:
        (:obj:`list`): List of estimated target states.
    """
    targets = []
    for gm in gm_list:
        if gm.weight > 0.5:
            targets.append(gm.mean)
    return targets


def gm_id_estimator(gm_list):
    """ GM-PHD State Estimator

    Args:
        gm_list (:obj:`list`): List of Gaussian components representing
            posterior probability hypothesis density of multiple targets.

    Returns:
        (:obj:`list`): List of estimated target states.
    """
    if not hasattr(gm_id_estimator, 'track_id'):
        setattr(gm_id_estimator, 'track_id', 0)
    targets = []
    for gm in gm_list:
        if gm.weight > 0.5:
            targets.append(gm.mean)
            if gm.mean[-1, :] == 0:
                gm_id_estimator.track_id += 1
                gm.mean[-1, :] = gm_id_estimator.track_id
    return targets
