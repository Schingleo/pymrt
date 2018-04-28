import copy
import logging
import numpy as np
import scipy.stats
from scipy.spatial import distance
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


def gm_id_cluster(gm_list, z_dim):
    """GM-PHD State Estimator by clustering algorithm

    In cases where the error is higher and no single GM has a weight that goes
    beyond 0.5, cluster algorithm is needed to assign track ID to each GMs.

    Args:
        gm_list (:obj:`list`): List of Gaussian components representing
            posterior probability hypothesis density of multiple targets.
        z_dim (:obj:`int`): The dimensionality of the measurement space.

    Returns:
        a `list` of Gaussian components with track ID assigned using the
        clustering algorithm.
    """
    if not hasattr(gm_id_cluster, 'max_id'):
        setattr(gm_id_cluster, 'max_id', 1)
    max_id = getattr(gm_id_cluster, 'max_id')
    track_gms_dict = {}
    # Organize Gms by track ID
    for gm in gm_list:
        assert(isinstance(gm, GaussianComponent))
        track_id = int(gm.mean.flatten()[-1])
        if track_id not in track_gms_dict:
            track_gms_dict[track_id] = []
        track_gms_dict[track_id].append(gm)
    # Calculate weight and determine if it needs clustering
    for track_id in track_gms_dict:
        expected_card = 0.
        for gm in track_gms_dict[track_id]:
            assert (isinstance(gm, GaussianComponent))
            expected_card += gm.weight
        # Round the expected cardinality. If greater than 1, form new tracks.
        num_tracks = int(round(expected_card))
        if num_tracks > 1 or (track_id == 0 and num_tracks > 0):
            gm_list, max_id = _gm_cluster_assign_id(
                gm_list=track_gms_dict[track_id],
                track_id=track_id,
                num_tracks=num_tracks,
                weight_threshold=expected_card/num_tracks,
                z_dim=z_dim,
                max_id=max_id
            )
    setattr(gm_id_cluster, 'max_id', max_id)
    return gm_list


def _gm_cluster_assign_id(gm_list, track_id, num_tracks, weight_threshold,
                          z_dim, max_id, max_iteration=1000):
    """The cluster algorithm that assign a new ID to the track

    Args:
        gm_list (:obj:`list`): List of ``GaussianComponent`` representing
            current multi-target PHD density.
        track_id (:obj:`int`): Current track id.
        num_tracks (:obj:`int`): The number of tracks that this list of Gaussian
            components need to split into.
        weight_threshold (:obj:`float`): Initial weight threshold for each newly
            spawned track.
        z_dim (:obj:`int`): The dimensionality of measurement space.
        max_id (:obj:`int`): The next track ID number that can be assigned.
        max_iteration (:obj:`int`): Max number of iterations in case that the
            clustering algorithm does not converge and oscillates.

    Returns:
        A `list` of Gaussian components with updated track ID and the next track
        ID that can be assigned to new tracks in the future.
    """
    clusters_mean = np.random.uniform(0, 1, (num_tracks, z_dim))
    previous_clusters_mean = None
    cluster_gms = [[] for i in range(num_tracks)]

    count = 0
    while np.any(clusters_mean != previous_clusters_mean) and \
            count < max_iteration:
        previous_clusters_mean = np.copy(clusters_mean)
        # There n_tracks means, calculate the distance between each track,
        # and sorted from high to low
        gm_distance_matrix = _gm_cluster_distance(gm_list=gm_list,
                                                  clusters_mean=clusters_mean,
                                                  num_tracks=num_tracks,
                                                  z_dim=z_dim)
        # Assign GM to each mean where the weight of each cluster equals or
        # just higher than the weight threshold.
        cluster_gms = _gm_group_cluster(gm_list=gm_list,
                                        distance_matrix=gm_distance_matrix,
                                        weight_threshold=weight_threshold)
        # Update mean
        for i in range(num_tracks):
            new_mean = np.zeros((z_dim,), dtype=np.float32)
            new_weight = 0.
            for gm in cluster_gms[i]:
                new_mean += gm.mean.flatten()[0:z_dim] * gm.weight
                new_weight += gm.weight
            if new_weight == 0.:
                new_weight = 1
            clusters_mean[i, :] = new_mean / new_weight
        # Update count
        count += 1

    # Assign ID to each cluster
    for i in range(num_tracks):
        # For every new track, start counting with max_id
        if track_id == 0 and i == 0:
            for gm in cluster_gms[i]:
                gm.mean[-1, :] = max_id
            max_id += 1
        elif i != 0:
            for gm in cluster_gms[i]:
                gm.mean[-1, :] = max_id
            max_id += 1

    return gm_list, max_id


def _gm_cluster_distance(gm_list, clusters_mean, num_tracks, z_dim):
    """Calculate the distance between each Gaussian component and the center
    of each cluster.

    Args:
        gm_list (:obj:`list`): List of Gaussian components.
        clusters_mean (:obj:`numpy.ndarray`): An array in shape
            `[num_cluster, z_dim]` representing the center of all clusters.
        num_tracks (:obj:`int`): Number of tracks (or clusters).
        z_dim (:obj:`int`): The dimensionality of the measurement space.

    Returns:
        An array of shape `[num_gm, num_cluster]` where each element in the
        array represents the distance of a Gaussian component to the cluster.
    """
    gm_distance_array = np.zeros((len(gm_list), num_tracks), np.float32)
    for i, gm in enumerate(gm_list):
        for j in range(num_tracks):
            gm_distance_array[i, j] = distance.euclidean(
                gm.mean.flatten()[0:z_dim], clusters_mean[j, :]
            )
    return gm_distance_array


def _gm_group_cluster(gm_list, distance_matrix, weight_threshold):
    """Group each gm to cluster based on distance and a weight threshold.

    Args:
        gm_list (:obj:`list`): List of Gaussian components.
        distance_matrix (:obj:`numpy.ndarray`): An array of shape
            `[num_gm, num_cluster]` where each element in the array represents
             the distance of a Gaussian component to the cluster.
        weight_threshold (:obj:`float`): The average weight of each cluster.

    Returns:
        A `list` of `list` of Gaussian mixtures that belongs to each cluster.
    """
    gms_to_assign = set(range(len(gm_list)))
    num_clusters = distance_matrix.shape[1]
    gm_id_in_cluster = [[] for i in range(num_clusters)]
    cluster_weight = [0. for i in range(num_clusters)]
    choice = 0
    gms_sorted_cluster_id = np.argsort(distance_matrix, axis=1)
    # Assign gms to each cluster based on the distance between the center of
    # each cluster and the mean of each Gaussian components with the constraint
    # that the total weight of each cluster does not exceed weight_threshold.
    while len(gms_to_assign) > 0 and choice < num_clusters:
        temp_cluster_gms = [[] for i in range(num_clusters)]
        # Group the gms to be assigned by minimum distance
        for gm_id in gms_to_assign:
            temp_cluster_gms[gms_sorted_cluster_id[gm_id, choice]].append(gm_id)
        # Sort the gms of each cluster by distance (small to large)
        for i in range(num_clusters):
            temp_cluster_gms[i] = sorted(
                temp_cluster_gms[i],
                key=lambda gm_id: distance_matrix[gm_id, i]
            )
            for gm_id in temp_cluster_gms[i]:
                temp_weight = cluster_weight[i] + gm_list[gm_id].weight
                if temp_weight < weight_threshold:
                    cluster_weight[i] = temp_weight
                    gms_to_assign.remove(gm_id)
                    gm_id_in_cluster[i].append(gm_id)
                else:
                    break
        # Add Choice
        choice += 1
    # Certainly, it cannot finish assign all GMs to each cluster, thus,
    # we need to take care of the rest of GMs that has left behind and assign
    # them to each cluster. Ideally, they should be assigned according to
    # distance between center of the cluster and the Gaussian component.
    # However, a simple approximation can be used to assign the rest of
    # Gaussian components to the nearest cluster while the total weight of
    # that cluster is smaller than 1..
    for gm_id in gms_to_assign:
        for j in range(num_clusters):
            cluster_id = gms_sorted_cluster_id[gm_id, j]
            temp_weight = cluster_weight[cluster_id] + gm_list[gm_id].weight
            if temp_weight < 1.5:
                cluster_weight[cluster_id] += gm_list[gm_id].weight
                gm_id_in_cluster[cluster_id].append(gm_id)
                break
    return [
        [gm_list[gm_id] for gm_id in gm_id_in_cluster[cluster_id]]
        for cluster_id in range(num_clusters)
    ]
