""" Abstract Model Class for Gaussian-Mixture Probability Hypothesis Density
Filter.
"""


class GmPhdModel:
    """ Defines the parameters needed for the model to be used with
    Gaussian-Mixture Probability Hypothesis Density Multi-target Tracking
    Filter.

    Attributes:
        birth (:obj:`list`): List of GMs defining the birth probability
            across the state space.
        ps (:obj:`float`): Persistence probability (between 0 to 1)
        pd (:obj:`float`): Detection probability (between 0 to 1)
        lambda_c (:obj:`float`): Poisson distribution parameter for clutter
            process (Assume it is a Poission Point Process).
        gm_T (:obj:`float`): GM truncate threshold.
        gm_U (:obj:`float`): GM merge threshold.
        gm_Jmax (:obj:`int`): Max number of GM to track in calculation.
    """
    def __init__(self):
        print('init GMPhd')
        # multi-target birth probability density presented by a list of
        # weighted GMs
        self.birth = []

        # Persistence Probability
        self.ps = 0.98
        # Detection Probability
        self.pd = 0.98
        # False Alarm (Clutter) Poisson Parameter
        self.lambda_c = 10
        # Distribution of false alarm - usually we assume that false alarm is
        # a uniform distribution.
        self._cz = 1.

        # Parameters to control approximation of Gaussian Mixtures
        # Truncate threshold T
        self.gm_T = 0.01
        # Merge threshold U
        self.gm_U = 0.001
        # Maximum GM to track Jmax
        self.gm_Jmax = 1000

    def c(self, z):
        """ Get false alarm probability density at point :math:`z` in
        measurement space.

        Args:
            z (:obj:`numpy.ndarray`): Column vector specifying a point in
                measurement space.

        Returns:
            (:obj:`float`): Returns the probability density at point
                :math:`z` provided in the measurement space.
        """
        return self._cz
