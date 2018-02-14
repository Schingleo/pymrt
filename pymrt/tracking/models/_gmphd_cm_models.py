from .cv_models import CVIDModel, CVModel
from .gmphd_abstract_model import GmPhdModel


class GmPhdCvModel(CVModel, GmPhdModel):
    """Constant Velocity Model for Gaussian-Mixture Probability Hypothesis
    Density Filter
    """
    def __init__(self, n):
        CVModel.__init__(self, n)
        GmPhdModel.__init__(self)


class GmPhdCvIdModel(CVIDModel, GmPhdModel):
    """Constant Velocity Model for Gaussian-Mixture Probability Hypothesis
    Density Filter with Track ID as part of state
    """
    def __init__(self, n):
        CVIDModel.__init__(self, n)
        GmPhdModel.__init__(self)
