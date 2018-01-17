""" Simple test for mayavi2 installation
"""

import numpy as np
import scipy.stats
from mayavi import mlab


if __name__ == '__main__':
    figure = mlab.figure('GMGaussian')
    #figure.scene.disable_render = True
    X, Y = np.mgrid[-1000:1000:50, -1000:1000:50]
    Z_PDF = np.zeros(X.shape, dtype=np.float)

    mean = np.array([[100.], [-100.]])
    cov = np.eye(2) * 10000
    mvnormal = scipy.stats.multivariate_normal(mean=mean.flatten(), cov=cov)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            eval_x = np.array([[X[i, j]], [Y[i, j]]])
            Z_PDF[i, j] = mvnormal.pdf(eval_x.flatten())

    scale_factor = np.max(Z_PDF)
    # mlab.surf(X, Y, f, opacity=.3, color=(1., 0, 0))
    mlab.surf(X, Y, Z_PDF * (2000 / scale_factor), opacity=.3, color=(0., 1., 0))

    mlab.outline(None, color=(.7, .7, .7), extent=[-1000, 1000, -1000, 1000, 0, 2000])
    mlab.axes(None, color=(.7, .7, .7), extent=[-1000, 1000, -1000, 1000, 0, 2000],
              ranges=[-1000, 1000, -1000, 1000, 0, scale_factor], nb_labels=6)
    #figure.scene.disable_render = False
    mlab.show()
