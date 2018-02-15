Dynamic Models for Multi-target Tracking
========================================

Namespace ``pymrt.tracking.models`` includes multiple classes for commonly used
dynamic models. They serve as the basis of state transition modelling for
multi-target tracking algorithm.

GM-based Constant Velocity Model
--------------------------------

:class:`pymrt.tracking.models.CVModel` provides a base class for constant
velocity maneuvering object in n-dimensional space.

The model is composed of the following parts of information:

State Space
^^^^^^^^^^^

The state of an object maneuvering in a n-dimensional space with constant
velocity can be expressed as an array composed of its location in the space
and velocity at a given time.

.. math::
    x = [s_{d_1}, s_{d_2}, ..., s_{d_n}, v_{d_1}, v_{d_2}, ..., v_{d_n}]^T

The state space where :math:`x` is drawn from is usually
:math:`\mathbb{R}^{2n}`.

State Update
^^^^^^^^^^^^

If the object is moving according to a constant velocity in the n-dimensional
space, then the new state (after time :math:`t`) can be expressed as

.. math::
    x_{t+1} = [s_{d_1} + v_{d_1} t,
               s_{d_2} + v_{d_2} t, ...,
               s_{d_n} + v_{d_n} t, v_{d_1}, v_{d_2}, ..., v_{d_n}]^T

In matrix format, we can rewrite it as

.. math::
    x_{t+1} = F \cdot x_{t}

where

.. math::
    F = \left[
    \begin{matrix}
        I_n & I_n t \\
        0   & I_n
    \end{matrix}
    \right]

:math:`I_n` here is identity matrix of n-dimension.

State Error Estimation
^^^^^^^^^^^^^^^^^^^^^^

Well, there is always error associated with a model, and to simulate the error,
the following error term (only first order is considered) is added.

Given disturbance :math:`w = \left[w_{d_0}, w_{d_1}, ..., w_{d_n} \right]^T`,
the state updated with error estimation can be written in matrix form as
follows:

.. math::
    x_{t+1} = F \cdot x_{t} + G \cdot w

where

.. math::
    F = \left[
    \begin{matrix}
        I_n & I_n t \\
        0   & I_n
    \end{matrix}
    \right]

and

.. math::
    G = \left[
    \begin{matrix}
    \frac{1}{2}t^2I_n\\
    tI_n
    \end{matrix}
    \right]

Measurement
^^^^^^^^^^^

Measurement, sometimes also referred to as observation, of a target at state
:math:`x` can be derived with the following equation

.. math::
    z_{t} = H \cdot x{t}

:math:`H` is called measurement multiplier, and :math:`z_t` is the
observation of the target at time :math:`t`.
Usually, only the space location of target can be measured, so the
measurement multipler :math:`H` is

.. math::
    H = \left[
        \begin{matrix}
        I_n \\
        0_n
        \end{matrix}
    \right]

Measurement Error Estimation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Measurement will have error. The error is modeled by a uncertainty measure
around the exact measurement. In Gaussian Mixture model, it is represented by
a multi-variant Gaussian covariance matrix :math:`D`.

If each dimension are independent of each other,

.. math::
    D = \sigma^2 I_n

GM-based Constant Velocity Model with Track ID
-----------------------------------------------

:class:`pymrt.tracking.models.CVIDModel` provides a base class for constant
velocity maneuvering object in n-dimensional space with track ID embedded in
state vector.

State Space
^^^^^^^^^^^

In order to accomplish data-track association during target tracking, the
track ID needs to be append to state vector.
At a given time, the state vector is described as follows

.. math::
    x = [s_{d_0}, s_{d_1}, ..., s_{d_n}, v_{d_0}, v_{d_1}, ..., v_{d_n}, tid]^T

The state space is usually :math:`\mathbb{R}^{2n}\times \mathbb{N}`.

State Update
^^^^^^^^^^^^

The state update, without considering spawning (one object separated into two),
can be expressed as follows:

.. math::
    x_{t+1} = F \cdot x_{t}

where

.. math::
    F = \left[
    \begin{matrix}
        I_n & I_n t & 0 \\
        0   & I_n & 0 \\
        0   & 0   & 1
    \end{matrix}
    \right]

State Error Estimation
^^^^^^^^^^^^^^^^^^^^^^

Given disturbance :math:`w = \left[w_{d_0}, w_{d_1}, ..., w_{d_n} \right]^T`,
the state updated with error estimation can be written in matrix form as
follows:

.. math::
    x_{t+1} = F \cdot x_{t} + G \cdot w

where

.. math::
    F = \left[
    \begin{matrix}
        I_n & I_n t & 0 \\
        0   & I_n & 0\\
        0   & 0   & 1
    \end{matrix}
    \right]

and

.. math::
    G = \left[
    \begin{matrix}
    \frac{1}{2}t^2I_n\\
    tI_n \\
    0
    \end{matrix}
    \right]

Measurement
^^^^^^^^^^^

Measurement equation of the model can be expressed as:

.. math::
    z_{t} = H \cdot x_{t}

where

.. math::
    H = \left[
        \begin{matrix}
        I_n \\
        0_n \\
        0
        \end{matrix}
    \right]


Other Model Parameters for MTT
------------------------------

In additional to the model parameters mentioned above, there are additional
parameters defined for the tracking algorithm to use.

Target Birth
^^^^^^^^^^^^

In Gaussian-Mixture PHD algorithm, target birth is represented by a series of
Gaussian Mixtures, where the sum of all GMs represents the possibility of
target birth in the state space.

Target Persistence
^^^^^^^^^^^^^^^^^^

Target persistence is modeled by parameter :math:`p_s`.
:math:`p_s` is the probability of the target being persistent into the next
time step.
:math:`1-p_s` gives the probability of the target death at the current time
step.

Target Detection
^^^^^^^^^^^^^^^^

In some cases, targets are not picked up by the observers.
Parameter :math:`p_d` gives the probability of the detection of a target in
at current time step.

Clutter Process
^^^^^^^^^^^^^^^

Observations are also noisy. All the false-alarms in the observation are
modeled by the clutter process.
It is usually safe to assume that the clutter process is a Poison Point
Process with parameter :math:`\lambda_c`, i.e. the number of false-alarms
follows Poisson distribution of :math:`\lambda_c`, where the exact observed
position of those false-alarms follows a spacial distribution :math:`c(z)`.
:math:`c(z)` can be represented as a series of Gaussian Mixtures or a uniform
distribution across space (constant).

Gaussian Mixture Approximation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

While propagating Gaussian Mixtures over time causes an exponential growth of
the number of GMs, the following parameters are set for the trade-off between
computation and accuracy.

:math:`gm_T` is the truncating threshold. If the weight of a Gaussian Mixture
is below the threshold, it is considered insignificant and can be safely
discarded without sacrificing a lot of tracking performance.

:math:`gm_U` defines the merging threshold of GMs.
If multiple GMs are centered at about the same location, the summation of two
GMs can be approximate by a single Gaussian Mixture.

:math:`gm_{Jmax}` sets the maximum number of Gaussian Mixture to track in the
process.

Appendix
--------

Derivation of Error Estimation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section, we provide derivation of the constant velocity model state
update and state error estimation. Here, we consider 2D and 3D models for
convenience of demonstration.

If the target moves in a 2D space, then the state space has a dimensionality
of 4: :math:`\left[x, y, v_x, v_y\right]^T`.
If the target moves in a 3D space, then the state space has a
dimensionality of 6: :math:`\left[x, y, z, v_x, v_y, v_z\right]^T`.
(:math:`z` in this case represent a dimension in state space).

The linear constant velocity model provides a linear relationship between
the target states before and after a time step.
Mathematically,

.. math::
   x_{k+1} = F \cdot x_k

Now, consider a 2D constant velocity model.
At time step :math:`k+1` with time interval :math:`t` , :math:`v_x` ,
:math:`v_y` stays the same, but :math:`x` and :math:`y` are both increased by
:math:`v_x \cdot t` and :math:`v_y \cdot t` respectively.
Formally,

.. math::
    \left[\begin{array}{c}x_{k+1}\\ y_{k+1}\\ v_x\\ v_y\end{array}\right] =
    \left[\begin{array}{cccc}
        1 & 0 & t & 0\\
        0 & 1 & 0 & t\\
        0 & 0 & 1 & 0\\
        0 & 0 & 0 & 1\end{array}\right] \cdot
    \left[\begin{array}{c}x_k\\ y_k\\ v_x\\ v_y\end{array}\right]

In another word,

.. math::
    F_{2D} = \left[
    \begin{array}{cccc}
        1 & 0 & t & 0\\
        0 & 1 & 0 & t\\
        0 & 0 & 1 & 0\\
        0 & 0 & 0 & 1
    \end{array}\right]

The equation above provides an ideal physical constant velocity model.
However, in real-world applications, due to various environmental noise such
as  air perturbation, an additive error term needs to be introduced.

In 4D CV model, we assume that there is a white noise  :math:`\mathbb{w}_k =
\left[w_x, w_y\right]^T` added on top of the velocity of the particle that
causes deviation between the actual model and the ideal physical model.

Thus, the following physical dynamic equation holds:

.. math::
    \begin{aligned}
    \frac{\partial \mathbb{x}(t)}{\partial t} = {} &
        \frac{\partial}{\partial t}
        \left[\begin{array}{c}x\\ y\\ v_x\\ v_y\end{array}\right]
        = \left[\begin{array}{c}v_x\\ v_y\\ a_x\\ a_y\end{array}\right]\\
    = {} & \left[\begin{array}{cccc}
                 0 & 0 & 1 & 0\\
                 0 & 0 & 0 & 1\\
                 0 & 0 & 0 & 0\\
                 0 & 0 & 0 & 0\end{array}\right] \cdot
           \left[\begin{array}{c}x\\ y\\ v_x\\ v_y\end{array}\right] +
           \left[\begin{array}{cc}
                 0 & 0 \\
                 0 & 0 \\
                 1 & 0 \\
                 0 & 1 \\
                 \end{array}\right] \cdot
           \left[\begin{array}{c}w_x\\ w_y\end{array}\right]\\
    = {} & A \cdot \mathbb{x}(t) + B \cdot \mathbb{w}(t)
    \end{aligned}

where

.. math::

    A = \left[\begin{array}{cccc}
                 0 & 0 & 1 & 0\\
                 0 & 0 & 0 & 1\\
                 0 & 0 & 0 & 0\\
                 0 & 0 & 0 & 0\end{array}\right],
    \textrm{and}
    B = \left[\begin{array}{cc}
                 0 & 0 \\
                 0 & 0 \\
                 1 & 0 \\
                 0 & 1 \\
                 \end{array}\right]


Solve the differential equation with Laplace transformation:

.. math::
    \begin{aligned}
    \frac{\partial \mathbb{x}}{\partial t} = {} &
        A \cdot \mathbb{x} + B \cdot \mathbb{w}\\
    s \mathbb{x}(s) - \mathbb{x}(0) = {} &
        A \cdot \mathbb{x}(s) + B \cdot \mathbb{w}(s)\\
    \left(s\cdot I - A\right) \mathbb{x}(s) = {} &
        \mathbb{x}(0) + B \cdot \mathbb{w}(s)\\
    \mathbb{x}(s) = {} & \left(s\cdot I - A\right)^{-1} \mathbb{x}(0) +
                    \left(s\cdot I - A\right)^{-1} \cdot B \cdot \mathbb{w}(s)\\
    \end{aligned}

.. math::
    \left(s\cdot I - A\right)^{-1} = \left[\begin{array}{cccc}
                 s & 0 & -1 & 0\\
                 0 & s & 0 & -1\\
                 0 & 0 & s & 0\\
                 0 & 0 & 0 & s\end{array}\right]^{-1}
    = \left[\begin{array}{cccc}
                 \frac{1}{s} & 0 & \frac{1}{s^2} & 0\\
                 0 & \frac{1}{s} & 0 & \frac{1}{s^2}\\
                 0 & 0 & \frac{1}{s} & 0\\
                 0 & 0 & 0 & \frac{1}{s}\end{array}\right]

Now, take inverse Laplace transformation,

.. math::
    \begin{aligned}
    \mathbb{x}(t) = {} & \left[\begin{array}{cccc}
                 u(t) & 0 & t\cdot u(t) & 0\\
                 0 & u(t) & 0 & t \cdot u(t)\\
                 0 & 0 & u(t) & 0\\
                 0 & 0 & 0 & u(t)\end{array}\right] \cdot \mathbb{x}(0)\\
                 & +
                 \int_0^t \left[\begin{array}{cccc}
                 u(t-\tau) & 0 & (t - \tau) \cdot u(t - \tau) & 0\\
                 0 & u(t - \tau) & 0 & (t - \tau) \cdot u(t - \tau)\\
                 0 & 0 & u(t - \tau) & 0\\
                 0 & 0 & 0 & u(t - \tau)\end{array}\right] \cdot
                 \left[\begin{array}{cc}
                 0 & 0 \\
                 0 & 0 \\
                 1 & 0 \\
                 0 & 1 \\
                 \end{array}\right] \cdot \mathbb{w}(\tau) d\tau
    \end{aligned}

As we are deriving discrete-time motion model, let :math:`t > 0` be the time
interval of two continuous sample.
Thus, the step function :math:`u(t) = 1`, and the :math:`u(t - \tau) = 1` as
:math:`\tau < t`.
Moreover, assume that with in the time step, :math:`\mathbb{w}(\tau) = w` is
constant.

.. math::
    \begin{aligned}
    \mathbb{x}(t) = {} & \left[\begin{array}{cccc}
                 1 & 0 & t & 0\\
                 0 & 1 & 0 & t\\
                 0 & 0 & 1 & 0\\
                 0 & 0 & 0 & 1\end{array}\right] \cdot \mathbb{x}(0) +
                 \int_0^t \left[\begin{array}{cccc}
                 1 & 0 & t - \tau & 0\\
                 0 & 1 & 0 & t - \tau\\
                 0 & 0 & 1 & 0\\
                 0 & 0 & 0 & 1\end{array}\right] \cdot
                 \left[\begin{array}{cc}
                 0\\
                 0\\
                 w_x(\tau) d\tau\\
                 w_y(\tau) d\tau\\
                 \end{array}\right]\\
                = {} & \left[\begin{array}{cccc}
                 1 & 0 & t & 0\\
                 0 & 1 & 0 & t\\
                 0 & 0 & 1 & 0\\
                 0 & 0 & 0 & 1\end{array}\right] \cdot \mathbb{x}(0) +
                \int_0^t \left[\begin{array}{cccc}
                 (t - \tau) w_x(\tau) d\tau\\
                 (t - \tau) w_y(\tau) d\tau\\
                 w_x(\tau) d\tau\\
                 w_y(\tau) d\tau\end{array}\right]\\
                = {} & \left[\begin{array}{cccc}
                 1 & 0 & t & 0\\
                 0 & 1 & 0 & t\\
                 0 & 0 & 1 & 0\\
                 0 & 0 & 0 & 1\end{array}\right] \cdot \mathbb{x}(0) +
                 \left[\begin{array}{cc}
                 \frac{t^2}{2} & 0\\
                 0 & \frac{t^2}{2}\\
                 t & 0\\
                 0 & t\end{array}\right] \cdot \mathbb{w}\\
    \end{aligned}

Alternatively,

.. math::
    \mathbb{x}_{k+1} = F \cdot \mathbb{x}_k + G \cdot \mathbb{w}_k

where

.. math::
    F_{2D} = \left[
    \begin{array}{cccc}
        1 & 0 & t & 0\\
        0 & 1 & 0 & t\\
        0 & 0 & 1 & 0\\
        0 & 0 & 0 & 1
    \end{array}\right]
    \textrm{and}
    G_{2D} = \left[
    \begin{array}{cc}
        \frac{t^2}{2} & 0\\
        0 & \frac{t^2}{2}\\
        t & 0\\
        0 & t\end{array}\right]

:math:`\mathbb{w}_k` is a Gaussian noise with standard deviation of :math:`\sigma`.

Similarly, for 3D space,

.. math::
    F_{3D} = \left[
    \begin{array}{cccccc}
        1 & 0 & 0 & t & 0 & 0\\
        0 & 1 & 0 & 0 & t & 0\\
        0 & 0 & 1 & 0 & 0 & t\\
        0 & 0 & 0 & 1 & 0 & 0\\
        0 & 0 & 0 & 0 & 1 & 0\\
        0 & 0 & 0 & 0 & 0 & 1\\
    \end{array}\right]
    \textrm{and}
    G_{3D} = \left[
    \begin{array}{ccc}
        \frac{t^2}{2} & 0 & 0\\
        0 & \frac{t^2}{2} & 0\\
        0 & 0 & \frac{t^2}{2}\\
        t & 0 & 0\\
        0 & t & 0\\
        0 & 0 & t\end{array}\right]
