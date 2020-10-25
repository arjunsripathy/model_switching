"""Module containing a car that maintains a fixed speed."""

from typing import Union, Iterable

import numpy as np
import tensorflow as tf

from interact_drive.car.fixed_control_car import FixedControlCar
from interact_drive.world import CarWorld


class FixedVelocityCar(FixedControlCar):
    """
    A car that just goes forward at a fixed velocity, specified in its initial
    state.
    """

    def __init__(self, env: CarWorld,
                 init_position: Union[np.array, tf.Tensor, Iterable],
                 velocity = 0.6, heading = np.pi/2,
                 color: str = 'gray', opacity=1.0, **kwargs):
        # with friction set to zero, we don't need to use any controls
        controls = tf.constant(np.array([0., 0.]), dtype=tf.float32)
        init_state = init_position + [velocity, heading]
        super().__init__(env, init_state, controls, color, opacity, friction=0.,
                         **kwargs)
