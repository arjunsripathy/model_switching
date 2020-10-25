""" Module containing base class for cars."""

from typing import Union, Iterable

import numpy as np
import tensorflow as tf

from interact_drive.simulation_utils import get_dynamics_fn
from interact_drive.world import CarWorld


class Car(object):
    """
    Parent class for cars.

    To build on this class, you will need to implement reward_fn and
    _get_next_control. See the other car classes for examples of how to do this.

    Attributes:
        state (tf.Tensor): A vector of the form (x, y, vel, angle), representing
                        the x-y coordinates, velocity, and heading of the car.
        index (int): the index of this car in the list of car belonging to the
                        CarWorld associated with this car.
        control (tf.Tensor): A vector of the form (acc, ang_vel), representing
                        the acceleration and angular velocity being applied to
                        the car.
    """

    def __init__(self, env: CarWorld,
                 init_state: Union[np.array, tf.Tensor, Iterable], color: str,
                 opacity: float = 1.0, friction: float = 0.2, index: int = 0,
                 debug: bool = False,
                 **kwargs):
        """
        Initializes a car.

        Note: we need to add **kwargs for multiple inheritance.

        Args:
            env: the carWorld associated with this car.
            init_state: a vector of the form (x, y, vel, angle), representing
                    the x-y coordinates, velocity, and heading of the car.
            color: the color of this car. Used for visualization.
            opacity: the opacity of the car. Used in visualization.
            friction: the friction of this car, used in the dynamics function.
            index: the index of this car in the list of car belonging to the
                    CarWorld associated with this car.
            debug: a debug flag. If true, we log the past trajectory of the car.
        """
        self.env = env
        self.lane_medians_t = env.lane_medians_t
        self.lane_normals_t = env.lane_normals_t

        self.dynamics_fn = get_dynamics_fn(tf.constant(friction))
        self.init_state = tf.constant(init_state, dtype=tf.float32)
        self.state = init_state

        self.debug = debug
        self.past_traj = []

        self.color = color
        self.opacity = opacity

        self.index = index

        self.control_log = []
        self.control = None
        self.control_already_determined_for_current_step = False

    def reset(self, seed = None):
        '''
        Reset while also perturbing initial state to allow for variance 
        across random seeds.  The initial state involves position, speed, 
        and heading.  Perturbation magnitude is based on an understanding of 
        an environment and what would create sufficient variance without
        entirely altering the spirit of an experiment.
        '''
        if (seed is not None):
            np.random.seed(seed * (self.index + 1))
            np_state = self.init_state.numpy()
            np_state[0] += np.random.normal() * 5e-4
            np_state[1] += np.random.normal() * 5e-4
            np_state[2] += np.random.normal() * 1e-3
            np_state[3] += np.random.normal() * 1e-3
            self.state = tf.constant(np_state, dtype=tf.float32)
        else:
            self.state = self.init_state

        self.control_log = []
        self.control_already_determined_for_current_step = False

        if self.debug:
            self.past_traj = []

    def step(self, dt):
        """
        Updates the state of the car based on self.control.

        Args:
            dt: the amount of time to increment the simulation by.
        """
        if self.debug:
            self.past_traj.append((self.state, self.control))

        self.control_already_determined_for_current_step = False
        self.state = self.dynamics_fn(self.state, self.control, dt)

        self.control_log.append(self.control.numpy())

    @tf.function
    def reward_fn(self, world_state, self_control):
        """
        The reward function of this car. That is, the reward the car receives
        upon transitioning into `world_state` with controls `self_controls`.
        """
        raise NotImplementedError

    def _get_next_control(self) -> tf.Tensor:
        """
        Computes and returns the next control for this car.

        Should only be called once per timestep - only this if the
        self.control_already_determined_for_current_step flag is false.

        Returns:
            tf.Tensor: the next control for this car.
        """
        raise NotImplementedError

    def set_next_control(self,
                         control: Union[None, np.array,
                                        tf.Tensor, Iterable] = None):
        """
        Sets the control to control if a new_control is specified.
        Otherwise, we call self._get_next_control.
        """

        if control is not None:
            self.control = tf.constant(control, dtype=tf.float32)
        else:
            if not self.control_already_determined_for_current_step:
                self.control = self._get_next_control()

        self.control_already_determined_for_current_step = True
