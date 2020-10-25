from typing import Iterable, Union

import numpy as np
import tensorflow as tf

from interact_drive.car.base_rational_car import BaseRationalCar


class LeftLaneCar(BaseRationalCar):
    '''
    In addition to the standard features, this car
    has a few additional incentives.

    First, instead of being rewarded for velocity it
    is rewarded for it's y-position relative to Car C,
    the fixed velocity black car.

    Second, it has a strong incentive to remain the
    left lane, and is penalized for it's squared distance
    from there.
    '''
    def __init__(self, env, init_position, color, 
                 planner_type = None, planner_args = None):

        weights = self.base_weights()

        # Adding weight for left lane deviance penalty.
        weights.append(-1e3)

        #Appending positional feat
        weights.append(-7e1)

        super().__init__(env, init_position, color, planner_type = planner_type, 
                         planner_args = planner_args, custom_weights = weights)

    @tf.function
    def features(self, state: Iterable[Union[tf.Tensor, tf.Variable]],
                 control: Union[tf.Tensor, tf.Variable], **kwargs) -> tf.Tensor:
        """
        Features different from base rational:
            - pos_feat (replaces velocity_feat): Squared y-distance to center of Car C
            - left_lane_feat: Squred left lane deviance.
        Args:
            state: the state of the world.
            control: the controls of this car in that state.

        Returns:
            tf.Tensor: the four features this car cares about.
        """

        feats = super().features(state, control, **kwargs)

        car_state = state[self.index]

        # Squared left lane deviance
        left_lane_feat = (car_state[0] - self.env.left_limit) ** 2
        # Squared y-distance to a spot right behind of Car C
        pos_feat = (car_state[1] - (state[2][1] - 0.25)) ** 2
        return tf.concat((feats, [left_lane_feat, pos_feat]), axis = 0)