from typing import Iterable, Union

import numpy as np
import tensorflow as tf

from interact_drive.car.base_rational_car import BaseRationalCar


class GiveWayCar(BaseRationalCar):
    '''
    In addition to being a left lane car, this car
    has additional incentive to allow the human car to 
    merge into its lane.
    '''
    def __init__(self, env, init_position, color, merge_desire = 1.,
                 planner_type = None, planner_args = None):

        weights = self.base_weights()

        # Left lane deviance penalty.
        weights.append(-1e3)

        # Positional desirability
        weights.append(-1e1)

        # Human left lane desirability
        weights.append(-3.5e2)

        # human in lane bonus
        weights.append(5.0)

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
        #Positional feature
        pos_feat = (car_state[1] - (state[2][1] - 0.325)) ** 2
        # Human Squared left lane deviance
        human_left_lane_feat = (state[1][0] - self.env.left_limit) ** 2
        #Bonus for human car head being in lane
        car_head = state[1][0] + 0.5 * self.env.car_wh[1][0] * tf.cos(state[1][3])
        in_ll_bonus = tf.nn.relu(tf.sign(self.env.left_limit + 0.05 - car_head))
        return tf.concat((feats, [left_lane_feat, pos_feat,
                          human_left_lane_feat, in_ll_bonus]), axis = 0)


