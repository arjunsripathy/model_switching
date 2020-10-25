from typing import Iterable, Union

import numpy as np
import tensorflow as tf

from interact_drive.car.base_rational_car import BaseRationalCar


class MergerCar(BaseRationalCar):
    '''
    In addition to the standard features, this car
    has a an additional incentive to enter the
    left lane, and is penalized for it's squared distance
    from there.
    '''
    def __init__(self, env, init_position, color, ll_bonus = False, 
                 merge_desire = 1., planner_type = None, planner_args = None):

        weights = self.base_weights()

        # Feature for left lane desirability
        weights.append(merge_desire * -1.5e2)

        # Feature for left lane bonus
        self.ll_bonus = ll_bonus
        if (self.ll_bonus):
            weights.append(5.0)

        super().__init__(env, init_position, color, planner_type = planner_type, 
                         planner_args = planner_args, custom_weights = weights)

    def reset(self, seed = None):
        # Add a little more variance for the car's initial y position.
        super().reset(seed)
        if (seed is not None):
            np.random.seed(seed * (self.index + 1))
            np_state = self.state.numpy()
            np_state[1] += np.random.normal() * 1e-2
            self.state = tf.constant(np_state, dtype=tf.float32)

    @tf.function
    def features(self, state: Iterable[Union[tf.Tensor, tf.Variable]],
                 control: Union[tf.Tensor, tf.Variable], **kwargs) -> tf.Tensor:
        """
        Features different from base rational:
            - left_lane_feat: Squred left lane deviance.
            - in_ll_bonus: An additional bonus for having the car's
                center within the left lane.
        Args:
            state: the state of the world.
            control: the controls of this car in that state.
        Returns:
            tf.Tensor: the four features this car cares about.
        """

        feats = super().features(state, control, **kwargs)

        car_state = state[self.index]
        # Squared left lane deviance.
        left_lane_feat = (car_state[0] - self.env.left_limit) ** 2
        # Bonus for car head being in left lane.
        car_head = car_state[0] + 0.5 * self.env.car_wh[1][0] * tf.cos(car_state[3])

        if (self.ll_bonus):
            in_ll_bonus = tf.nn.relu(tf.sign(self.env.left_limit + 0.05 - car_head))
            return tf.concat((feats, [left_lane_feat, in_ll_bonus]), axis = 0)
        else:
            return tf.concat((feats, [left_lane_feat]), axis = 0)