"""Module containing the base class for planner cars."""

from typing import Iterable, Union

import numpy as np
import tensorflow as tf

from interact_drive.car.car import Car
from interact_drive.world import CarWorld


class PlannerCar(Car):
    """
    A car that performs some variant of model predictive control, maximizing the
     sum of rewards.
    """

    def __init__(self, env: CarWorld,
                 init_state: Union[np.array, tf.Tensor, Iterable],
                 color: str = 'orange', opacity: float = 1.0,
                 friction: float = 0.2,
                 planner_type: str = "Naive",
                 planner_args: dict = None,
                 **kwargs):
        """
        Args:
            env: the carWorld associated with this car.
            init_state: a vector of the form (x, y, vel, angle), representing
                        the x-y coordinates, velocity, and heading of the car.
            horizon: the planning horizon for this car.
            color: the color of this car. Used for visualization.
            opacity: the opacity of the car. Used in visualization.
            friction: the friction of this car, used in the dynamics function.
            planner_args: the arguments to the planner (if any).
        """
        super().__init__(env, init_state, color=color, opacity=opacity,
                         friction=friction)

        self.planner_type = planner_type
        self.planner = None
        self.plan = []
        if planner_args is None:
            planner_args = {}
        self.planner_args = planner_args


    def initialize_planner(self):
        '''
        Initializes planner, intended to be used once rest of world is set up.
        '''
        from interact_drive.planner import NaivePlanner, TurnPlanner, TomPlanner
        from interact_drive.planner import ModelSwitcher

        planners = {"Naive": NaivePlanner,
                    "Turn": TurnPlanner,
                    "Tom": TomPlanner,
                    "ModelSwitcher": ModelSwitcher}
                    
        self.planner = planners[self.planner_type](self.env, self, **self.planner_args)
        

    def _get_next_control(self):
        if self.planner is None:
            self.initialize_planner()

        self.plan = self.planner.generate_plan()
        return tf.identity(self.plan[0])

    def reset(self, seed = None):
        super().reset(seed)
        if (self.planner):
            self.planner.reset_planner()
