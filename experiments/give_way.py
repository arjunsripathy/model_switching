'''
'Hold Ground' Experiment

(Related to 'Merger' experiment, more on this below)

Preface:

Car A will be used to refer the "robot" car, colored orange.
Car B will be used to refer to the blue "human" car.
Car C will be used to refer to the black "human car".
The "robot" car will be the one using the model switching mechanism,
with the capability of model switching.  More information about customizable
experiment parameters can be found in the main function below.

Car Types:

Car A: LeftLaneCar
Car B: MergerCar
Car C: FixedVelocityCar


Experiment Setup:

This experiment has 4 lanes and a stationary truck in the middle
of the road near where the cars initially are placed.
In this experiment the Car A begins in the left lane a few
car lengths behind Car C.  Car B begins in the right lane a little
ahead of Car A.

In addition to the standard motivations, Car A does not want to leave
the left lane and would like to close the gap present between itself and 
Car C ahead.  Furthermore, Car B would like to enter the left lane.

Related Experiment:
'Merger' is identical except the roles for Car A and Car B are swapped
and the LeftLaneCar is given a little headstart to make the situation more
challenging for the robot car.

The initial position are setup so that if it was not for Car A, Car B
would smoothly merge behind Car C.  However since Car A would not like
to lag behind, it is in it's best interests to catch up and drive directly
behind Car C: effectively blocking Car A from entering the lane there.
'''

from typing import Iterable, Union

import numpy as np
import tensorflow as tf

from interact_drive.world import TwoLaneCarWorld
from interact_drive.car import FixedVelocityCar, MergerCar, GiveWayCar
from interact_drive import feature_utils
from interact_drive import switching_utils


def run(experiment_args):
    '''
    Main method that runs experiment for 30 time steps.
    '''

    exp_params = switching_utils.default_experiment_params()
    EXP_TYPE_STR = switching_utils.exp_str(exp_params, experiment_args)
    EXP_NAME = f"Give Way, {EXP_TYPE_STR}"

    NUM_TIME_STEPS = 40

    world = TwoLaneCarWorld(visualizer_args= switching_utils.default_visualizer_args(EXP_NAME),
                              experiment_args = experiment_args)

    p_type, p_args = switching_utils.planner_params(exp_params, experiment_args = experiment_args)

    car_0 = GiveWayCar(world, init_position = [-0.1, -0.275],
                        color = 'orange', planner_type = p_type,
                        planner_args = p_args)

    car_1 = MergerCar(world, init_position = [0.0, -0.275],
                        color = 'blue', merge_desire = 0.7)

    car_2 = FixedVelocityCar(world, init_position = [-0.1, 0.05])
    car_3 = FixedVelocityCar(world, init_position = [-0.1, -0.6])

    world.add_cars([car_0, car_1, car_2, car_3])
    
    
    return switching_utils.execute_many_experiments(EXP_NAME, world, NUM_TIME_STEPS, experiment_args)
  
    
