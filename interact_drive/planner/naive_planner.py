"""Planner that assumes all other cars travel at constant velocity, but
   still acounts for collision risk while planning"""


from interact_drive.planner.cv_planner import CVPlanner
from interact_drive.world import CarWorld
from interact_drive.car.car import Car

import tensorflow as tf

from inspect import signature


class NaivePlanner(CVPlanner):
    """
    MPC-based CarPlanner that assumes all the other cars are FixedVelocityCars,
    but accounts for their presence.
    """

    def __init__(self, world: CarWorld, car: Car, h_index, horizon: int,
                 n_iter: int):
        '''
        Setups up NaivePlanner with the following arguments

        world: Car's world
        car: Car object
        horizon: Planning horizon
        n_iter: Number of iterations to run Adam to optimize.

        A CVPlanner with compute_col_feat = True
        '''
        CVPlanner.__init__(self, world, car, h_index, 
                           horizon, n_iter,
                           consider_other_cars = True)
        self.name = "Naive"

