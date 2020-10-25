"""Planner that assumes all other cars travel at constant velocity."""

from typing import Union

import time

import numpy as np
import tensorflow as tf

from interact_drive.planner.car_planner import CarPlanner
from interact_drive.simulation_utils import batched_next_car_state, next_car_state
from interact_drive.world import CarWorld
from interact_drive.car.car import Car

from inspect import signature


class CVPlanner(CarPlanner):
    """
    MPC-based CarPlanner that assumes all the other cars are FixedVelocityCars.
    """

    def __init__(self, world: CarWorld, car: Car, h_index, horizon: int, 
                 n_iter: int, consider_other_cars):
        '''
        Setups up CVPlanner with the following arguments

        world: Car's world
        car: Car object
        consider_other_cars: Flag whether to consider collisions with other cars.
        horizon: Planning horizon
        n_iter: Number of iterations to run Adam to optimize.
        '''
        CarPlanner.__init__(self, world, car)
        self.setup_planner(h_index, horizon, n_iter, plan_for_human = False)
        self.feature_args = {"consider_other_cars": consider_other_cars}

        self.planning_processes = [self.initialize_parameters, self.optimize_plan]

    @property
    def control_prediction(self):
        return np.zeros(self.NC * self.horizon, dtype = np.float32)

    @property
    def influence_prediction(self):
        return np.zeros([self.NC * self.horizon, self.NC * self.horizon], dtype = np.float32)
    

    @tf.function
    def compute_grads(self, init_state):
        func_key = "grads"

        with tf.GradientTape() as tape:
            world_state = tf.stack(init_state)

            #robot rewards
            rrs = []
            
            for t in range(self.horizon):

                rc = self.robot_control[t * self.NC: (t + 1) * self.NC]

                controls = []
                for i in range(len(self.world.cars)):
                    if (i == self.ri):
                        controls.append(rc)
                    else:
                        controls.append(tf.zeros(self.NC))
                
                controls = tf.stack(controls)
                #Advance all car states simulataneously based on the controls
                world_state = batched_next_car_state(world_state, controls, self.world.dt)


                rrs.append(self.world.cars[self.ri].reward_fn(world_state, rc, 
                                                              feature_args = self.feature_args))

            #robot reward
            rr = tf.math.reduce_sum(rrs)

        #Derivative of robot reward with respect to robot control
        drr_rc = tf.convert_to_tensor(tape.gradient(rr, self.robot_control))

        self.report_run(func_key)

        return drr_rc

    def optimize_plan(self,
                      init_state: Union[
                          None, tf.Tensor, tf.Variable, np.array] = None,
                      void_graph_setup = False):

        """
        Generates a sequence of controls of length self.horizon by performing
        gradient ascent on the predicted reward of the resulting trajectory.
        Args:
            init_state: The initial state to plan from. If none, we use the
                current state of the world associated with the car.
            void_graph_setup: The first time we run through the computation
                graph we have to set up which takes significantly longer than
                a normal run.  When we care about computation time this can
                provide misinformation.  If void_graph_setup, then when the planner
                is not setup it'll simply run one iteration to setup then return
                (with the intention of being called again for a normal planning step.)
        Returns:
        """

        for i in range(self.n_iter):
            r_grad = self.compute_grads(self.init_state)
            self.optimizer.apply_gradients([(-r_grad, self.robot_control)])

            if (not self.graph_set_up and void_graph_setup):
                break

