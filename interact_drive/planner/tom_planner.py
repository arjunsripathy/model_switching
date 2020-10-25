"""Planner that uses provided knowledge of the reward functions of other
cars.  It plans in such a way that captures potential interactions between
itself and other cars, accounting for how the others may respond to proposed
actions.
"""

from typing import Union
import time

import numpy as np
import tensorflow as tf

from interact_drive.planner.car_planner import CarPlanner
from interact_drive.simulation_utils import batched_next_car_state, next_car_state
from interact_drive.world import CarWorld
from interact_drive.car.car import Car


class TomPlanner(CarPlanner):
    """
    MPC-based CarPlanner that uses a TOM strategy. Using the
    reward function of other car's it jointly optimizes a trajectory
    that is most beneficial.
    """

    def __init__(self, world: CarWorld, car: Car, horizon: int, n_iter: int, 
                 h_index: int,):
        '''
        Setups up TomPlanner with the following arguments

        world: Car's world
        car: Car object
        horizon: Planning horizon
        h_index: index of the car that we reason about.
        n_iter: Number of iterations to run Adam to optimize.
        learing_rate: Adam Learning Rate
        '''
        CarPlanner.__init__(self, world, car)
        self.name = "Tom"
        self.setup_planner(h_index, horizon, n_iter, plan_for_human = True)

        self.planning_processes = [self.initialize_parameters, self.optimize_plan]
        self.process_reqs["control_prediction"] = len(self.planning_processes)
        self.process_reqs["influence_prediction"] = len(self.planning_processes)

    @property
    def control_prediction(self):
        return self.human_control.numpy()

    @property
    def influence_prediction(self):
        return self.dhc_rc

    @tf.function
    def compute_grads(self, init_state):
        # Returns gradients with respect to reward (considering second order terms)
        func_key = "grads"

        with tf.GradientTape(persistent = True) as tape2:

            with tf.GradientTape(persistent = True) as tape:
                world_state = tf.stack(init_state)
                
                #robot rewards
                rrs = []
                #human rewards
                hrs = []
                
                for t in range(self.horizon):

                    rc = self.robot_control[t * self.NC: (t + 1) * self.NC]
                    hc = self.human_control[t * self.NC: (t + 1) * self.NC]

                    controls = []
                    for i in range(len(self.world.cars)):
                        if (i == self.ri):
                            controls.append(rc)
                        elif (i == self.hi):
                            controls.append(hc)
                        else:
                            controls.append(tf.zeros(self.NC))
                    
                    controls = tf.stack(controls)
                    #Advance all car states simulataneously based on the controls
                    world_state = batched_next_car_state(world_state, controls, self.world.dt)


                    rrs.append(self.world.cars[self.ri].reward_fn(world_state, rc))
                    hrs.append(self.world.cars[self.hi].reward_fn(world_state, hc))


                #robot reward
                rr = tf.math.reduce_sum(rrs)
                #human reward
                hr = tf.math.reduce_sum(hrs)

            #Derivative of human reward with respect to robot control
            dhr_rc = tf.convert_to_tensor(tape.gradient(hr, self.robot_control))

            #Derivative of human reward with respect to human control
            dhr_hc = tf.convert_to_tensor(tape.gradient(hr, self.human_control))

            #Derivative of robot reward with respect to robot control
            drr_rc = tf.convert_to_tensor(tape.gradient(rr, self.robot_control))

            #Derivative of robot reward with respect to human control
            drr_hc = tf.convert_to_tensor(tape.gradient(rr, self.human_control))

            del tape

        #Second derivatives of human reward with respect to robot and human controls.
        dhr_hc_rc = tape2.jacobian(dhr_hc, self.robot_control)

        #Second derivatives of human reward with respect to human controls.
        dhr_hc_hc = tape2.jacobian(dhr_hc, self.human_control)

        del tape2

        '''
        ToM robot gradient involves both naive gradient and the gradient
        through influence on the human.
        '''  
        dhc_rc = -tf.linalg.solve(dhr_hc_hc, dhr_hc_rc)

        r_grad = drr_rc + tf.linalg.matvec(tf.transpose(dhc_rc), drr_hc)

        self.report_run(func_key)

        return r_grad, dhr_hc, dhc_rc

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
            r_grad, h_grad, self.dhc_rc = self.compute_grads(self.init_state)
            self.optimizer.apply_gradients([(-r_grad, self.robot_control), 
                                            (-h_grad, self.human_control)])
            if (not self.graph_set_up and void_graph_setup):
                break
