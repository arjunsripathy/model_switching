"""
Extends planner class, but simply used for the utility of computing
gradients with respect to controls in the environment.  Used as signal
for switching heuristics.
"""

from typing import Union
import time

import numpy as np
import tensorflow as tf

from interact_drive.planner.car_planner import CarPlanner
from interact_drive.simulation_utils import batched_next_car_state
from interact_drive.world import CarWorld
from interact_drive.car.car import Car


class GradComputer(CarPlanner):
    """
    Utility class computing gradients with a setup similar to an
    actual car planner.
    """

    def __init__(self, world: CarWorld, car: Car, h_index):
        '''
        Setups up GradComputer with the following arguments
        world: Car's world
        car: Car object
        kwargs: used ones are h_index (human car index) and horizon
        '''
        CarPlanner.__init__(self, world, car)
        self.name = "GradComputer"

        self.ri = self.car.index
        self.hi = h_index

        self.zeros_control = np.zeros(self.NC, dtype = np.float32)
        self.robot_control = tf.Variable(self.zeros_control)
        self.human_control = tf.Variable(self.zeros_control)

    def get_np_grads(self, influence_potential):
        '''
        Initializes control and state parameters based on last time step.
        Then run TF graph to generate all required reward/control tensor gradients.  
        Then compile a dictionary mapping gradient names to the ndarrays.

        If we don't have to deal with influence potential we only need a subset of
        the gradients.
        '''
        self.robot_control.assign(self.world.cars[self.ri].control_log[-1])
        self.human_control.assign(self.world.cars[self.hi].control_log[-1])

        self.graph_set_up = True

        if (influence_potential):
            tensor_grads = self.compute_all_tensor_grads(self.world.last_state)
            grad_names = ['drr_rc', 'drr_hc', 'dhr_hc', 'drr_rc_hc', 'drr_rc_rc', 'drr_hc_hc', 'dhc_rc']
        else:
            tensor_grads = self.compute_non_inf_tensor_grads(self.world.last_state)
            grad_names = ['drr_rc', 'dhr_hc', 'drr_rc_hc', 'drr_rc_rc', 'dhc_rc']

        np_grad_dict = dict()
        for i in range(len(grad_names)):
            np_grad_dict[grad_names[i]] = tensor_grads[i].numpy()

        return np_grad_dict

    @tf.function
    def compute_all_tensor_grads(self, init_state):
        '''
        Compute and return all relative gradients for constructing a 
        local quadratic approximation for the reward around the initial state.
        '''
        func_key = "all_grads"

        with tf.GradientTape(persistent = True) as tape2:

            with tf.GradientTape(persistent = True) as tape:
                world_state = tf.stack(init_state)

                controls = []
                for i in range(len(self.world.cars)):
                    if (i == self.ri):
                        controls.append(self.robot_control)
                    elif (i == self.hi):
                        controls.append(self.human_control)
                    else:
                        controls.append(tf.constant([0.] * self.NC))
                
                controls = tf.stack(controls)
                #Advance all car states simulataneously based on the controls
                world_state = batched_next_car_state(world_state, controls, self.world.dt)

                #Human and robot rewards
                rr = self.world.cars[self.ri].reward_fn(world_state, self.robot_control)
                hr = self.world.cars[self.hi].reward_fn(world_state, self.human_control)

            #Derivative of rewards with respect to controls.
            drr_rc = tf.convert_to_tensor(tape.gradient(rr, self.robot_control))
            drr_hc = tf.convert_to_tensor(tape.gradient(rr, self.human_control))
            dhr_hc = tf.convert_to_tensor(tape.gradient(hr, self.human_control))

            del tape

        #Second derivatives of robot reward with respect to controls.
        drr_rc_hc = tape2.jacobian(drr_rc, self.human_control)
        drr_rc_rc = tape2.jacobian(drr_rc, self.robot_control)
        drr_hc_hc = tape2.jacobian(drr_hc, self.human_control)

        #Second derivatives of human reward with respect to controls
        dhr_hc_rc = tape2.jacobian(dhr_hc, self.robot_control)
        dhr_hc_hc = tape2.jacobian(dhr_hc, self.human_control)

        del tape2

        dhc_rc = -tf.linalg.solve(dhr_hc_hc, dhr_hc_rc)

        self.report_run(func_key)

        return drr_rc, drr_hc, dhr_hc, drr_rc_hc, drr_rc_rc, drr_hc_hc, dhc_rc


    @tf.function
    def compute_non_inf_tensor_grads(self, init_state):
        '''
        Compute and return all relative gradients for constructing a 
        local quadratic approximation for the reward around the initial state.
        Leave out factors which are only relevant if some model captures influence.
        '''
        func_key = "non_inf_grads"

        with tf.GradientTape(persistent = True) as tape2:

            with tf.GradientTape(persistent = True) as tape:
                world_state = tf.stack(init_state)

                controls = []
                for i in range(len(self.world.cars)):
                    if (i == self.ri):
                        controls.append(self.robot_control)
                    elif (i == self.hi):
                        controls.append(self.human_control)
                    else:
                        controls.append(tf.constant([0.] * self.NC))
                
                controls = tf.stack(controls)
                #Advance all car states simulataneously based on the controls
                world_state = batched_next_car_state(world_state, controls, self.world.dt)

                #Human and robot rewards
                rr = self.world.cars[self.ri].reward_fn(world_state, self.robot_control)
                hr = self.world.cars[self.hi].reward_fn(world_state, self.human_control)

            #Derivative of rewards with respect to controls.
            drr_rc = tf.convert_to_tensor(tape.gradient(rr, self.robot_control))
            dhr_hc = tf.convert_to_tensor(tape.gradient(hr, self.human_control))

            del tape

        #Second derivatives of robot reward with respect to controls.
        drr_rc_hc = tape2.jacobian(drr_rc, self.human_control)
        drr_rc_rc = tape2.jacobian(drr_rc, self.robot_control)

        #Second derivatives of human reward with respect to controls
        dhr_hc_rc = tape2.jacobian(dhr_hc, self.robot_control)
        dhr_hc_hc = tape2.jacobian(dhr_hc, self.human_control)

        del tape2

        dhc_rc = -tf.linalg.solve(dhr_hc_hc, dhr_hc_rc)

        self.report_run(func_key)

        return drr_rc, dhr_hc, drr_rc_hc, drr_rc_rc, dhc_rc
