"""
Planner that handles switching between a set of planners.

Currently only supports switching between Theory of Mind and
Naive (Constant velocity) planning methods.
"""

from typing import Union
import time

import numpy as np
import tensorflow as tf
import pickle

from interact_drive.planner.car_planner import CarPlanner
from interact_drive.planner.timed_planner import TimedPlanner
from interact_drive.planner.switch_planner import SwitchPlanner, SwitchHeuristic

from interact_drive.planner.naive_planner import NaivePlanner
from interact_drive.planner.turn_planner import TurnPlanner
from interact_drive.planner.tom_planner import TomPlanner

from interact_drive.planner.grad_computer import GradComputer
from interact_drive.simulation_utils import next_car_state
from interact_drive.world import CarWorld
from interact_drive.car.car import Car


class ModelSwitcher(CarPlanner, TimedPlanner):
    """
    MPC-based CarPlanner that is capable of switching between
    different planning methods.
    """

    def __init__(self, world: CarWorld, car: Car, init_model = "Tom", 
                 planner_specific_args = dict(), use_models = None,
                 enable_switching = True):
        '''
        init_model: Model that switcher car begins by using.

        switching_mode: either a single string specifiying the heuristic
            to use for all planners, or a dictionary specifying a potentially
            different switching heuristic for each one.

        switch_thresh: either a single numerical value specifying a common
            heuristic threshold to trigger switching (the direction > or < 
            is based on context), or a dictionary specifying a potentially
            different switching threshold for each planner.

        use_models: Collections of models to switch between (if None use all)


        Currently supported switching modes are
        'hc_norm': Switches based on the norm of the human control.
                   Low norm => Naive; High norm => Theory of Mind
        'h_prox': Switches based on the proximity of the human car
                  Far away => Naive; Close by => Theory of Mind
        'update_diff': Switches based on the situational utilization
                       of the Tom-specific component of the iterative
                       optimization update.
                       Low Utilization => Naive; High => Theory of Mind
        '''
        CarPlanner.__init__(self, world, car)
        TimedPlanner.__init__(self)

        self.switching_enabled = enable_switching

        #For use by switching heurisitics, standard deviations of various controls
        self.control_stats = pickle.load(open("cache/control_stats.pkl", "rb"))

        #If a set of models is not specified only use init model
        if (use_models is None):
            use_models = {init_model}

        model_classes = {"Naive": NaivePlanner,
                         "Turn": TurnPlanner,
                         "Tom": TomPlanner}

        # Left: Low computation, simpler model
        # Right: High computation, complex model
        self.model_ladder = ["Naive", "Turn", "Tom"]

        # Filter out models we don't want to use and construct ladder
        model_classes = {mn: model_classes[mn] for mn in model_classes if mn in use_models}
        self.model_ladder = [mn for mn in self.model_ladder if mn in use_models]
        self.num_models = len(self.model_ladder)

        if (enable_switching and self.num_models < 2):
            raise Exception(f"Attempted switching with only {self.num_models} models")

        self.ladder_potential = {"control_prediction"}
        if ("Tom" in use_models):
            self.ladder_potential.add("influence_prediction")


        # Initialize all planners and switching heuristics
        self.planners = dict()
        for mn in self.model_ladder:
            self.planners[mn] = SwitchPlanner(model_classes[mn], world, car, self,
                                              planner_specific_args.get(mn, dict()))
        
        # Setup switch heuristics for all planners in the ladder
        if (self.switching_enabled):
            SwitchPlanner.setup_switch_heuristics([self.planners[mn] for mn in self.model_ladder])

        # Set up common GradComputer for gradient computation in heuristics
        h_index = planner_specific_args[self.model_ladder[-1]]["h_index"]
        grad_computer = GradComputer(world, car, h_index)

        # Give every switch planner a reference to the grad computer
        for mn in self.model_ladder:
            self.planners[mn].grad_computer = grad_computer

        # Current model, corresponding index, and planner
        self.model_type = init_model
        self.model_index = self.model_ladder.index(self.model_type)
        self.planner = self.planners[self.model_type]

        # Time series of models used this run
        self.models_used = []

        #0: stay, -1: switch down, 1: switch up
        self.switch_direction = 0

        # For easy access (do we want to log heuristic values and be verbose)
        self.verbose = world.verbose

    def reset_planner(self):
        '''
        Called at the end of a trial, should reset all relevant values,
        but reuse one time computations such as graph setups.
        '''

        # Reset planners and switching heuristics
        for mn in self.planners:
            self.planners[mn].reset_planner()
        SwitchHeuristic.general_reset()

        # Reset model usage tracking and time tracking
        self.models_used = []
        TimedPlanner.__init__(self)

    def update_switching_parameteters(self, heuristic_customization):
        SwitchHeuristic.update_parameters(heuristic_customization)

    @property
    def heuristic_computation_log(self):
        return SwitchHeuristic.computation_log

    def check_switch(self):
        '''
        Method asks the current planner if it thinks
        switching is a good idea.

        Current convention is to set switch_direction to
        one of one of:
        -1: Switch to less expensive model
         0: Stick with current model
         1: Switch to more expensive model
        '''

        
        psi = self.planner.switch_indicator()


        if (self.switch_direction <= 0 and self.model_index > 0 and psi == -1):
            self.switch_direction = -1
        elif (self.switch_direction >= 0 and self.model_index < (self.num_models - 1) and psi == 1):
            self.switch_direction = 1
        else:
            self.switch_direction = 0


    def execute_switch(self):
        '''
        If switch direction is 0 sticks with current model.
        If switch direction is 1 switch to the most complex available.
        If switch direction is -1 switch down one model.
        '''
        assert self.switch_direction in {-1, 0, 1}

        if (self.switch_direction == 0):
            return False
        
        if (self.switch_direction == 1):
            self.model_index = self.num_models -1
        elif (self.switch_direction == -1):
            self.model_index -= 1

        self.model_type = self.model_ladder[self.model_index]
        self.planner = self.planners[self.model_type]
        if (self.verbose):
            print(f"Switching to {self.model_type}\n")
        return True

    def generate_plan(self,
                      init_state: Union[
                          None, tf.Tensor, tf.Variable, np.array] = None):

        """
        Generates a sequence of controls of length self.horizon by performing
        gradient ascent on the predicted reward of the resulting trajectory.

        Track time usage for decision making (checking and switching if necessary)
        as well as planning (trajectory optimization).  Rollback graph setup time,
        so only true graph runs are considered.

        Args:
            init_state: The initial state to plan from. If none, we use the
                        current state of the world associated with the car.
        """
        self.record_section_time('step_start')

        self.record_section_time('sub_start', 'decision')

        if (self.switching_enabled and self.world.get_time_step() > 1):
            self.checkpoint('record')
            self.switch_direction = self.planner.switch_indicator()

            if (not self.planner.heuristic_graph_set_up):
                self.checkpoint('rollback')
                self.switch_direction = self.planner.switch_indicator()
   
            self.execute_switch()

        self.switch_direction = 0
        self.record_section_time('sub_end', 'decision')

        self.record_section_time('sub_start', 'planning')

        self.checkpoint('record')
        control = self.planner.generate_plan(init_state)
        if (not self.planner.model.graph_set_up):
            self.checkpoint('rollback')
            control = self.planner.generate_plan(init_state)
  
        self.models_used.append(self.model_type)

        self.record_section_time('sub_end', 'planning')

        self.record_section_time('step_end')

        return control

