"""Definition of Model Switching Planners and Switching Heuristics"""

import tensorflow as tf
from interact_drive.world import CarWorld
from interact_drive.car import Car, PlannerCar
from interact_drive.planner.car_planner import CarPlanner
from interact_drive.simulation_utils import batched_next_car_state
import pickle
import time
import numpy as np
import os

class SwitchPlanner(CarPlanner):
    """
    Parent class for all the trajectory finders for a car,
    that are capable of model switching.
    """

    def __init__(self, model_class, world: CarWorld, car: Car, 
                 meta_planner, planner_specific_args = dict()):
        '''
        model_class: Planner class being used (e.g. NaivePlanner)
        meta_planner: Metaplanner making use of this (e.g. ModelSwitcher instance)
        planner_specific_args: Contains switch heuristic rules as well as parameters
            for underlying trajectory optimization model. 
        '''
        super().__init__(world, car)

        # Reference to meta planner (i.e. meta planner)
        self.meta_planner = meta_planner

        # Collect switch heuristic rule
        heuristic_expressions = planner_specific_args.pop("heuristic_expressions") \
                                if "heuristic_expressions" in planner_specific_args else dict()


        # Initialize trajectory optimization model
        self.model = model_class(world, car, **planner_specific_args)

        # Initialize switch heuristics
        self.heuristics = dict()

        #Indicates if heuristic graph set up prior to last run
        self.heuristic_graph_set_up = True

    def setup_switch_heuristics(planners):
        '''
        Accepts a list of switch planners representing the complexity
        ladder being used.  The first model is the least expensive
        and the last model is the most expensive.

        Setups switch heuristics to switch between the ladder of planners.
        '''
        for i in range(len(planners)):
            h_dict = planners[i].heuristics
            alt_down = planners[i - 1].model if i > 0 else None
            h_dict["down"] = SwitchHeuristic(planners[i], "down", alt_model = alt_down)

            alt_up = planners[-1].model if i < (len(planners) - 1) else None
            h_dict["up"] = SwitchHeuristic(planners[i], "up", alt_model = alt_up)


    def generate_plan(self, init_state = None):
        # Trajectory optimize plan, voiding iterations involving graph setup 
        self.model.compute("plan", init_state, void_graph_setup = True)

        return self.model.split_robot_control


    def switch_indicator(self):
        '''
        Should Return 

        -1: Switch to one lower computation model.
        0: Stay with current model.
        1: Switch to highest computation model.
        '''

        # Will be flipped to False if any heuristic involves graph setup
        self.heuristic_graph_set_up = True

        indicator = 0

        if (self.heuristics["up"].should_switch()):
            indicator = 1
        elif (self.heuristics["down"].should_switch()):
            indicator = -1

        return indicator

    def reset_planner(self):
        # Reset planner by reinitializing optimizer and heuristics
        self.model.initialize_optimizer()
        for heuristic in self.heuristics.values():
            heuristic.heuristic_reset()


class SwitchHeuristic():

    # Default hyperparameters are stored as default command line arguments
    # Meta mdp reward is R - lambd * (comp time)
    lambd = None

    # Empirical values for the compuational times of various models
    comp_times = dict()

    # We wait cooldown timesteps before revaluating a heuristic (depending on switch direction)
    cooldowns = dict()

    # Trust radius we consider for estimating optimal control change given an alternate model
    trust_radius = None

    cached_values = dict()
    last_computed = dict()
    computation_log = []

    def __init__(self, switch_planner, direction, alt_model):
        '''
        swtitch_planner: The SwitchPlanner that is using this heuristic.
        direction: 'up' or 'down'. ('up' indicates this heuristics determines when we
                    should switch to a more complex model)
        alt_model: Alternative model to switch to (right below for down, top most for
                   up).  None if no alternate model in which case this heuristic never
                   activates.
        '''
        self.switch_planner = switch_planner
        self.direction = direction
        self.alt_model = alt_model

        # For easy access
        self.verbose = self.switch_planner.world.verbose

        # Time step of last computing heuristic
        self.last_computed = None

        # Number of components to control
        self.NC = self.switch_planner.model.NC

        # Potential areas of model improvement
        self.ladder_potential = self.switch_planner.meta_planner.ladder_potential
        self.influence_potential = "influence_prediction" in self.ladder_potential

    def should_switch(self):
        '''
        Returns if we should switch or not using switch heuristics,
        potentially differing due to cooldown.  If no alternate model
        always return False.
        '''
        if (self.alt_model is None):
            return False

        sp = self.switch_planner

        cooldown = SwitchHeuristic.cooldowns[self.direction]
        time_step = sp.world.get_time_step()
        if (self.last_computed is None or time_step >= (self.last_computed + cooldown)):

            evaluation = self.estimated_meta_tradeoff()
            if (sp.heuristic_graph_set_up):
                self.last_computed = time_step

            return evaluation
        
        return False


    # SWITCHING HEURISTIC DEFINITIONS

    def estimated_meta_tradeoff(self):
        '''
        Estimates the metatradeoff for the current state and model, given the 
        alternate model based on the direction this heuristic is setup for.

        The metareward is low level reward - lambd * computational time.
        The meta tradeoff is positive when a different model achieves a better
        combination of low level reward and computation time.
        '''

        sp = self.switch_planner
        time_step = sp.world.get_time_step()
        switch_key = f"{sp.model.name}_{self.alt_model.name}"

        if (self.direction == "up"):
            # Approximate best model with observation.
            control_prediction = sp.world.cars[sp.model.hi].control_log[-1]
            if (self.influence_potential):
                self.verify_grad_cache()
                influence_prediction = SwitchHeuristic.cached_values["grads"]["dhc_rc"]
        else:
            # Get predictions from model directly below.
            self.alt_model.compute("control_prediction", sp.world.last_state, void_graph_setup = True)
            control_prediction = self.alt_model.control_prediction[:self.NC]
            SwitchHeuristic.register_computation(time_step, f"{self.alt_model.name}_ctrl_pred")

            if (self.influence_potential):
                self.alt_model.compute("influence_prediction", sp.world.last_state, void_graph_setup = True)
                influence_prediction = self.alt_model.influence_prediction[:self.NC, :self.NC]
                SwitchHeuristic.register_computation(time_step, f"{self.alt_model.name}_inf_pred")

            if (not self.alt_model.graph_set_up):
                sp.heuristic_graph_set_up = False

        influence_prediction = influence_prediction if self.influence_potential else None
        control_change = self.estimate_control_change(control_prediction, influence_prediction)
        reward_change = self.estimate_reward_change(control_change)

        cur_model_time = SwitchHeuristic.comp_times[sp.model.name]
        alt_model_time = SwitchHeuristic.comp_times[self.alt_model.name]
        time_change = alt_model_time - cur_model_time

        meta_tradeoff = reward_change - SwitchHeuristic.lambd * time_change

        if (not sp.heuristic_graph_set_up):
            # We've finished setting up the graph at this point so simply return
            return None


        SwitchHeuristic.register_computation(time_step, switch_key, meta_tradeoff)

        # Display and return decision!
        if (self.verbose):
            model_name = self.switch_planner.model.name
            print(f"{model_name} {self.direction} Meta Tradeoff: {meta_tradeoff:.1e}")
        return meta_tradeoff > 0

    # Core Heuristic Functionality

    def estimate_control_change(self, control_prediction, influence_prediction = None):
        '''
        Estimates the change in robot control if an alternate model was used.
        The alternate model would have offered control_prediction and 
        influence_prediction had it been used to plan for the previous timestep.

        Estimates control change by using a second-order taylor expansion of the reward
        centerd around the observed robot control and human control at the previous
        timestep.  The optimal control change (d) is the optimizer for the below expression
        R(d) = 0.5 * dT * A * d + B * d + C
        where A, B, C are computed based on gradients evaluated at the observed
        state and controls, as well as the provided control and influence prediction.

        The optimal d is -A^-1 * B.
        '''
        sp = self.switch_planner
        time_step = sp.world.get_time_step()
        self.verify_grad_cache()
        grads = SwitchHeuristic.cached_values["grads"]

        # Convert model predictions into components for taylor expansion
        dhc_rc = influence_prediction
        hc_diff = control_prediction - sp.world.cars[sp.model.hi].control_log[-1]

        # Compute A matrix (and it's inverse)
        drr_rc_rc = grads['drr_rc_rc']
        drr_rc_hc = grads['drr_rc_hc']
        if (dhc_rc is not None):
            drr_hc_hc = grads['drr_hc_hc']
            cross_term = (drr_rc_hc @ dhc_rc)
            A = drr_rc_rc + cross_term + cross_term.T + (dhc_rc.T @ drr_hc_hc @ dhc_rc)
        else:
            A = drr_rc_rc

        # Compute B vector
        drr_rc = grads['drr_rc']
        if (dhc_rc is not None):
            drr_hc = grads['drr_hc']
            B = drr_rc + (dhc_rc.T @ drr_hc) + (drr_rc_hc @ hc_diff) + (dhc_rc.T @ drr_hc_hc @ hc_diff)
        else:
            B = drr_rc + (drr_rc_hc @ hc_diff)

        return SwitchHeuristic.constrained_optimum(A, B)

    def estimate_reward_change(self, control_change):
        '''
        Estimates change in reward compared to what was garnered in the previous time
        step.  Accounts for how the robot's control_change may influence the human's
        control.  Note that in reality the robot's control cannot affect the human's
        control at the same timestep, but this allows to better approximate what will
        happen next timestep.
        '''
        sp = self.switch_planner
        time_step = sp.world.get_time_step()
        self.verify_grad_cache()

        rc = sp.world.cars[sp.model.ri].control_log[-1]
        hc = sp.world.cars[sp.model.hi].control_log[-1]

        reward_key = "reward"
        if (SwitchHeuristic.last_computed.get("reward", -1) != time_step):
            reward = sp.world.cars[sp.model.ri].reward_fn(tf.stack(sp.world.state), rc).numpy()
            SwitchHeuristic.register_computation(time_step, reward_key, reward)
        else:
            reward = SwitchHeuristic.cached_values["reward"]

        # Identify where new controls take you and evaluate alternate reward
        controls = []
        for i in range(len(sp.world.cars)):
            if (i == sp.model.ri):
                controls.append(rc + control_change)
            elif (i == sp.model.hi):
                controls.append(hc + SwitchHeuristic.cached_values["grads"]["dhc_rc"] @ control_change)
            else:
                controls.append(tf.constant([0.] * self.NC))
        controls = tf.stack(controls)
        alt_world_state = batched_next_car_state(tf.stack(sp.world.last_state), controls, sp.world.dt)

        alt_reward = sp.world.cars[sp.model.ri].reward_fn(alt_world_state, rc + control_change).numpy()

        return alt_reward - reward
    
    # Helper Functions

    def constrained_optimum(A, B, tol = 0.05):
        '''
        Returns the d that optimizes
        R(d) = 0.5 * dT * A * d + B * d + C
        subject to the constraint that ||d||_2 <= trust_radius.

        A is assumed to be symmetric.  Tolerance is a considered as
        a proportion of the trust radius.
        '''
        # A = U @ diag(S) @ UT
        S, U = np.linalg.eig(A)
        UT_nB = U.T @ -B

        tr = SwitchHeuristic.trust_radius
        tr_tolerance = tr * tol

        min_ev, max_ev = np.min(S), np.max(S)
        if (max_ev < 0):
            unconstrained_optima = UT_nB / S
            if (np.linalg.norm(unconstrained_optima) <= SwitchHeuristic.trust_radius):
                return U @ unconstrained_optima

        # Critical scalar that maps vector to UTB to trust radius
        critical_ev = (np.linalg.norm(UT_nB) / tr)

        # Binary search for lambda within tolerance
        lm_min = max(0, max_ev, min_ev + critical_ev)
        lm_max = max_ev + critical_ev
        while (True):
            lm_mid = (lm_min + lm_max) / 2
            optima = UT_nB / (S - lm_mid)
            optima_norm = np.linalg.norm(optima)
            if (np.abs(optima_norm - tr) < tr_tolerance):
                return U @ optima
            if (optima_norm > tr):
                lm_min = lm_mid
            else:
                lm_max = lm_mid

    def verify_grad_cache(self):
        # Recomputes and caches grads if current is outdated.
        grad_key = "grads"
        sp = self.switch_planner
        time_step = sp.world.get_time_step()

        if (SwitchHeuristic.last_computed.get(grad_key, -1) != time_step):
            np_grad_dict = sp.grad_computer.get_np_grads(self.influence_potential)
            if (not sp.grad_computer.graph_set_up):
                sp.heuristic_graph_set_up = False
            SwitchHeuristic.register_computation(time_step, grad_key, np_grad_dict)

    # Bookeeping and Parameter Maintenance

    def register_computation(time_step, computation_key, computation_value = None):
        # Caches computation and registers what computation was done in the log
        if (computation_value is not None):
            SwitchHeuristic.cached_values[computation_key] = computation_value
        SwitchHeuristic.last_computed[computation_key] = time_step

        if (time_step > len(SwitchHeuristic.computation_log)):
            SwitchHeuristic.computation_log += [""] * (time_step - len(SwitchHeuristic.computation_log))
            SwitchHeuristic.computation_log[-1] = computation_key
        else:
            SwitchHeuristic.computation_log[-1] += f"; {computation_key}"


    def update_parameters(heuristic_customization):
        # Updates general heuristic parameters
        SwitchHeuristic.lambd = heuristic_customization.get('lambda', SwitchHeuristic.lambd)
        SwitchHeuristic.comp_times.update(heuristic_customization.get('comp_times', dict()))
        SwitchHeuristic.cooldowns.update(heuristic_customization.get('cooldowns', dict()))
        SwitchHeuristic.trust_radius = heuristic_customization.get('trust_radius', SwitchHeuristic.trust_radius)

    def heuristic_reset(self):
        # Resets specific heuristic
        self.last_computed = None

    def general_reset():
        # Resets shared data structures
        SwitchHeuristic.last_computed = dict()
        SwitchHeuristic.cached_values = dict()
        SwitchHeuristic.computation_log = []

