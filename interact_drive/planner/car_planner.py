"""Base class for planners."""

from interact_drive.world import CarWorld
from interact_drive.car import Car, PlannerCar
import numpy as np
import tensorflow as tf


class CarPlanner(object):
    """
    Parent class for all the trajectory finders for a car.
    """

    def __init__(self, world: CarWorld, car: Car):
        self.world = world
        self.car = car

        self.graph_set_up = False
        self.graphs_ran = set()

        self.NC = 2

        self.world.initial_setup_complete = False

        self.last_state_computed_for = None
        self.planning_processes = []
        self.process_reqs = dict()
        self.next_process_index = 0

    def setup_planner(self, h_index, horizon, n_iter, plan_for_human):
        '''
        Initializes parameters for a planner that optimizes using Adam. plan_for_human indicates
        if we want to create a control tensor for optimizing our perception of the human behavior
        or not.
        '''
        self.horizon = horizon
        self.n_iter = n_iter
        self.learning_rate = 3e-2

        self.ri = self.car.index
        self.hi = h_index
        self.initialize_optimizer()

        self.plan_for_human = plan_for_human

        self.zeros_control = np.zeros(self.NC * self.horizon, dtype = np.float32)
        self.robot_control = tf.Variable(self.zeros_control)
        if (self.plan_for_human):
            self.human_control = tf.Variable(self.zeros_control)

    def initialize_optimizer(self):
        self.optimizer = tf.keras.optimizers.SGD(learning_rate = self.learning_rate)

    def compute(self, process_name, init_state = None, void_graph_setup = False):
        '''
        Computation is modelled as a sequence of sequential sub computations.  A
        specific process may only require some of these to be done, so we execute
        as few of the sub computations necessary to complete the requested process
        name.  What sub computations are necessary for various processes are defined
        in the specific planner.
        
        For example 'control_prediction' probably only requires some of the computations.
        'plan' is a special process_name which requires all computations to be done.
        '''

        if init_state is None:
            self.init_state = self.world.state
        else:
            self.init_state  = init_state

        if (process_name in self.process_reqs):
            process_index = self.process_reqs[process_name]
        elif (process_name == "plan"):
            process_index = len(self.planning_processes)
        else:
            process_index = 0

        flat_init_state = np.concatenate(self.init_state, axis = 0)
        
        if (not np.all(flat_init_state == self.last_state_computed_for)):
            self.next_process_index = 0

        while (self.next_process_index < process_index):
            self.planning_processes[self.next_process_index](self.init_state, void_graph_setup)
            self.next_process_index += 1

        if (not self.graph_set_up and void_graph_setup):
            self.next_process_index = 0
        else:
            self.last_state_computed_for = flat_init_state


    def initialize_parameters(self, init_state = None, void_graph_setup = False):
        '''
        Initializes robot and human controls to base values for optimization.
        By default graph is set up until it goes to a gradient computation that
        says otherwise.
        '''
        self.graph_set_up = True
        self.robot_control.assign(self.zeros_control)
        if (self.plan_for_human):
            self.human_control.assign(self.zeros_control)

        
    def generate_plan(self, init_state = None):
        self.compute("plan", init_state)
        return self.split_robot_control

    @property
    def split_robot_control(self):
        '''
        Assumes self.robot_control contains a self.NC * self.horizon long tensor of
        control values.  Returns a list of self.horizon elements each containing the
        self.NC control for the corresponding time step.
        '''
        return [self.robot_control[t * self.NC: (t + 1) * self.NC] for t in range(self.horizon)]

    def report_run(self, func_key):
        '''
        Report that a computational graph has been ran.  If this is the first time
        then we take steps to denote and account for the fact that it was a graph 
        setup run and thus much longer than a normal run.
        '''

        if (func_key in self.graphs_ran):
            return

        self.graphs_ran.add(func_key)
        self.graph_set_up = False

        if (self.world.verbose):
            print(f"Car {self.car.index + 1}/{len(self.world.cars)} {self.name} {func_key} Graph Set Up")

        if (not self.world.initial_setup_complete):
            last_planner = True
            for i in range(self.car.index + 1, len(self.world.cars)):
                if (isinstance(self.world.cars[i], PlannerCar)):
                    last_planner = False
                    break

            if (last_planner):
                self.world.initial_setup_complete = True
        

class CoordinateAscentPlanner(CarPlanner):
    """
    CarPlanner that performs coordinate ascent to find an approximate Nash
    equilibrium trajectory.
    """


# class HierarchicalPlanner(CarPlanner):
#     def __init__(self, world, car):
#         pass
#
#     def initial
