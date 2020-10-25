'''
'Stay Back' Experiment

Preface:

Car A will be used to refer the "robot" car, colored orange.
Car B will be used to refer to the "human" car, colored black.
The "robot" car will be the one using the model switching mechanism,
with the capability of model switching.

Car Types:

Car A: BaseRationalCar
Car B: BaseRationalCar

Experiment Setup:

In this experiment Car A and B begin roughly alongside each other
in adjacent lanes.   There are cones up ahead that create
a traffic bottleneck, where only one car can safely pass through at
a time.

The primary challenge here is safely navigating the bottleneck, and avoiding
a situation where both cars try to pass through at the same time.  If one car
yields to the other car, however, then the both will be able to pass through
safely.
'''
from interact_drive.world import TwoLaneConeCarWorld
from interact_drive.car import FixedVelocityCar, BaseRationalCar
from interact_drive import switching_utils

def run(experiment_args):
    '''
    Main method that runs experiment for 40 time steps.
    '''

    exp_params = switching_utils.default_experiment_params()
    EXP_TYPE_STR = switching_utils.exp_str(exp_params, experiment_args)
    EXP_NAME = f"Stay Back, {EXP_TYPE_STR}"

    NUM_TIME_STEPS = 35

    world = TwoLaneConeCarWorld(cone_positions = [(-0.18, -0.25), (-0.17, -0.2), 
                                                  (-0.15, -0.15), (-0.15, -0.1),
                                                  (0.01, -0.1), (0.01, -0.15), 
                                                  (0.03, -0.2), (0.04, -0.25)],
                                visualizer_args = switching_utils.default_visualizer_args(EXP_NAME),
                                experiment_args = experiment_args)

    p_type, p_args = switching_utils.planner_params(exp_params, experiment_args = experiment_args)
    
    car_0 = BaseRationalCar(world, init_position = [-0.14, -0.9],
                        planner_type = p_type,
                        color = 'orange',
                        planner_args= p_args)
  
    car_1 = BaseRationalCar(world, init_position = [0.0, -0.9],
                        color = 'blue',)


    world.add_cars([car_0, car_1])

    return switching_utils.execute_many_experiments(EXP_NAME, world, NUM_TIME_STEPS, experiment_args)
