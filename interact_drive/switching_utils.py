"""
Utility functions for setting up, conducting, and analyzing
model switching experiments.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import tensorflow as tf
import pickle
import datetime
import contextlib
import time
from interact_drive.car import PlannerCar
with contextlib.redirect_stdout(None):
    from moviepy.editor import ImageSequenceClip

MODELS = ["Naive", "Turn", "Tom"]
COLORS = ['g','b','y']

def execute_experiment(exp_name, world, time_steps, experiment_args, 
                       seed = None, ms_car_index = 0):
    '''
    Runs experiment.

    Tracks reward per time step, which models are being used,
    and the computational time for planning.

    Displays and saves relevant time series information at end.

    Saves GIF of experiment if requested.

    Args:
        world: Car World Object 
        time_steps: number of time steps to execute experiment for.
        ms_car_index: index of model switching car (default 0)
        save_gif: True iff you would like to save the simulation as a GIF.
    '''

    #Initialize planners if not yet done
    for car in world.cars:
        if (isinstance(car, PlannerCar) and car.planner is None):
            car.initialize_planner()

    if (world.verbose):
        print(f"Executing {exp_name} for {time_steps} time steps...")

    #Model Switching Car
    ms_car = world.cars[ms_car_index]

    world.reset(seed)
    if (world.verbose):
        world.render()

    if (experiment_args.save_gif):
        frames = []
        frames.append(world.render("rgb_array"))

    #Reward accrued at each time step.
    reward_ts = []

    for t in range(time_steps):

        '''
        Step world and get controls all cars
        took and the new state of the world.
        '''
        _, control, new_state = world.step()
        if (world.verbose):
            world.render()

        if (experiment_args.save_gif):
            frames.append(world.render("rgb_array"))

        ms_control = control[ms_car_index]
        #Reward for the model switching car.
        rew = ms_car.reward_fn(tf.stack(new_state), ms_control).numpy()
        reward_ts.append(rew)

        #Computational Time Breakdown
        if (world.verbose):
            ct_breakdown = ms_car.planner.avg_comp_time_breakdown()
            print(f"T: {t + 1}; R: {rew:.2f}; CT: {ct_breakdown}")

    
    if (experiment_args.num_run == 1):
        model_ts = ms_car.planner.models_used

        #Average Computational Time Time Series
        avg_step_times = ms_car.planner.get_avg_comp_times()

        print()
        #Gather reward and computation time information
        #Single Computational Time Time Series
        step_times = ms_car.planner.get_comp_times()

        #Display reward and computation time graphs
        display_rewards(reward_ts, model_ts)
        display_computational_times(step_times['overall'],
                                    avg_step_times['overall'])


    if (experiment_args.save_gif):
        clip = ImageSequenceClip(frames, fps=int(1 / world.dt))
        clip.speedx(0.5).write_gif(f"{exp_name}.gif", program="ffmpeg")

    #return np.mean(reward_ts), avg_step_times['overall'][-1], model_usage
    return reward_ts

def execute_many_experiments(exp_name, world, time_steps, experiment_args,
                             ms_car_index = 0):
    switching_parameters = {"comp_times": {"Naive": experiment_args.naive_ct,
                                           "Turn": experiment_args.turn_ct,
                                           "Tom": experiment_args.tom_ct},
                            "cooldowns": {"up": experiment_args.up_cd,
                                          "down": experiment_args.down_cd},
                            "trust_radius": experiment_args.tr}
    world.cars[ms_car_index].initialize_planner()
    world.cars[ms_car_index].planner.update_switching_parameteters(switching_parameters)

    run_log = dict()
    for lambd in experiment_args.lambdas:
        print()
        print(f"Using Lambda: {lambd}")
        world.cars[ms_car_index].planner.update_switching_parameteters({"lambda": float(lambd)})

        lambd_log = []
        run_time = 0
        for i, seed in enumerate(experiment_args.seeds):
            if (run_time == 0):
                print(f"Running Experiment {i + 1}/{experiment_args.num_run}", end = "\r")
            else:
                et = (experiment_args.num_run - i) * run_time / (i - 1)
                print(f"Running Experiment {i + 1}/{experiment_args.num_run}, Expected Time Left: {et:.0f}s   ", end = "\r")
            if (i >= 1):
                start_time = time.time()
            #mean_rew, mean_ct, model_usage 
            reward_ts = execute_experiment(exp_name, world, time_steps, experiment_args, 
                                           seed = seed, ms_car_index = ms_car_index)
            if (i >= 1):
                run_time += time.time() - start_time

            ms_car = world.cars[ms_car_index]
            states = [[list(car_s.numpy()) for car_s in all_car_s] for all_car_s in world.past_states]
            actions =  [list(c) for c in ms_car.control_log]
            models = ms_car.planner.models_used
            heur_comp_log = ms_car.planner.heuristic_computation_log
            if (len(heur_comp_log) < time_steps):
                heur_comp_log += [""] * (time_steps - len(heur_comp_log))
            planning_times = ms_car.planner.get_comp_times()['planning']
            decision_times = ms_car.planner.get_comp_times()['decision']

            lambd_log += list(zip(states, actions, reward_ts, models, heur_comp_log, planning_times, decision_times))
            assert(len(lambd_log) == time_steps * (i + 1))

        run_log[lambd] = lambd_log

        print(f"Finished Running {experiment_args.num_run} Experiments!" + " " * 30)

    complete_experiment_log= {'name': exp_name, 'time_steps': time_steps,
                               'num_run': experiment_args.num_run,
                               'models': ms_car.planner.model_ladder,
                               'log_indices': ['state', 'action', 'reward',
                                               'model', 'heur_comp_log',
                                               'planning_time','decision_time'],
                                'runs': run_log}

    return complete_experiment_log


def display_rewards(reward_ts, model_ts):
    '''
    Displays reward for each time step gained by the car
    in a plot.  Color codes by model used and presents
    an appropriate legend.
    '''
    plt.title("Reward by Model")

    start_time = 0
    cur_model = model_ts[0]
    used_models = [cur_model]
    for t, model in enumerate(model_ts):
        if (model != cur_model):
            plt.plot(range(start_time, t + 1), 
                     reward_ts[start_time:t + 1], 
                     color = COLORS[MODELS.index(cur_model)])

            start_time = t
            cur_model = model
            if (model not in used_models):
                used_models.append(model)

    plt.plot(range(start_time, len(model_ts)), 
             reward_ts[start_time:], 
             color = COLORS[MODELS.index(cur_model)])

    patch = lambda i: mpatches.Patch(color=COLORS[MODELS.index(used_models[i])], 
                                     label=used_models[i])
    plt.legend(handles=[patch(i) for i in range(len(used_models))])
    plt.xlabel("Time Step Number")
    plt.ylabel("Reward")
    plt.show()

def display_computational_times(sing_ct_ts, avg_ct_ts):
    '''
    Displays computation time for each time step
    of planning done by the car.  Also plots for
    every value of t, the average of the first
    t computation times.
    '''

    plt.title("Computational Time")
    plt.plot(sing_ct_ts, color = 'r')
    plt.plot(avg_ct_ts, color = 'b')
    plt.legend(["Single Step", "Average"])
    plt.xlabel("Time Step Number")
    plt.ylabel("Computational Time")
    plt.show()

def default_experiment_params():
    '''
    Returns a dictionary containing default paramters
    for an experiment.

    Convention is Car 0 is the robot car, Car 1 is
    the human car.  Any further cars are either fixed
    velocity or we don't bother about.

    An experiment may modify the returned dictionary
    for any custom parameter settings.
    '''
    exp_params = dict()
    exp_params["Naive"] = {"horizon": 5, "n_iter": 20} 
    exp_params["Turn"] = {"horizon": 5, "n_iter": 20}
    exp_params["Tom"] = {"horizon": 5, "n_iter": 20}

    for m in MODELS:
        exp_params[m]["h_index"] = 1
    
    return exp_params

def default_visualizer_args(EXP_NAME):
    return {'name': EXP_NAME, 'display_model': True, 
            'display_y': True, 'follow_main_car': True}

def switch_model_pstr(model_name, experiment_params):
    # Returns a string describing the model parameters
    model_params = experiment_params[model_name]
    return f"({model_params['horizon']},{model_params['n_iter']})"


def exp_str(experiment_params, experiment_args):
    '''
    Return the string capturing the experiment's planning mechanisms 
    based on the experiment parameters and experiment type.
    ''' 
    EXP_STRS = [None] * 11

    pstrs = {m: switch_model_pstr(m, experiment_params) for m in MODELS}

    # Single Model Experiments
    EXP_STRS[0] = f"Naive{pstrs['Naive']}"
    EXP_STRS[1] = f"Tu{pstrs['Turn']}"
    EXP_STRS[2] = f"Tom{pstrs['Tom']}"


    # Model Switching Experiments
    EXP_STRS[3] = f"switch({EXP_STRS[0]}, {EXP_STRS[1]})"
    EXP_STRS[4] = f"switch({EXP_STRS[1]}, {EXP_STRS[2]})"
    EXP_STRS[4] = f"switch({EXP_STRS[0]}, {EXP_STRS[2]})"
    EXP_STRS[6] = f"switch({EXP_STRS[0]}, {EXP_STRS[1]}, {EXP_STRS[2]})"

    return EXP_STRS[experiment_args.exp_type]

def setup_switching(planner_args, use_models):
    '''
    Sets up appropriate planner arguments for switching along
    series of models.
    '''

    planner_args["init_model"] = use_models[0]
    planner_args["use_models"] = set(use_models)


def planner_params(experiment_params, experiment_args = None, exp_type = None):
    '''
    Supply the planner type and arguments based on the parameters
    of the experiment and the type of the experiment.
    '''
    planner_params = dict()
    for model in MODELS:
        planner_params[model] = experiment_params[model]

    planner_args = {"planner_specific_args": planner_params}

    if (experiment_args is not None):
        exp_type = experiment_args.exp_type

    if (exp_type == 0):
        planner_type = "ModelSwitcher"
        planner_args["init_model"] = "Naive"
        planner_args["enable_switching"] = False

    elif (exp_type == 1):
        planner_type = "ModelSwitcher"
        planner_args["init_model"] = "Turn"
        planner_args["enable_switching"] = False

    elif (exp_type == 2):
        planner_type = "ModelSwitcher"
        planner_args["init_model"] = "Tom"
        planner_args["enable_switching"] = False

    elif (exp_type == 3):
        planner_type = "ModelSwitcher"
        use_models = ["Naive", "Turn"]
        setup_switching(planner_args, use_models)

    elif (exp_type == 4):
        planner_type = "ModelSwitcher"
        use_models = ["Turn", "Tom"]
        setup_switching(planner_args, use_models)

    elif (exp_type == 5):
        planner_type = "ModelSwitcher"
        use_models = ["Naive", "Tom"]
        setup_switching(planner_args, use_models)

    elif (exp_type == 6):
        planner_type = "ModelSwitcher"
        use_models = ["Naive", "Turn", "Tom"]
        setup_switching(planner_args, use_models)
    else:
        raise Exception(f"Invalid Experiment Type: {exp_type}")

    return planner_type, planner_args



