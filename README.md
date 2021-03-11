# Model Switching

Implementation of Human Model Switching experiments in a Self-Driving Car simulator.  Experiments present a variety of situations to the robot car, 
and during portions of each it's necessary to have a more complex human model.  But at other times, a less expensive model is more than sufficient.
The robot's task it to dynamically switch between these models online to optimize the reward-compute tradeoff in real time.  This  involves only
using more complex models when they are likely to provide additional reward worth the computational expensive, and switching to less expensive ones
whenever this is not the case.  To get a sense for how these experiments look, see the **experiments/model_switching_gifs**.

Work on this project was done in association with the <a href = "http://interact.berkeley.edu/"> InterACT Lab </a> at Berkeley.

Below is a high level description of the relevant files for those interested in future work.  Note these are only intended as a supplement for the
lower level comments in files.

- **experiments**: Folder for running and analyzing model switching experiments
  - **run_experiment.py**: Main entry point to running experiments allowing for configurations & hyperparameter arguments.
  - **stay_back.py, give_way.py, merger.py**: Correspond to the three experiments we share in the paper, set up the environment as described and begin executing rollouts as prescribed by run_experiment.py.
  - **logs**: Raw experiment logs, and aggregate visualizations, and files for converting logs into the visualizations.  Additionally enables ad-hoc log analysis.
  - **model_switching_gifs**: GIFs of the model switcher being used in the various scenarios.
- **interact_drive**: Folder containing base driving simulator including extensions to supoport model switching.
  - **planner**: Implementations of different human model + trajectory planners.
    - **naive_planner.py**: Assumes human maintains constant velocity.
    - **turn_planner.py**: Assumes human optimizes reward, then robot does based on fixed human prediction.
    - **tom_planner.py**: Jointly optimizes human and robot plan based on each optimizing their own reward.
    - **model_switcher.py**: Leverages subset of above models, and uses methods for switching described in paper.
    - **switch_planner.py**: model_switcher creates one for each lower level planner.  Includes to model switching heuristic implementations.
  - **car**: Describes the car agents, their reward functions, and how they act.
    - **base_rational_car.py**: All experiments used this car or subclasses such as left_lane_car with scenario specific reward features.
    - **planner_car.py**: All intelligent cars additionally subclass this, serving as an entry point for calling the planner functions above.
  - **feature_utils.py**: Definitions of the various reward features used for experiments.
  - **switching_utils.py**: Helper functions that make model switching experiments easy to configure and run.
  - **world.py, visualizer.py, simulation_utils.py**: Maintain and optionally visualize state as it evolves.

