import sys
sys.path.append('../')
import os
import argparse
import pickle

import stay_back
import give_way
import merger

'''
To execute a model switching experiment run the following:

	python run_experiment.py <EXP_NAME> <EXP_TYPE> [Additional Params]

exp_name can currently be one of:
	stay_back
	give_way
	merger

	For more information on experiments see respective files.

exp_type can be one of:

Single Model Experiments
	0 -> Uses Naive only
	1 -> Uses Turn only
	2 -> Uses TOM only

Model Switching Experiments
	3 -> Switches between Naive and Turn
	4 -> Switches between Turn and Tom
	5 -> Switches between Naive and Tom
	6 -> Switches between Naive, Turn, and Tom

See below for additional parameters.
'''

import numpy as np
SEED_RANGE = 10000

parser = argparse.ArgumentParser()
parser.add_argument('exp_name', type = str, help = 'Name of Experiment')
parser.add_argument('exp_type', type = int, help = 'Experiment Type Number')
parser.add_argument('--save_gif', default = False, action = 'store_true', help = 'Saves GIF of experiment')
parser.add_argument('--num_run', type = int, default = 1, help = 'Number of Times to Seed & Run Experiment (defualt: 1)')
parser.add_argument('--lambdas', nargs ='+', default = [0.1], help = 'Number of lambdas to repeat all trials for.')
parser.add_argument('--verbose', default = False, action = 'store_true', help = 'Print information')
parser.add_argument('--naive_ct', type = float, default = 0.09, help = 'Naive Computational Time')
parser.add_argument('--turn_ct', type = float, default = 0.17, help = 'Turn Compuational Time')
parser.add_argument('--tom_ct', type = float, default = 0.37, help = 'Tom Compuational Time')
parser.add_argument('--up_cd', type = int, default = 1, help = "Switch up cooldown")
parser.add_argument('--down_cd', type = int, default = 3, help = "Switch down cooldown")
parser.add_argument('--tr', type = float, default = 1.5, help = "bound on delta robot control")
experiment_args = parser.parse_args()

# Verify and parse Arguments

# Save multi trial runs
experiment_args.log_all_steps = experiment_args.num_run > 1
# If only one run always verbose
experiment_args.verbose = experiment_args.verbose or experiment_args.num_run == 1

EXPERIMENTS = {"stay_back": stay_back,
			   "give_way": give_way,
			   "merger": merger}

if (experiment_args.exp_name not in EXPERIMENTS):
	raise Exception(f"Experiment {experiment_args.exp_name} not a valid experiment")

if (experiment_args.num_run > 1 and experiment_args.save_gif):
	raise Exception(f"Cannot run many and save gif")

exp_file = EXPERIMENTS[experiment_args.exp_name]

if (experiment_args.num_run > 1):
	# Seed experiments if running many for reproducibility
	np.random.seed(859)
experiment_args.seeds = np.random.randint(SEED_RANGE, size = [experiment_args.num_run])

# Run experiments and get log
complete_experiment_log = exp_file.run(experiment_args)

log_indices = complete_experiment_log['log_indices']
def get_values(log, value_name):
	return [record[log_indices.index(value_name)] for record in log]

# Display basic statistics for each threshold tried out
for lambd in experiment_args.lambdas:
	lambd_log = complete_experiment_log['runs'][lambd]
	log_indices = complete_experiment_log['log_indices']

	rewards = get_values(lambd_log, 'reward')
	planning_time = get_values(lambd_log, 'planning_time')
	decision_time = get_values(lambd_log, 'decision_time')


	print()
	print(f"Lambda: {lambd}")
	print(f"Results from {experiment_args.num_run} Experiments:" + " " * 30)
	print(f"Mean Reward: {np.mean(rewards):.3e}")
	print(f"Mean Planning Time: {np.mean(planning_time):.3e}")
	print(f"Std Dev Planning Time: {np.std(planning_time):.3e}")
	print(f"Mean Decision Time: {np.mean(decision_time):.3e}")

# Save log if requested
if(experiment_args.log_all_steps):
	log_id = f"{experiment_args.exp_name}_{experiment_args.exp_type}"
	version_num = sum([log_id in log_name for log_name in os.listdir("logs")])
	log_id += f"_v{version_num}"

	pickle.dump(complete_experiment_log, open(f"logs/{log_id}.pkl", "wb"))



