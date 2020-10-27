'''
Suite of functions for analyzing logs of experiments run under many seeds
lambdas and other configurations.  Intended usage is to run with python
interactive mode providing the pickle log file to analyze.  Then ad-hoc using
the methods here to generate insights and visualizations.

Run
python -i explore_exp_logs.py <experiment pickle log path>

agg_episode_stats: print out average time step metrics further aggregated
over random seeds.  For example one may analyze the "reward", "planning_time",
"decision_time", "overall_time" and more.
e.g. agg_episode_stats("reward")

model_usage(): Displays the breakdown of how frequently each model was used.

ovr_time_by_y(): Displays aggregate time series analysis for the computational
time usage over the course of the experiment.  Useful for understanding when
more/less complex models were used.
'''

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
plt.rcParams.update({'hatch.linewidth': 3, 
                     'font.family': 'palatino',
                     'font.size': 18,
                     'figure.figsize': (8, 6)})

log_filename = sys.argv[1]
experiment_log = pickle.load(open(f"{log_filename}", "rb"))
print(f"Analyzing log_filename...")

NUM_RUN = experiment_log['num_run']
TIME_STEPS = experiment_log['time_steps']
NUM_STEPS = NUM_RUN * TIME_STEPS
LAMBDAS = sorted(list(experiment_log['runs'].keys()))
print(f"{NUM_RUN} trials of {TIME_STEPS} saved for {len(LAMBDAS)} lambdas.")

LOG_INDICES = experiment_log['log_indices']
COLORS = ['gold','darkturquoise'][:len(LAMBDAS)]

def extract(value, log):
    if (value in LOG_INDICES):
        return [r[LOG_INDICES.index(value)] for r in log]
    elif (value == 'overall_time'):
        return [r[LOG_INDICES.index('planning_time')] + 
                r[LOG_INDICES.index('decision_time')] for r in log]

def agg_episode_stats(value):
    
    print(f"Analyzing {value}")
    for lambd in LAMBDAS:
        lambd_log = experiment_log['runs'][lambd]

        mean_values = []
        for i in range(NUM_RUN):
            episode_log = lambd_log[i * TIME_STEPS: (i + 1) * TIME_STEPS]
            episode_values = extract(value, episode_log)
            mean_values.append(np.mean(episode_values))

        print(f"\nLAMBDA: {lambd}")
        print(f"EPISODE MEAN: {np.mean(mean_values):.2e}")
        print(f"EPISODE STD DEV: {np.std(mean_values):.2e}")
        print(f"EPISODE Q1: {np.percentile(mean_values, 25):.2e}")
        print(f"EPISODE MEDIAN: {np.median(mean_values):.2e}")
        print(f"EPISODE Q3: {np.percentile(mean_values, 75):.2e}")

def model_usage():
    model_index = LOG_INDICES.index('model')

    for lambd in LAMBDAS:
        lambd_log = experiment_log['runs'][lambd]

        model_cnts = dict()
        for i in range(NUM_STEPS):
            model = lambd_log[i][model_index]
            model_cnts[model] = model_cnts.get(model, 0) + 1

        print(f"LAMBDA {lambd}")
        for model in model_cnts:
            print(f"{model}: {model_cnts[model] * 100 / NUM_STEPS:.2f}%")


def ovr_time_by_y(relative_radius = 0.05, num_points = 100, err_sd = 2.0):

    state_index = LOG_INDICES.index('state')
    pt_index = LOG_INDICES.index('planning_time')
    dt_index = LOG_INDICES.index('decision_time')
    ovr_time = lambda log: log[pt_index] + log[dt_index]

    plt.gcf().set_size_inches(8, 4)
    for lambd_index, lambd in enumerate(LAMBDAS):
        lambd_log = experiment_log['runs'][lambd]
        datapoints = []

        min_y, max_y = float('inf'), -float('inf')
        for i in range(NUM_STEPS):
            y_value = lambd_log[i][state_index][0][1]
            min_y, max_y = min(min_y, y_value), max(max_y, y_value)

            datapoints.append((y_value, ovr_time(lambd_log[i])))

        datapoints = sorted(datapoints, key = lambda x: x[0])

        radius = relative_radius * (max_y - min_y)
        spacing = (max_y - min_y) / num_points

        mean_time = []
        sd_time = []

        y = min_y + spacing/2

        start = 0
        end = 0
        total_time = 0

        for i in range(num_points):
            while (start < len(datapoints) and datapoints[start][0] < (y - radius)):
                start += 1

            while (end < len(datapoints) and datapoints[end][0] < (y + radius)):
                end += 1

            times = [datapoints[i][1] for i in range(start, end)]
            mean_time.append(np.mean(times))
            sd_time.append(np.std(times) / np.sqrt(end - start))
            y += spacing 

        mean_time, sd_time = np.array(mean_time), np.array(sd_time)
        y_values = np.arange(min_y + spacing/2, max_y, spacing)
        plt.plot(y_values, mean_time, color = COLORS[lambd_index])
        err_time = err_sd * sd_time
        plt.fill_between(y_values, mean_time - err_time, mean_time + err_time, 
                         color = COLORS[lambd_index], alpha = 0.5)

    plt.title(f"Overall Time by Y Position")
    legend_elements = [Line2D([0], [0], color=COLORS[0], lw=6, label=f'\u03BB: {LAMBDAS[0]}'),
                       Line2D([0], [0], color=COLORS[1], lw=6, label=f'\u03BB: {LAMBDAS[1]}'),]
    plt.legend(handles=legend_elements, loc='upper right', framealpha=1.0, 
           frameon = False, fontsize = 14)
    plt.xlabel("Y position")
    plt.ylabel(f"Avg. Overall Comp Time (s)")
    plt.gcf().subplots_adjust(top=0.9)
    plt.gcf().subplots_adjust(left=0.12)
    plt.gcf().subplots_adjust(right=0.98)
    plt.gcf().subplots_adjust(bottom=0.2)
    #plt.tight_layout()
    plt.savefig(f'y_time.png')
    plt.show()


