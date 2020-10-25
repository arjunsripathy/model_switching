'''
Displays results manually provided for each experiment, analyzing
the computation time and reward tradeoff for different models and
switchers.  Further provides 90% confidence bounds around reported
values, and normalizes the relative scale of reward and compute.

Run the following command to display results for an experiment.
python display_results.py <EXP NAME>

where <EXP NAME> is one of stay_back, merger, or give_way.
'''

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
import sys

l = '\u03BB'
num_runs = 30

if (sys.argv[1] == 'stay_back'):
    EXP_NAME = 'Stay Back'
    names = ["Naive", f"MS {l}: 5.0", f"MS {l}: 0.4", f"Turn"]
    P = [0, 3]
    MS = [1, 2]
    rewards = [(-7.73,7.69), (2.34,7.73e-1), (2.59,4.38e-2), (2.52,5.72e-1)]
    comp_times = [(1.05e-1,0,3.62e-03), (1.07e-1,9.49e-3,3.54e-3), (1.23e-1,8.84e-3,3.52e-3), 
                  (2.12e-1,0,5.37e-3)]
elif (sys.argv[1] == 'merger'):
    EXP_NAME = 'Merger'
    names = ["Naive", f"MS {l}: 0.6", f"MS {l}: 0.1", f"Turn", "ToM"]
    P = [0, 3, 4]
    MS = [1, 2]
    rewards = [(1.55,1.28e-2), (3.53,1.76), (5.54, 7.77e-1), (1.54,1.40e-2), (5.33,3.07)]
    comp_times = [(8.44e-2,0,5.91e-03), (1.16e-1,1.34e-2,2.93e-2), (1.45e-1,1.77e-2,1.96e-2),
                  (1.63e-1,0,6.36e-3), (3.80e-1,0,1.36e-2)]
elif (sys.argv[1] == 'give_way'):
    EXP_NAME = 'Give Way'
    names = ["Naive", f"Turn", f"MS {l}: 0.03", f"MS {l}: 0.01", "ToM"]
    P = [0, 1, 4]
    MS = [2, 3]
    rewards = [(-1.78e-2,6.23e-1), (3.35e-1,9.93e-1), (1.78, 2.18), (4.44,7.01e-1), (4.91,2.33)]
    comp_times = [(9.21e-2,0,3.47e-4), (1.81e-1,0,9.19e-4), (1.73e-1,1.99e-2,5.61e-2),
                  (2.54e-1,2.92e-2,2.62e-2), (4.48e-1,0,1.86e-2)]
else:
    raise Exception(f"Invalid Experiment Identifier {sys.argv[1]}")

def get(values, indices):
    return [values[i] for i in indices]
plt.rcParams.update({'hatch.linewidth': 3, 
                     'font.family': 'palatino',
                     'font.size': 18,
                     'figure.figsize': (8, 6)})
P_STYLE = {'alpha': 0.5}

N = len(names)
bar_width = 0.25
bar_spacing = bar_width * 2 + 0.3

time_x = [bar_spacing * i for i in range(N)]
rew_x = [bar_width + bar_spacing * i for i in range(N)]
min_x = min(time_x) - bar_width
max_x = max(rew_x) + bar_width

# 90% confidence interval
ERR_SCALE = 1.645 / np.sqrt(num_runs)
plan_time_bars = [ct[0] for ct in comp_times]
ovr_time_bars = [ct[0] + ct[1] for ct in comp_times]
time_confs = [ct[2] * ERR_SCALE for ct in comp_times]
rew_bars = [r[0] for r in rewards]
rew_confs = [r[1] * ERR_SCALE for r in rewards] 

fig, ax1 = plt.subplots()
plt.title(f"{EXP_NAME}: Average Comp. Time and Reward")

PAD_TOP = 0.25
PAD_BOT = 0.25

min_time = ovr_time_bars[0]
max_time = ovr_time_bars[-1]
min_time -= (max_time - min_time) * PAD_BOT
max_time += (max_time - min_time) * PAD_TOP

ax1.set_xlim(min_x, max_x)
ax1.set_ylim(min_time, max_time)
ax1.set_ylabel("Avg. Overall Comp Time (s)")
ax1.set_xticks(np.array(time_x) + bar_width * 0.5)
ax1.set_xticklabels(names)

ax1.bar(get(time_x, MS), get(ovr_time_bars, MS), width = bar_width, 
        color = 'cornflowerblue', yerr = get(time_confs, MS), capsize = 6.0)
ax1.bar(get(time_x, MS), get(plan_time_bars, MS), width = bar_width, color = 'gray')

ax1.bar(get(time_x, P), get(ovr_time_bars, P), width = bar_width, 
        color = 'cornflowerblue', yerr = get(time_confs, P), capsize = 6.0, **P_STYLE)
ax1.bar(get(time_x, P), get(plan_time_bars, P), width = bar_width, color = 'gray', **P_STYLE)

rew_shift = rew_bars[0]
max_rew = rew_bars[-1]
rew_shift -= (max_rew - rew_shift) * PAD_BOT
max_rew += (max_rew - rew_shift) * PAD_TOP
rew_bars = [r - rew_shift for r in rew_bars]

ax2 = ax1.twinx()
ax2.set_ylim(0, max_rew - rew_shift)
ax2.set_ylabel("Avg. Reward")
low = int(rew_shift) - rew_shift + (1 if rew_shift > 0 else 0)
step = max(int((max_rew - rew_shift)/6), 1)
yticks = [low]
while ((yticks[-1] + step) < (max_rew - rew_shift)):
  yticks.append(yticks[-1] + step)
ax2.set_yticks(yticks)
ax2.set_yticklabels([f"{v + rew_shift:.0f}" for v in yticks])

ax2.bar(get(rew_x, MS), get(rew_bars, MS), width = bar_width, 
        color = 'orange', yerr = get(rew_confs, MS), capsize = 6.0)

ax2.bar(get(rew_x, P), get(rew_bars, P), width = bar_width, 
        color = 'orange', yerr = get(rew_confs, P), capsize = 6.0, **P_STYLE)

ax2.hlines(rew_bars[0] , min_x, max_x, lw = 1, linestyle = 'dashed')
ax2.hlines(rew_bars[-1], min_x, max_x, lw = 1, linestyle = 'dashed')

legend_elements = [Line2D([0], [0], color='orange', lw=6, label='Reward'),
                   Line2D([0], [0], color='gray', lw=6, label='Planning Time'),
                   Line2D([0], [0], color='cornflowerblue', lw=6, label='Decision Time'),]

ax2.legend(handles=legend_elements, loc='upper left', framealpha=1.0, 
           frameon = False, fontsize = 14)

plt.tight_layout()
plt.gcf().savefig(f'{EXP_NAME}_results.png')
plt.show()




