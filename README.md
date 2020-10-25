# Model Switching

Implementation of Human Model Switching experiments in a Self-Driving Car simulator.  Experiments present a variety of situations to the robot car, 
and during portions of each it's necessary to have a more complex human model.  But at other times, a less expensive model is more than sufficient.
The robot's task it to dynamically switch between these models online to optimize the reward-compute tradeoff in real time.  This  involves only
using more complex models when they are likely to provide additional reward worth the computational expensive, and switching to less expensive ones
whenever this is not the case.  To get a sense for how these experiments look, see the gifs stored under the experiments folder.

The experiments folder holds the files responsible for launching experiments, as well as the stored results.  The interact-drive folder holds all the
code for the simulator as well as planning and switching algorithms.

Work on this project was done in association with the <a href = "http://interact.berkeley.edu/"> InterACT Lab </a> at Berkeley.
