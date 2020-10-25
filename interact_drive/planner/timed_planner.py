"""Base class for planners that track computational time."""

from interact_drive.planner.car_planner import CarPlanner
import time


class TimedPlanner(object):
    """
    Parent class for all the planners that support computational
    time tracking.
    """

    def __init__(self, sections = ['decision', 'planning', 'overall']):
        '''
        We track the amount of time used in each step of planning
        dividing into potentially overlapping sections.
        '''
        self.sections = sections
        self.section_start_flag = {s: None for s in self.sections}
        self.section_nested_count = {s: 0 for s in self.sections}
        self.section_step_times = {s: [] for s in self.sections}
        self.avg_section_times = {s: [] for s in self.sections}

        self.checkpoint_recorded = None
        self.cumulative_rollback = {s: 0 for s in self.sections}

    def get_comp_times(self):
        '''
        Returns a dictionary which maps section names to lists
        who's length is the number of time steps elapsed in the 
        current experiment.
        The ith element indicates the time taken for the specific
        section during the computation of the ith robot step.

        Currently sections are 'overall' and 'decision', where planning
        time may be extracted as the difference of the two.
        '''
        return self.section_step_times

    def get_avg_comp_times(self):
        '''
        Returns a dictionary which maps section names to lists
        who's length is the number of time steps elapsed in the 
        current experiment.
        The ith element indicates the average time taken for the specific
        section during the computation of the first i robot steps.

        Currently sections are 'overall' and 'decision', where planning
        time may be extracted as the difference of the two.
        '''
        return self.avg_section_times

    def avg_comp_time_breakdown(self):
        '''
        Returns a string indicating the average computation
        overall and amount of time taken for the last step
        again overall.
        '''
        if (len(self.avg_section_times['overall']) == 0):
            return "N/A"
        last = self.section_step_times['overall'][-1]
        avg = self.avg_section_times['overall'][-1]
        mcp_str = f"LAST: {last:.2f}; OVERALL: {avg:.2f}"
        return mcp_str

    def avg_comp_time_overall(self):
        '''
        Returns the overall average computation time
        '''
        ovr_total_num = len(self.computation_times)
        ovr_total_time = sum(self.computation_times)

        if (ovr_total_num == 0):
            return 0

        ovr_avg = ovr_total_time / ovr_total_num

        return ovr_avg

    def record_section_time(self, point_type, section = None):
        '''
        Method used for all computational time tracking.  At the beginning
        of a timestep we initialize the counters.  Everytime a section starts
        we flag the start time and when it ends increment the computational
        time accordingly.  When step t is complete we additionally compute
        average computation time for the first t time steps.

        Note if you sub_start when you previously have there is no effect.
        If you sub_end a second time then only the later one's effect will 
        count.  This is to allow nested sub intervals where only the outermost
        one has effect.
        '''

        if (point_type == 'step_start'):
            for s in self.sections:
                self.section_step_times[s].append(0)
            self.record_section_time('sub_start', 'overall')

        if (point_type == 'sub_start'):
            if (self.section_start_flag[section] is None):
                self.section_start_flag[section] = time.time()
            else:
                self.section_nested_count[section] += 1

        elif (point_type == 'sub_end'):
            if (self.section_nested_count[section] > 0):
                self.section_nested_count[section] -= 1
            else:
                time_elapsed = time.time() - self.section_start_flag[section]
                step_time = time_elapsed - self.cumulative_rollback[section]
                self.section_step_times[section][-1] += step_time
    
                self.section_start_flag[section] = None
                self.cumulative_rollback[section] = 0

        elif (point_type == 'step_end'):
            self.record_section_time('sub_end', 'overall')
            for s in self.sections:
                total_time = sum(self.section_step_times[s])
                num_steps = len(self.section_step_times[s])
                avg_time = total_time / num_steps
                self.avg_section_times[s].append(avg_time)


    def checkpoint(self, point_type):
        '''
        point_type: 'record'
            Records checkpoint timestamp.
        point_type: 'rollback'
            Rollbacks to last checkpoint by deducting the time
            between from sections currently active.  (If sections
            completed between the checkpoint and rollback they will
            be unaffected by this.)
        '''
        if (point_type == 'record'):
            self.checkpoint_recorded = time.time()
        
        elif (point_type == 'rollback'):
            if (self.world.verbose):
                print("Computational Time Rollback")
            rollback_time = time.time() - self.checkpoint_recorded
            for s in self.sections:
                if (self.section_start_flag[s] is not None):
                    self.cumulative_rollback[s] += rollback_time

    

