"""Base class for driving scenarios."""

from typing import Dict, Iterable, List, Optional, Tuple

import tensorflow as tf
import numpy as np
import pickle


class CarWorld(object):
    """
    Contains the objects in a driving scenario - cars, lanes, obstacles, etc.

    In addition, contains a step() function that increments the state of the
    environment over time.

    Finally, this class provides visualizations for the environment.
    """

    def __init__(self, dt: float = 0.1, lanes: Optional[List] = None,
                 obstacles: Optional[List] = None,
                 visualizer_args: Optional[Dict] = None,
                 experiment_args = None,
                 **kwargs):
        """
        Initializes this CarWorld. Note: the visualizer is *not* initialized
        until the first render() call.

        Args:
            dt: the time increment per tick of simulation.
            lanes: a list of lanes this world should contain.
            obstacles: a list of obstacles this world should contain.
                       Formatted as [(obs_type_name, (x, y)), ...]
            visualizer_args: a dict of arguments for the visualizer.
            **kwargs:
        """

        if (experiment_args):
            self.exp_name = experiment_args.exp_name
            self.verbose = experiment_args.verbose
        else:
            self.exp_name = None
            self.verbose = True

        self.cars = []
        self.dt = dt

        if lanes is None:
            self.lanes = []
        else:
            self.lanes = lanes
            self.lane_medians_t = tf.cast(tf.stack([l.p for l in lanes]), dtype = tf.float32)
            self.lane_normals_t = tf.cast(tf.stack([l.n for l in lanes]), dtype = tf.float32)

        if obstacles is None:
            self.obstacles = []
            self.obstacle_states_t = None
        else:
            self.obstacles = obstacles


        if visualizer_args is None:
            self.visualizer_args = dict()
        else:
            self.visualizer_args = visualizer_args

        self.visualizer = None

        self.parsed_obj_atts = False
        self.obj_atts_cache = pickle.load(open("cache/obj_atts.pkl", "rb"))
        if (self.exp_name in self.obj_atts_cache):
            self.parse_obj_atts(self.obj_atts_cache[self.exp_name])

        self.past_states = []

    def add_car(self, car):
        car.index = len(self.cars)
        self.cars.append(car)

    def add_cars(self, cars: Iterable):
        for car in cars:
            self.add_car(car)
            

    def add_clones(self):
        '''
        Adds clones of all cars, for augmenting rollout
        visualizations.  You will be able to see where
        the car actually is and where it envisions it can
        be simultaneously as a result.
        '''
        from interact_drive.car import Car
        self.car_clones = []
        for car in self.cars:
            self.car_clones.append(Car(self, car.state, car.color,
                                   car.opacity * 0.35))

    @property
    def state(self):
        return [c.state for c in self.cars]

    @property
    def last_state(self):
        return self.past_states[-1]
    

    @state.setter
    def state(self, new_state: Iterable):
        for c, x in zip(self.cars, new_state):
            c.state = x

    def reset(self, seed = None):
        for car in self.cars:
            car.reset(seed)

        self.past_states = []

        if (self.visualizer is None and (self.verbose or not self.parse_obj_atts)):
            self.initialize_visualizer()
        elif self.verbose:
            self.visualizer.reset()

        if not self.parsed_obj_atts:
            self.parse_obj_atts(self.visualizer.obj_atts)
            if (self.exp_name is not None):
                self.obj_atts_cache[self.exp_name] = self.visualizer.obj_atts
                pickle.dump(self.obj_atts_cache, open("cache/obj_atts.pkl", "wb"))


    def revert_state(self, t):
        '''
        Set the state of the world to what it was t time steps ago.
        Pop off last t past states.  The last one to be popped is
        the new current state.
        '''
        for i in range(t):
            s = self.past_states.pop()
        self.state = s



    def step(self, dt: Optional[float] = None,
             trying_rollout = False) -> Tuple[List[tf.Tensor],
                                        List[tf.Tensor],
                                        List[tf.Tensor]]:
        """
        Asks all cars to generate plans, and then updates the world state based
        on those plans

        We need to split the plan generation and car state updating because
        all the cars act at once (in the simulation)

        Args:
            dt: the amount of time to increment the simulation forward by.
            trying_rollout: If maintaining clones, then clones will mirror
                            the actual car's position, not following their
                            counterpart iff just trying rollout.

        Returns:
            past_state: the previous state of the world, before this tick.
            controls: the controls applied to all the cars in this timestep.
            state: the current state of the world.
        """
        past_state = self.state

        if dt is None:
            dt = self.dt

        for car in self.cars:
            if not car.control_already_determined_for_current_step:
                car.set_next_control()

        for car in self.cars:
            car.step(dt)

        if (not trying_rollout and hasattr(self, "car_clones")):
            for i in range(len(self.cars)):
                self.car_clones[i].state = self.cars[i].state

        self.past_states.append(past_state)

        return past_state, [c.control for c in self.cars], self.state

    def get_time_step(self):
        # Returns the time step of the world (1, 2, 3 ...)
        return len(self.past_states) + 1

    def initialize_visualizer(self):
        from interact_drive.visualizer import CarVisualizer
        self.visualizer = CarVisualizer(world=self, **self.visualizer_args)
        self.visualizer.set_main_car(index=0)

    def render(self, mode: str = "human") -> Optional[np.array]:
        """
        Renders the state of this car world. If mode="human", we display
        it using the visualizer. If mode="rgb_array", we return a np.array
        with shape (x, y, 3) representing RGB values, useful for making gifs
        and videos.

        Note: we currently assume that the main car is the first car in
            self.cars.

        Args:
            mode: One of ["human", "rgb_array"].

        Returns:
            rgb_representation: if str="rgb_array", we return an np.array of
                    shape (x, y, 3), representing the rendered image.
        """
        if self.visualizer is None:
            self.initialize_visualizer()

        if mode == "human":
            self.visualizer.render(display=True, return_rgb=False)
        elif mode == "rgb_array":
            return self.visualizer.render(display=False, return_rgb=True)
        else:
            raise ValueError("Mode must be either `human` or `rgb_array`.")

    def parse_obj_atts(self, obj_atts):
        '''
        Populates dictionaries containing dimensions
        for all objects in the world.
        '''


        '''
        Gathers the following information for all non car
        objects (obstacles) classes from the visualizer:
        - width and height
        - orientation
        - offsets for corners (boxes)
        '''
        self.obj_whs = dict()
        self.obj_angs = dict()
        self.obj_boxes = dict()
        for obj_name in obj_atts:
            w, h = obj_atts[obj_name]["wh"]
            self.obj_whs[obj_name] = tf.constant([[w], [h]], dtype = tf.float32)
            self.obj_boxes[obj_name] = tf.constant([[-w/2, -w/2, w/2, w/2],
                                                    [-h/2, h/2, h/2, -h/2]],
                                                    dtype = tf.float32)
            ang = obj_atts[obj_name]["ang"]
            if (ang is not None):
                self.obj_angs[obj_name] = tf.constant(ang, dtype = tf.float32)

        '''
        Gathers the width and height of cars and the offsets for their
        corners from the visualizer.
        '''
        self.car_wh = self.obj_whs["car"]
        self.car_box = self.obj_boxes["car"]

        '''
        Populates lists that contain the locations, angles,
        width, and height of all obstacles in the world.

        Currently only stationary obstacles are supported. 
        '''
        self.obs_locs = []
        self.obs_wh = []
        self.obs_ang = []
        for (obs_type, pos) in self.obstacles: 
            self.obs_locs.append(tf.reshape(pos, [2, 1]))
            self.obs_wh.append(self.obj_whs[obs_type])
            self.obs_ang.append(self.obj_angs[obs_type])

        self.parsed_obj_atts = True


class TwoLaneCarWorld(CarWorld):
    """
    A car world initialized with three straight lanes that extend for
    a long while in either direction.
    """

    def __init__(self, **kwargs):
        lane = StraightLane((0.0, -5.), (0.0, 10.), 0.1)
        lanes = [lane.shifted(1), lane]
        super().__init__(lanes=lanes, **kwargs)

        self.left_limit = -0.1
        self.right_limit = 0.0

class ThreeLaneCarWorld(CarWorld):
    """
    A car world initialized with three straight lanes that extend for
    a long while in either direction.
    """

    def __init__(self, **kwargs):
        lane = StraightLane((0.0, -5.), (0.0, 10.), 0.1)
        lanes = [lane.shifted(1), lane, lane.shifted(-1)]
        super().__init__(lanes=lanes, **kwargs)

        self.left_limit = -0.1
        self.right_limit = 0.1


class TwoLaneConeCarWorld(CarWorld):
    """
    A car world initialized with two straight lanes that extend for
    a long while in either direction.

    In addition cones are placed at the specified locations.
    """

    def __init__(self, cone_positions = [], **kwargs):
        if (len(cone_positions) > 0):
            cones = [("cone", cp) for cp in cone_positions]
        else:
            cones = None
        lane = StraightLane((0.0, -5.), (0.0, 10.), 0.14)
        lanes = [lane.shifted(1), lane]
        super().__init__(lanes=lanes, obstacles = cones, **kwargs)

        self.left_limit = -0.14
        self.right_limit = 0.0


class FourLaneTruckCarWorld(CarWorld):
    """
    A car world initialized with four straight lanes that extend for
    a long while in either direction.

    In addition stationary trucks are placed at the specified
    loacations.
    """

    def __init__(self, truck_positions = [], **kwargs):
        if (len(truck_positions) > 0):
            trucks = [("truck", tp) for tp in truck_positions]
        else:
            trucks = None
        lane = StraightLane((0.0, -5.), (0.0, 10.), 0.1)
        lanes = [lane.shifted(2), lane.shifted(1), lane, lane.shifted(-1)]
        super().__init__(lanes=lanes, obstacles = trucks, **kwargs)

        self.left_limit = -0.2
        self.right_limit = 0.1


class StraightLane(object):
    """
    Defines a lane with median defined by the line segment between points
    p and q, and width w.
    """

    def __init__(self, p: Tuple[float, float], q: Tuple[float, float],
                 w: float):
        """
        Initializes the straight lane.

        Args:
            p: the x,y coordinates of the start point for the center of the lane
            q: the x,y coordinates of the end point for the center of the lane
            w: the width of the lane
        """

        self.p = np.asarray(p)
        self.q = np.asarray(q)
        self.w = w

        self.m = (self.q - self.p) / np.linalg.norm(
            self.q - self.p)  # unit vector in direction of lane
        self.n = np.asarray(
            [-self.m[1], self.m[0]])  # normal vector to the lane

    def shifted(self, n_lanes: int):
        """
        Returns a lane that is shifted n_lanes in the direction of self.n.

        When n_lanes < 0, this is shifted in the other direction instead.

        Args:
            n_lanes: number of lanes to shift

        Returns:
            (StraightLane): a straight lane shifted in the appropriate way.

        """
        return StraightLane(self.p + self.n * self.w * n_lanes,
                            self.q + self.n * self.w * n_lanes, self.w)

    def dist2median(self, point: Tuple[float]):
        """
        Returns the squared distance of a point to the median of the lane.

        Args:
            point: the x,y coordinates of the point.

        Returns:
            (float): the distance to the median of this lane
        """
        r = ((point[0] - self.p[0]) * self.n[0]
             + (point[1] - self.p[1]) * self.n[1])
        return r ** 2

    def on_road(self, point):
        raise NotImplementedError
