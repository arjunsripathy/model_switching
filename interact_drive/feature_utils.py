"""Defines useful features for feature-based cars.
"""

import tensorflow as tf
import numpy as np


@tf.function
def lane_allignment(car_state, lane_medians, lane_normals):
    '''
    The lane allignment features is a combination
    of two components designed to measure how well
    alligned a car is to the lanes of the road.
    The components are:
        - median_deviance: squared distance to closest
          median.  The lower this value the more centered
          a car is in its lane.
        - heading_allignment: All lanes in our experiment
          are oriented due north.  The more alligned a car
          is with this direction the greater this feature.

    This feature is intended to be associated with a positive
    weight, encouraging the car to stay centered in its lane 
    and head due north.

    Args:
        car_state: the state of the car
        lane_medians: tensor containing the median positions of the lanes
        lane_normals: tensor containing the normal vectors to the lanes.

    Returns:
        tf.Tensor: allignment feature
    '''

    '''
    The below constant allows us to provide the  weight
    of the second component (heading_allignment) relative
    to the importance of the first (median_deviance)
    '''
    #HEADING_K = 5e-3
    HEADING_K = 1e-2

    sqd_lane_dists = tf.reduce_sum(((car_state[:2] - lane_medians) * lane_normals) ** 2, axis = 1)
    closest_median = tf.reduce_min(sqd_lane_dists, axis=-1)

    median_deviance = -closest_median
    heading_allignment = tf.sin(car_state[3])

    allignment_feat = median_deviance + HEADING_K * heading_allignment
    return allignment_feat




'''
Collision Risk Algorithm Overview:

The collision risk feature is the combination
of two penalties.  Both of these penalties are
computed with respect to every other vehicle
and obstacle.  The vehicle/obstacle that results
in the greatest combination of these penalties
is deemed to pose the greatest collision risk
and the corresponding value is returned.

Now we'll discuss the two penalties.

**************************************************************
Proximity Penalty: 
The proximity penalty intends to provide a sharply 
increasing reward signal as two bodies come close or 
even overlap (which is certainly possible in this simulation).

Suppose we are computing the collision risk of
Car A with respect to Object B.  First we linearly
transform coordinates of the world so that the bounding
box of Object B is a square with height and width 2.
As a result the bounding box of Object B is simply
the unit norm l-infinity ball centered at the same
location.

Now for every point in A we can calculate it's l-infinity
distance to the center of Object B and get a sense of how
close it is to collision (l-infinity distance < 1 => collision).

We could solve an optimization and find the point in A
that minimizes the l-infinity distance to the center of B.
This gives us how close A is to colliding with B.  However
we feel this optimization is not worth the computation,
and instead approximate this value by only considering a
select few points in A and taking the minimum over them.

In order to ensure that our points cover the whole body of A
and give us a good approximation we impose a 
(GRAN + 1) x (GRAN + 1) grid over the bounding box of A considering
each of those points as candidates for closest to B's center. The 
larger GRAN is the more accurate this approximation is guaranteed
to be.

To efficiently calculate these (GRAN + 1)^2 locations in A
to compare with the normalized object B we construct the below matrix,
GRID, which allows us to linearly transform the positions of the corners
of A into the (GRAN + 1)^2 grid points.

Finally once we have the approximate minimum l-infinity distance, d,
we use this to arrive at an exponential penalty of the form
                e^(COLLISION_RISK_K * (d - 1))  
This exponential is designed to be 0 when the objects are 
just touching and sharply increase as they overlap.  Likewise 
it sharply decays as the objects get away from one another.

For better reward shaping we actually compute this exponential penalty
for each one of the grid points and sum over them.  It will still be
similar since the exponential nature greatly devalues farther points.

In addition we incur a significant penalty if d ever goes below 1
indicating an actual collision.  The above provides reward shaping for
this true goal.
'''


'''
Here we compute the above referenced GRID which helps
us efficiently compute proximity penalty.
'''

ll = np.array([1, 0, 0 , 0])
x = np.array([-1, 0, 0 , 1])
y = np.array([-1, 1, 0, 0])

'''
R of np.sqrt(2)/2 results in the corner grid points
at the corner of the car's bounding box.
R of 0.5 results in the corner grid points tracing
an eliptical shape in accord with the front, back,
and side points.
'''
R = 0.675
R2 = R / np.sqrt(2)
points = [(-R2, R2),  (0, 0.5),  (R2, R2), 
          (-0.5, 0),    (0, 0),  (0.5, 0),
          (-R2, -R2), (0, -0.5), (R2, -R2)]
grid = np.transpose([ll + (0.5 + xd) * x + (0.5 + yd) * y for (xd, yd) in points])

GRID = tf.constant(grid, dtype = tf.float32)


'''
Constant that controls how sharply the proximity
penalty changes as the bodies get very close
and as they begin to overlap.

Greater constant => Harsher penalty for mild collision
                    Lesser penalty for bare avoidance
'''
CAR_RISK_K = 1.5
OBS_RISK_K = 4.0
OBS_W = 0.5
COLLISION_K = 1e2


@tf.function
def collision_risk(state, car_state, car_index, car_wh, car_box, 
                   obs_locs = [], obs_wh = [], obs_ang = [],
                   consider_other_cars = True):
    '''
    This feature returns a greater value the more at risk
    the target car is to be in collision with another car 
    or obstacle.

    Args:
        car_state
        car_index
        car_wh: Tensor containing the width and height of the car.
        car_box: Tensor containing the offsets of car corners.
        obs_locs: Tensor of object locations. (Empty list if None)
        obs_wh: Tensor containing the width and height of obstacles
        obs_ang: Tensor containing the orientations of obstacles.
        consider_other_cars: If False then ignore the other cars
            for this feature (only worry about obstacles if any).

    Returns:
        tf.Tensor: collision risk feature
    '''

    '''
    Calculate corners of the target car by rotating the standard 
    bounding box corners to allign with the current heading and 
    shifting by the center's position.
    '''

    car_vel = car_state[2]

    car_cos_t = tf.cos(car_state[3])
    car_sin_t = tf.sin(car_state[3])
    car_rotation = [[car_cos_t, -car_sin_t], [car_sin_t, car_cos_t]]

    car_pos = tf.reshape(car_state[:2], [2, 1])

    car_corners = tf.matmul(car_rotation, car_box) + car_pos

    all_col_risks = []
    all_min_sqd_dists = []

    if (consider_other_cars):
        #Consider all other cars...
        for i in range(len(state)):
            if (i == car_index):
                continue

            '''
            norm_grid_pts will contain the position of the grid
            points of the target car with respect to the other
            car.  Further the coordinates will be normalized so
            that
                - 1 along the first coordinate indicates 0.5 car width
                  right of the center of the other car
                - 1 along the second coordinate indicates 0.5 car length
                  behind the center of the other car.
            'right' and 'behind' refer to the perpendicular axis that
            are based on the heading of the other car.
            '''
            other_pos = tf.reshape(state[i][:2], [2, 1])
            other_vel = state[i][2]
            other_cos_t = tf.cos(state[i][3])
            other_sin_t = tf.sin(state[i][3])
            inv_rotation = [[other_cos_t, other_sin_t], [-other_sin_t, other_cos_t]]

            rel_corners = car_corners - other_pos
            norm_corners = tf.matmul(inv_rotation, (rel_corners * 2)) / car_wh
            norm_grid_pts = tf.matmul(norm_corners, GRID)

            '''
            We consider the squared L-infinity distance between the 
            grid points and the the center of the car.
            '''
            other_sqd_dists = tf.reduce_sum(norm_grid_pts ** 2, axis = 0)

            '''
            Minimum normalized squared distance of any of the grid 
            points to the center of the other car.
            '''
            min_other_sqd_dist = tf.reduce_min(other_sqd_dists)
            all_min_sqd_dists.append(min_other_sqd_dist)

            '''
            Proximity penalty based on all of the squared distances, not just
            the least, but the exponential nature heavily weighs the least.
            '''
            proximity_pen = tf.reduce_sum(tf.exp(-CAR_RISK_K * (other_sqd_dists - 1)))
            all_col_risks.append(proximity_pen)

    for (loc, wh, ang) in zip(obs_locs, obs_wh, obs_ang):
        # Identical logic as before, but now for obstacles.
        obs_cos_t = tf.cos(ang)
        obs_sin_t = tf.sin(ang)
        inv_rotation = [[obs_cos_t, obs_sin_t], [-obs_sin_t, obs_cos_t]]

        rel_corners = car_corners - loc
        norm_corners = tf.matmul(inv_rotation, (rel_corners * 2)) / wh
        norm_grid_pts = tf.matmul(norm_corners, GRID)

        #other_sqd_dists = tf.reduce_max(norm_grid_pts ** 2, axis = 0)
        other_sqd_dists = tf.reduce_sum(norm_grid_pts ** 2, axis = 0)
        min_other_sqd_dist = tf.reduce_min(other_sqd_dists)
        all_min_sqd_dists.append(min_other_sqd_dist)

        proximity_pen = OBS_W * tf.reduce_sum(tf.exp(-OBS_RISK_K * (other_sqd_dists - 1)))
      
        all_col_risks.append(proximity_pen)

    
    # first component is reward shaping component, second is harsh step penalty for an actual collision
    collision = tf.stop_gradient(tf.cast(tf.reduce_min(all_min_sqd_dists) < 1, tf.float32))
    return tf.reduce_sum(all_col_risks) + COLLISION_K * collision


'''
Parameter controlling penalty for driving off road.
We begin to penalize the car if it is more than
OFF_ROAD_BUFFER further out than the leftmost
or rightmost lane medians.
'''
OFF_ROAD_BUFFER = 0.05

@tf.function
def off_road(car_state, car_box, left_limit = -0.1, right_limit = 0.1):
    '''
    Args:
        car_state: the state of the car
        car_box: Tensor containing the offsets of car corners
        left_limit: Central x value of leftmost lane median
        right_limit: Central x value of rightmost lane median


    Returns:
        tf.Tensor: off road feature 0 if in road, linearly positive if off road  
    '''

    cos_t = tf.cos(car_state[3])
    sin_t = tf.sin(car_state[3])
    car_rotation = [[cos_t, -sin_t], [sin_t, cos_t]]

    car_pos = tf.reshape(car_state[:2], [2, 1])

    car_corners_x = (tf.matmul(car_rotation, car_box) + car_pos)[0]

    off_road_amt = tf.reduce_max([tf.reduce_max(car_corners_x) - (right_limit + OFF_ROAD_BUFFER),
                                 (left_limit - OFF_ROAD_BUFFER) - tf.reduce_min(car_corners_x),
                                 0.]) ** 2

    return off_road_amt


@tf.function
def forward_velocity(car_state):
    '''
    Args:
        car_state: the state of the car

    Returns:
        tf.Tensor: forward velocity
    '''
    forward_velocity_feat = car_state[2] * tf.sin(car_state[3])
    
    return forward_velocity_feat


@tf.function
def speeding(car_state, speed_limit = 0.6):
    '''
    Args:
        car_state: the state of the car

    Returns:
        tf.Tensor: amount above the maximum desired speed, speed_limit
    '''
    penalty = tf.maximum(0., car_state[2] * tf.sin(car_state[3]) - speed_limit)
    
    return penalty

@tf.function
def speed_desirability(car_state, speed_limit = 0.6):
    '''
    Args:
        car_state: the state of the car

    Returns:
        tf.Tensor: Parabolic desirability which is maximal (1.0) at speed_limit.
    '''
    f_vel = car_state[2] * tf.sin(car_state[3])
    desirability = -f_vel * (f_vel - 2 * speed_limit) / (speed_limit ** 2)

    return desirability
