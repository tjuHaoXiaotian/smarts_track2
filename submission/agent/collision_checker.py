#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：SMARTS
@File    ：behavior_planner.py
@Author  ：Hao Xiaotian
@Date    ：2022/11/7 1:56
'''

from math import cos, pi, sin, sqrt

import numpy as np
import scipy.spatial

# COLLISION_BUFFER_AHEAD = 20
# COLLISION_BUFFER_BEHIND = 5

COLLISION_BUFFER_AHEAD = 12
COLLISION_BUFFER_BEHIND = 5
COLLISION_BUFFER_BEHIND_MIN = 0
COLLISION_BUFFER_TIME = 1

print("COLLISION_BUFFER_AHEAD={}, COLLISION_BUFFER_BEHIND={}, COLLISION_BUFFER_BEHIND_MIN={}".format(
    COLLISION_BUFFER_AHEAD, COLLISION_BUFFER_BEHIND, COLLISION_BUFFER_BEHIND_MIN))


class CollisionChecker:
    def __init__(self, circle_offsets, circle_radii, weight):
        self._circle_offsets = np.asarray(circle_offsets)
        self._circle_radii = np.asarray(circle_radii)
        self._weight = weight

    ######################################################
    ######################################################
    # MODULE 7: CHECKING FOR COLLISSIONS
    #   Read over the function comments to familiarize yourself with the
    #   arguments and necessary variables to return. Then follow the TODOs
    #   (top-down) and use the surrounding comments as a guide.
    ######################################################
    ######################################################
    # Takes in a set of paths and obstacles, and returns an array
    # of bools that says whether or not each path is collision free.
    def collision_check(self, paths, obstacles):
        """Returns a bool array on whether each path is collision free.

        args:
            paths: A list of paths in the global frame.  
                A path is a list of points of the following format:
                    [x_points, y_points, t_points]:
                        x_points: List of x values (m)
                        y_points: List of y values (m)
                        t_points: List of yaw values (rad)
                    Example of accessing the ith path, jth point's t value:
                        paths[i][2][j]
            obstacles: A list of [x, y] points that represent points along the
                border of obstacles, in the global frame.
                Format: [[x0, y0],
                         [x1, y1],
                         ...,
                         [xn, yn]]
                , where n is the number of obstacle points and units are [m, m]

        returns:
            collision_check_array: A list of boolean values which classifies
                whether the path is collision-free (true), or not (false). The
                ith index in the collision_check_array list corresponds to the
                ith path in the paths list.
        """

        if len(obstacles) == 0:
            return [1] * len(paths)

        collision_check_array = np.zeros(len(paths), dtype=bool)
        for i in range(len(paths)):
            collision_free = True
            path = paths[i]

            # Iterate over the points in the path.
            for j in range(len(path[0])):
                # Compute the circle locations along this point in the path.
                # These circle represent an approximate collision
                # border for the vehicle, which will be used to check
                # for any potential collisions along each path with obstacles.

                # The circle offsets are given by self._circle_offsets.
                # The circle offsets need to placed at each point along the path,
                # with the offset rotated by the yaw of the vehicle.
                # Each path is of the form [[x_values], [y_values],
                # [theta_values]], where each of x_values, y_values, and
                # theta_values are in sequential order.

                # Thus, we need to compute:
                # circle_x = point_x + circle_offset*cos(yaw)
                # circle_y = point_y circle_offset*sin(yaw)
                # for each point along the path.
                # point_x is given by path[0][j], and point _y is given by
                # path[1][j]. 
                circle_locations = np.zeros((len(self._circle_offsets), 2))

                # --------------------------------------------------------------
                circle_locations[:, 0] = path[0][j] + self._circle_offsets * cos(path[2][j])
                circle_locations[:, 1] = path[1][j] + self._circle_offsets * sin(path[2][j])
                # --------------------------------------------------------------

                # Assumes each obstacle is approximated by a collection of
                # points of the form [x, y].
                # Here, we will iterate through the obstacle points, and check
                # if any of the obstacle points lies within any of our circles.
                # If so, then the path will collide with an obstacle and
                # the collision_free flag should be set to false for this flag
                collision_dists = scipy.spatial.distance.cdist(obstacles, circle_locations)
                collision_dists = np.subtract(collision_dists, self._circle_radii)
                collision_free = collision_free and \
                                 not np.any(collision_dists < 0)

                if not collision_free:
                    break

            collision_check_array[i] = collision_free

        return collision_check_array

    def batch_collision_check(self, ego_speed, batch_paths, neighbor_speeds, batch_obstacles):
        """Returns a bool array on whether each path is collision free.
        args:
            batch_paths: [B, T, 3]
            batch_obstacles: [T, K, 4, 2]
        returns:
            collision_check_array: A list of boolean values which classifies
                whether the path is collision-free (true), or not (false).
        """

        if len(batch_obstacles) == 0:
            return [1] * len(batch_paths)

        collision_check_array = np.zeros(len(batch_paths), dtype=bool)
        collision_ahead_ego = round(COLLISION_BUFFER_AHEAD + ego_speed * COLLISION_BUFFER_TIME)
        collision_behind_ego = max(round(COLLISION_BUFFER_BEHIND - ego_speed * COLLISION_BUFFER_TIME),
                                   COLLISION_BUFFER_BEHIND_MIN)
        for i in range(len(batch_paths)):
            collision_free = True
            path = batch_paths[i]  # [T, 3]
            for t in range(len(path)):
                start_t_ego, end_t_ego = max(t - collision_behind_ego, 0), min(t + collision_ahead_ego, len(path) - 1)

                _sub_path = path[start_t_ego: end_t_ego + 1]  # [L, 3]
                _point_locations = np.expand_dims(_sub_path[:, :2], 1)  # [L, 2]->[L, 1, 2]
                _point_yaws = np.expand_dims(_sub_path[:, 2:], 1)  # [L, 1]->[L, 1, 1]
                _heading_vector = np.concatenate([np.cos(_point_yaws), np.sin(_point_yaws)], axis=2)  # [L, 1, 2]
                _circle_offsets = np.reshape(self._circle_offsets, [1, len(self._circle_offsets), 1])  # [1, N, 1]
                # [L, N, 2] = [L, 1, 2] + [1, N, 1] * [L, 1, 2]
                circle_locations = _point_locations + _circle_offsets * _heading_vector  # [L, N, 2]

                start_t_neigh, end_t_neigh = max(t - COLLISION_BUFFER_BEHIND, 0), min(t + COLLISION_BUFFER_AHEAD,
                                                                                      len(path) - 1)
                obstacles = batch_obstacles[start_t_neigh: end_t_neigh + 1].reshape(-1, 2)  # [L * K * 4, 2]

                collision_dists = scipy.spatial.distance.cdist(circle_locations.reshape(-1, 2), obstacles).reshape(
                    end_t_ego - start_t_ego + 1, len(self._circle_offsets), -1
                )  # [L, N, L * K * 4]

                # print(obstacles.shape)
                # print(circle_locations.shape)
                # print(collision_dists.shape)
                collision_dists = collision_dists - self._circle_radii.reshape(1, len(self._circle_offsets), 1)
                collision_free = collision_free and not np.any(collision_dists < 0)
                if not collision_free:
                    break

            collision_check_array[i] = collision_free
        return collision_check_array

    ######################################################
    ######################################################
    # MODULE 7: SELECTING THE BEST PATH INDEX
    #   Read over the function comments to familiarize yourself with the
    #   arguments and necessary variables to return. Then follow the TODOs
    #   (top-down) and use the surrounding comments as a guide.
    ######################################################
    ######################################################
    # Selects the best path in the path set, according to how closely
    # it follows the lane centerline, and how far away it is from other
    # paths that are in collision. 
    # Disqualifies paths that collide with obstacles from the selection
    # process.
    # collision_check_array contains True at index i if paths[i] is
    # collision-free, otherwise it contains False.
    def select_best_path_index(self, paths, collision_check_array, goal_state):
        """Returns the path index which is best suited for the vehicle to
        traverse.

        Selects a path index which is closest to the center line as well as far
        away from collision paths.

        args:
            paths: A list of paths in the global frame.  
                A path is a list of points of the following format:
                    [x_points, y_points, t_points]:
                        x_points: List of x values (m)
                        y_points: List of y values (m)
                        t_points: List of yaw values (rad)
                    Example of accessing the ith path, jth point's t value:
                        paths[i][2][j]
            collision_check_array: A list of boolean values which classifies
                whether the path is collision-free (true), or not (false). The
                ith index in the collision_check_array list corresponds to the
                ith path in the paths list.
            goal_state: Goal state for the vehicle to reach (centerline goal).
                format: [x_goal, y_goal, v_goal], unit: [m, m, m/s]
        useful variables:
            self._weight: Weight that is multiplied to the best index score.
        returns:
            best_index: The path index which is best suited for the vehicle to
                navigate with.
        """
        best_index = None
        best_score = float('Inf')
        for i in range(len(paths)):
            # Handle the case of collision-free paths.
            if collision_check_array[i]:
                # Compute the "distance from centerline" score.
                # The centerline goal is given by goal_state.
                # The exact choice of objective function is up to you.
                # A lower score implies a more suitable path.
                # --------------------------------------------------------------
                score = np.sqrt((paths[i][0][-1] - goal_state[0]) ** 2 + (paths[i][1][-1] - goal_state[1]) ** 2)
                # --------------------------------------------------------------

                # # Compute the "proximity to other colliding paths" score and
                # # add it to the "distance from centerline" score.
                # # The exact choice of objective function is up to you.
                # for j in range(len(paths)):
                #     if j == i:
                #         continue
                #     else:
                #         if not collision_check_array[j]:
                #             # --------------------------------------------------
                #             # score += self._weight * ...
                #             # score += self._weight * np.sqrt((paths[i][0][-1]-paths[j][0][-1])**2+(paths[i][1][-1]-paths[j][1][-1])**2) * -1
                #             # --------------------------------------------------
                #             pass

            # Handle the case of colliding paths.
            else:
                score = float('Inf')

            # Set the best index to be the path index with the lowest score
            if score < best_score:
                best_score = score
                best_index = i

        return best_index

    def get_ego_car_positions(self, x, y, yaw):
        circle_locations = np.zeros((len(self._circle_offsets), 2))

        # --------------------------------------------------------------
        circle_locations[:, 0] = x + self._circle_offsets * cos(yaw)
        circle_locations[:, 1] = y + self._circle_offsets * sin(yaw)
        return circle_locations
