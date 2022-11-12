#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：SMARTS 
@File    ：social_car_tracker.py
@Author  ：Hao Xiaotian
@Date    ：2022/11/11 16:27 
'''
import math
import numpy as np
from collections import deque
from numpy.polynomial import Polynomial
from math import sin, cos, atan2

POLYNOMIAL_DEGREE = 1
TRACK_TRAJ_LENGTH = 3
MINIMUM_DISTANCE_TO_FOLLOWER = 10
# MIN_COS_SIMILARITY = 1 / math.sqrt(2)
MIN_COS_SIMILARITY = math.sqrt(3) / 2
print("TRACK_TRAJ_LENGTH={}, MINIMUM_DISTANCE_TO_FOLLOWER={}, MIN_COS_SIMILARITY={}".format(TRACK_TRAJ_LENGTH,
                                                                                            MINIMUM_DISTANCE_TO_FOLLOWER,
                                                                                            MIN_COS_SIMILARITY))


def clip_yaw(theta):
    if theta > math.pi:
        theta -= 2 * math.pi
    elif theta < -math.pi:
        theta += 2 * math.pi
    return theta


def convert_heading(smarts_heading):
    """
    SMARTS uses radians, 0 is facing north (y-axis), and turn counter-clockwise. i.e., -pi/2 is facing east (x-axis)
    """
    return clip_yaw(smarts_heading + math.pi / 2)


def inverse_heading(converted_heading):
    return clip_yaw(converted_heading - math.pi / 2)


class SocialCarTracker():
    def __init__(self, track_radius=33):
        self.track_radius = track_radius
        self.tracked_car_info = {}

    def check_for_following_vehicle(self, neighbor_state, min_dot_product=MIN_COS_SIMILARITY):
        follow_car_delta_vector = self.ego_state.position[:2] - neighbor_state.position[:2]
        follow_car_distance = np.linalg.norm(follow_car_delta_vector)
        follow_car_delta_vector = follow_car_delta_vector / follow_car_distance

        lead_car_yaw = convert_heading(self.ego_state.heading)
        lead_car_heading_vector = [cos(lead_car_yaw), sin(lead_car_yaw)]
        return np.dot(follow_car_delta_vector,
                      lead_car_heading_vector) > min_dot_product, follow_car_distance, follow_car_delta_vector

    def the_following_car_is_about_to_rear_end(self):
        for id, info in self.tracked_car_info.items():
            is_following, distance, delta_vector = self.check_for_following_vehicle(info[0], 0)
            if is_following and distance < MINIMUM_DISTANCE_TO_FOLLOWER:
                print(
                    "Car at lane-{} approaches the ego car at lane-{}".format(info[0].lane_id, self.ego_state.lane_id))
                return True, info[0].speed, delta_vector
        return False, 0, None

    def update(self, ego_state, neighborhood_vehicle_states):
        self.ego_state = ego_state
        current_env_cars = set()
        for vehicle_state in neighborhood_vehicle_states:
            current_env_cars.add(vehicle_state.id)
            distance = np.linalg.norm(vehicle_state.position[:2] - ego_state.position[:2])
            is_following_car = self.check_for_following_vehicle(vehicle_state)[0]
            if distance < self.track_radius and not is_following_car:
                if not self.tracked_car_info.get(vehicle_state.id, None):
                    self.tracked_car_info[vehicle_state.id] = [vehicle_state, deque(maxlen=TRACK_TRAJ_LENGTH)]
                    self.tracked_car_info[vehicle_state.id][1].appendleft(vehicle_state)
                else:
                    previous_state = self.tracked_car_info[vehicle_state.id][1][0]
                    delta_distance = np.linalg.norm(vehicle_state.position[:2] - previous_state.position[:2])
                    if delta_distance > 1:  # 只记录有变化的趋势
                        self.tracked_car_info[vehicle_state.id][1].appendleft(vehicle_state)
                    self.tracked_car_info[vehicle_state.id][0] = vehicle_state
            else:
                if self.tracked_car_info.get(vehicle_state.id, None):
                    self.tracked_car_info.pop(vehicle_state.id)

        # 删除不在env中的：
        recorded_ids = list(self.tracked_car_info.keys())
        for id in recorded_ids:
            if id not in current_env_cars:
                self.tracked_car_info.pop(id)
        # if vehicle_state.id in self.tracked_car_info:
        #     print(vehicle_state.id, distance, "is_following_car={}".format(is_following_car), convert_heading(self.ego_state.heading),
        #           self.ego_state.position[:2] - vehicle_state.position[:2], vehicle_state.speed, self.tracked_car_info[vehicle_state.id][0].speed)

    def _predict_location(self, neighbor_state_t, historical_trajectory, delta_t):
        # [N, 2]
        positions = np.asarray([state.position[:2] for state in historical_trajectory])
        yaw = convert_heading(neighbor_state_t.heading)
        speed = neighbor_state_t.speed
        x, y = neighbor_state_t.position[0], neighbor_state_t.position[1]
        delta_vector = None

        if speed > 0 and delta_t > 0:
            speed_delta_x, speed_delta_y = speed * cos(yaw) * 0.1, speed * sin(yaw) * 0.1
            new_x = x + speed_delta_x
            new_y = y + speed_delta_y
            positions = np.concatenate([np.array([[new_x, new_y]]), positions], axis=0)  # [N+1, 2]
            # 三次样条插值
            cumu_distance, target_distance = 0, speed * delta_t
            if abs(speed_delta_x) < abs(speed_delta_y):
                # inverse the x and y
                f = Polynomial.fit(x=positions[:, 1], y=positions[:, 0], deg=min(len(positions), POLYNOMIAL_DEGREE))
                while cumu_distance < target_distance:
                    y += speed_delta_y
                    predicted_x = f(y)
                    delta_vector = [predicted_x - x, speed_delta_y]
                    delta_distance = np.linalg.norm(delta_vector)
                    cumu_distance += delta_distance
                    x = predicted_x

            else:
                f = Polynomial.fit(x=positions[:, 0], y=positions[:, 1], deg=min(len(positions), POLYNOMIAL_DEGREE))
                while cumu_distance < target_distance:
                    x += speed_delta_x
                    predicted_y = f(x)
                    delta_vector = [speed_delta_x, predicted_y - y]
                    delta_distance = np.linalg.norm(delta_vector)
                    cumu_distance += delta_distance
                    y = predicted_y

        new_x, new_y, new_heading = x, y, atan2(delta_vector[1], delta_vector[0]) if delta_vector else yaw

        length, width = neighbor_state_t.bounding_box.length, neighbor_state_t.bounding_box.width
        _theta = atan2(width, length)
        half_diagonal = width / cos(_theta) / 2

        _theta_1 = new_heading - _theta
        x1 = new_x + half_diagonal * cos(_theta_1)  # 右下
        y1 = new_y + half_diagonal * sin(_theta_1)

        _theta_2 = new_heading + _theta
        x2 = new_x + half_diagonal * cos(_theta_2)  # 右上
        y2 = new_y + half_diagonal * sin(_theta_2)

        x3 = new_x - half_diagonal * cos(_theta_1)  # 左上
        y3 = new_y - half_diagonal * sin(_theta_1)

        x4 = new_x - half_diagonal * cos(_theta_2)  # 左下
        y4 = new_y - half_diagonal * sin(_theta_2)
        return [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

    def _just_starting_to_turn(self, pos_trajectory):
        start_x, start_y = pos_trajectory[0][0], pos_trajectory[0][1]
        start_yaw = atan2(start_y, start_x)
        for point in pos_trajectory[1:]:
            x, y = point[0], point[1]
            yaw = atan2(y, x)
            if abs(yaw - start_yaw) >= math.pi / 5:
                return True
        return False

    def _predict_location_traj(self, neighbor_state_t, historical_trajectory, delta_t_list, predicted_traj):
        # [N, 2]
        positions = np.asarray([state.position[:2] for state in historical_trajectory])

        if predicted_traj is not None:
            # We currently only incorporate the first point predicted by the model.
            predicted_first_point = predicted_traj[:1, :2]
            positions = np.concatenate([predicted_first_point, positions], axis=0)

        yaw = convert_heading(neighbor_state_t.heading)
        speed = neighbor_state_t.speed
        x, y = neighbor_state_t.position[0], neighbor_state_t.position[1]
        speed_delta_x, speed_delta_y = speed * cos(yaw) * 0.1, speed * sin(yaw) * 0.1

        # %%%%%%%%%%%%%%%%%%% Bounding box related %%%%%%%%%%%%%%%%%%
        length, width = neighbor_state_t.bounding_box.length, neighbor_state_t.bounding_box.width
        _theta = atan2(width, length)
        half_diagonal = width / cos(_theta) / 2
        # %%%%%%%%%%%%%%%%%%% Bounding box related %%%%%%%%%%%%%%%%%%
        if speed > 0.1:
            new_x = x + speed_delta_x
            new_y = y + speed_delta_y
            positions = np.concatenate([np.array([[new_x, new_y]]), positions], axis=0)  # [N+1, 2]

            # if self._just_starting_to_turn(positions):
            #     polynomial_degree = 2
            # else:
            #     polynomial_degree = POLYNOMIAL_DEGREE

            # 插值函数
            if abs(speed_delta_x) < abs(speed_delta_y):
                # inverse the x and y
                f = Polynomial.fit(x=positions[:, 1], y=positions[:, 0], deg=min(len(positions), POLYNOMIAL_DEGREE))
            else:
                f = Polynomial.fit(x=positions[:, 0], y=positions[:, 1], deg=min(len(positions), POLYNOMIAL_DEGREE))

            obstacles, cumulative_distance = [], 0
            for delta_t in delta_t_list:
                target_distance = speed * delta_t
                if abs(speed_delta_x) < abs(speed_delta_y):
                    while cumulative_distance < target_distance:
                        y += speed_delta_y
                        predicted_x = f(y)
                        delta_vector = [predicted_x - x, speed_delta_y]
                        delta_distance = np.linalg.norm(delta_vector)
                        cumulative_distance += delta_distance
                        x = predicted_x

                else:
                    while cumulative_distance < target_distance:
                        x += speed_delta_x
                        predicted_y = f(x)
                        delta_vector = [speed_delta_x, predicted_y - y]
                        delta_distance = np.linalg.norm(delta_vector)
                        cumulative_distance += delta_distance
                        y = predicted_y

                # %%%%%%%%%%%%%%%%%%%%%%%%% Computing bounding box %%%%%%%%%%%%%%%%%%%%%%
                new_x, new_y, new_heading = x, y, atan2(delta_vector[1], delta_vector[0])
                _theta_1 = new_heading - _theta
                _theta_2 = new_heading + _theta
                x1 = new_x + half_diagonal * cos(_theta_1)  # 右下
                y1 = new_y + half_diagonal * sin(_theta_1)
                x2 = new_x + half_diagonal * cos(_theta_2)  # 右上
                y2 = new_y + half_diagonal * sin(_theta_2)
                x3 = new_x - half_diagonal * cos(_theta_1)  # 左上
                y3 = new_y - half_diagonal * sin(_theta_1)
                x4 = new_x - half_diagonal * cos(_theta_2)  # 左下
                y4 = new_y - half_diagonal * sin(_theta_2)
                obstacles += [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            return obstacles
        else:  # speed == 0
            # %%%%%%%%%%%%%%%%%%%%%%%%% Computing bounding box %%%%%%%%%%%%%%%%%%%%%%
            new_x, new_y, new_heading = x, y, yaw
            _theta_2 = new_heading + _theta
            _theta_1 = new_heading - _theta
            x1 = new_x + half_diagonal * cos(_theta_1)  # 右下
            y1 = new_y + half_diagonal * sin(_theta_1)
            x2 = new_x + half_diagonal * cos(_theta_2)  # 右上
            y2 = new_y + half_diagonal * sin(_theta_2)
            x3 = new_x - half_diagonal * cos(_theta_1)  # 左上
            y3 = new_y - half_diagonal * sin(_theta_1)
            x4 = new_x - half_diagonal * cos(_theta_2)  # 左下
            y4 = new_y - half_diagonal * sin(_theta_2)
            return [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

    def predict_neighbor_positions(self, delta_t):
        obstacle_points = []
        for id, info in self.tracked_car_info.items():
            obstacle_points += self._predict_location(info[0], info[1], delta_t)
        return obstacle_points

    def predict_neighbor_position_traj(self, delta_t_list):
        obstacle_points = []
        for id, info in self.tracked_car_info.items():
            obstacle_points += self._predict_location_traj(info[0], info[1], delta_t_list, None)
        return obstacle_points

    def predict_neighbor_position_traj_2d(self, predicted_trajectories, delta_t_list):
        obstacle_points, obstacle_speeds = [], []
        for id, info in self.tracked_car_info.items():
            predicted_traj = predicted_trajectories.get(id, None)
            car_traj = np.asarray(self._predict_location_traj(info[0], info[1], delta_t_list, predicted_traj))  # [T, 4]
            car_traj = car_traj.reshape([-1, 4, 2])
            if len(car_traj) == 1:
                car_traj = np.tile(car_traj, (len(delta_t_list), 1, 1))  # [1, 4, 2] -> [T, 4, 2]
            obstacle_points.append(car_traj)
            obstacle_speeds.append(info[0].speed)
        if len(obstacle_points) > 0:
            return np.stack(obstacle_points, axis=1), obstacle_speeds  # [T, K, 4, 2]
        return obstacle_points, obstacle_speeds
