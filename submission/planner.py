#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：SMARTS
@File    ：bot.py
@Author  ：Hao Xiaotian
@Date    ：2022/11/3 21:16
'''

import math

import numpy as np

from agent.local_planner import LocalPlanner, transform_paths
from agent.social_car_tracker import clip_yaw, convert_heading, inverse_heading, SocialCarTracker

LOOKAHEAD_LENGTH = 30
LOOKAHEAD_TIME = 2
MIN_SPEED = 5  # m/s
MAX_ACCELERATION = 100  # m/s^2
FORCE_TO_GOAL_DISTANCE = 50
CIRCLE_RADII = 2.4
print("LOOKAHEAD_LENGTH={}, LOOKAHEAD_TIME={}, FORCE_TO_GOAL_DISTANCE={}, MIN_SPEED={}, CIRCLE_RADII={}".format(
    LOOKAHEAD_LENGTH,
    LOOKAHEAD_TIME,
    FORCE_TO_GOAL_DISTANCE,
    MIN_SPEED,
    CIRCLE_RADII))


class Planner(object):

    def __init__(self, social_car_predictor):
        self.debug = False
        self.agent_name = "Agent_0"
        self.local_planner = LocalPlanner(
            num_paths=7,
            path_offset=1.5,  # m
            # circle_offsets=[-2, 0, 2],  # m
            # circle_radii=[1.5, 1.5, 1.5],  # m
            circle_offsets=[0],  # m
            circle_radii=[CIRCLE_RADII],  # m
            path_select_weight=10,
            time_gap=1.0,  # s
            a_max=1.5,  # m/s^2
            slow_speed=2.0,  # m/s
            stop_line_buffer=3.5,  # m
        )
        self.social_car_tracker = SocialCarTracker()
        self.social_car_predictor = social_car_predictor

    def print(self, *args):
        if self.debug:
            print(self.agent_name, ": {}".format(args))

    def generate_candidate_speeds(self, current_speed, max_speed, min_speed):
        _speed = max(current_speed, min_speed)
        return [min(_speed * 1.5, max_speed), min(_speed * 1.25, max_speed), _speed, _speed * 0.8, _speed * 0.6]

    def get_action(self, raw_obs):
        """
        [x-coordinate, y-coordinate, heading, and time-delta]
        action_space = gym.spaces.Box(low=np.array([-1e10, -1e10, -π, 0]), high=np.array([1e10, 1e10, π, 1e10]), dtype=np.float32)
        """
        ego_info = raw_obs.ego_vehicle_state
        self.social_car_tracker.update(ego_info, raw_obs.neighborhood_vehicle_states)

        # Using pre-trained model to predict the future trajectories of the nearing social cars.
        predicted_trajectories = self.social_car_predictor.predict_with_raw_obs(raw_obs)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Select a path for ego car. %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        current_x, current_y = ego_info.position[:2]
        current_yaw = convert_heading(ego_info.heading)
        current_speed = ego_info.speed
        ego_state = [current_x, current_y, current_yaw, current_speed]

        waypoint_path, candidate_lines, target_lane_index, force_to_goal = get_ego_waypoints(
            ego_info, raw_obs.waypoint_paths, raw_obs.road_waypoints)

        try:
            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Generate the goal sets. %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            max_speed = waypoint_path[0].speed_limit
            planning_traj_length = LOOKAHEAD_LENGTH + max(current_speed, MIN_SPEED) * LOOKAHEAD_TIME
            # [x, y, max_speed, yaw]
            _goal_state, planning_traj_length = get_goal_state(waypoint_path, lookahead_length=planning_traj_length)

            # Compute the goal state set: [[x, y, yaw, max_speed], ...]
            goal_state_set = self.local_planner.get_goal_state_set(_goal_state, ego_state, candidate_lines)
            has_multiple_candidate_paths = False
            if len(goal_state_set) > 1 or force_to_goal:  # 有多条路径时，才需要换道
                has_multiple_candidate_paths = True
                # Calculate planned paths in the local frame. path: [x, y, heading]
                paths, path_validity, _ = self.local_planner.plan_paths(goal_state_set, target_lane_index)
                # Transform those paths back to the global frame.
                paths = transform_paths(paths, ego_state)
                if len(paths) == 0:
                    has_multiple_candidate_paths = False
                    paths = build_paths_using_waypoint_path(waypoint_path, lookahead_length=planning_traj_length)
            else:  # 只有一条路径，无需sample
                paths = build_paths_using_waypoint_path(waypoint_path, lookahead_length=planning_traj_length)

            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Perform collision checking. %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            for speed_config in self.generate_candidate_speeds(current_speed, max_speed, MIN_SPEED):
                planning_step = planning_traj_length * 10 // speed_config
                # 0 - 0.1 * planning_step s均不能发生碰撞
                time_list = np.arange(1, planning_step) * 0.1
                # [T, K, 4, 2]
                batch_neighbor_trajectory, neighbor_speeds = self.social_car_tracker.predict_neighbor_position_traj_2d(
                    predicted_trajectories, time_list)
                # [B, T, 3]
                batch_ego_trajectory = predict_ego_position_traj(ego_info, paths, speed_config, time_list)
                collision_check_array = self.local_planner._collision_checker.batch_collision_check(
                    ego_info.speed,
                    batch_ego_trajectory,  # 最保守的策略（理论上，应该计算对应各个时刻是预估的ego position和预估的social car position否会碰撞）
                    neighbor_speeds,
                    batch_neighbor_trajectory
                )
                # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Compute the best local path. %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                best_index = self.local_planner._collision_checker.select_best_path_index(paths, collision_check_array,
                                                                                          _goal_state)
                if best_index is not None:
                    break

            if best_index is not None:
                best_path = paths[best_index]
                target_distance = speed_config * 0.1
                target_x, target_y, target_yaw = get_target_at(best_path, ego_info.position[:2], target_distance)
                # %%%%%%%%%%%%%%%%%%% clip the planned position by road boundary %%%%%%%%%%%%%%%%%%%
                if has_multiple_candidate_paths:
                    target_x, target_y = clip_the_target_position_by_road_boundary(target_x, target_y,
                                                                                   raw_obs.waypoint_paths)
                return [target_x, target_y, target_yaw, 0.1]
            else:
                target_x, target_y = ego_info.position[:2]
                target_yaw = ego_info.heading

            return [target_x, target_y, target_yaw, 0.1]


        except Exception as e:
            print("There is an exception:", e)
            target_point = waypoint_path[min(1, len(waypoint_path) - 1)]
            target_x, target_y, target_yaw = target_point.pos[0], target_point.pos[1], target_point.heading
            return [target_x, target_y, target_yaw, 0.1]


def _predict_neighbor_positions(npc_states, delta_t):
    obstacle_points = []
    for vehicle_obs in npc_states:
        x, y = vehicle_obs.position[:2]
        yaw = convert_heading(vehicle_obs.heading)
        speed = vehicle_obs.speed
        speed_delta_x = speed * math.cos(yaw)
        speed_delta_y = speed * math.sin(yaw)
        x = x + speed_delta_x * delta_t
        y = y + speed_delta_y * delta_t

        length = vehicle_obs.bounding_box.length
        width = vehicle_obs.bounding_box.width
        theta = clip_yaw(math.atan2(width, length))
        half_diagonal = width / math.cos(theta) / 2

        theta_1 = yaw - theta
        x1 = x + half_diagonal * math.cos(theta_1)  # 右下
        y1 = y + half_diagonal * math.sin(theta_1)

        theta_2 = yaw + theta
        x2 = x + half_diagonal * math.cos(theta_2)  # 右上
        y2 = y + half_diagonal * math.sin(theta_2)

        x3 = x - half_diagonal * math.cos(theta_1)  # 左上
        y3 = y - half_diagonal * math.sin(theta_1)

        x4 = x - half_diagonal * math.cos(theta_2)  # 左下
        y4 = y - half_diagonal * math.sin(theta_2)
        obstacle_points += [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

    return obstacle_points


def predict_ego_positions(ego_info, paths, speed, delta_t):
    target_length = speed * delta_t
    path_fragments = []
    for path in paths:
        current_x, current_y = ego_info.position[:2]
        current_yaw = None
        cumulative_distance = 0
        for idx in range(len(path)):
            x, y, yaw = path[0][idx], path[1][idx], path[2][idx]
            cumulative_distance += np.linalg.norm([x - current_x, y - current_y])
            current_x, current_y, current_yaw = x, y, yaw
            if cumulative_distance > target_length:
                break
        path_fragment = [[current_x], [current_y], [current_yaw]]
        path_fragments.append(path_fragment)
    return path_fragments


def predict_ego_position_traj(ego_info, paths, set_speed, time_list):
    batch_trajectory = []
    for path in paths:
        cumulative_distance = 0
        current_x, current_y = ego_info.position[:2]
        current_yaw = None
        idx = 0
        path_fragments = []
        for delta_t in time_list:
            target_length = set_speed * delta_t
            while cumulative_distance < target_length:
                if idx == len(path[0]):
                    break
                x, y, yaw = path[0][idx], path[1][idx], path[2][idx]
                cumulative_distance += np.linalg.norm([x - current_x, y - current_y])
                current_x, current_y, current_yaw = x, y, yaw
                idx += 1
            path_fragments += [[current_x, current_y, current_yaw]]

        batch_trajectory.append(np.asarray(path_fragments))  # [T, 3]
    return np.stack(batch_trajectory, axis=0)  # [B, T, 3]


def is_endless_mission(mission):
    if hasattr(mission.goal, "position"):
        return False
    else:
        return True


def get_goal_position(mission):
    if not is_endless_mission(mission):
        return mission.goal.position[:2]
    return None


# def try_to_find_the_path_to_the_goal(goal_position, waypoint_paths, road_waypoints):
#     lanes = road_waypoints.lanes
#     minimal_distance, target_path = 100000000, None
#     for name, road_paths in lanes.items():
#         for road_path in road_paths:
#             end_waypoint_pos = road_path[-1].pos[:2]
#             distance_to_goal = np.linalg.norm(end_waypoint_pos - goal_position)
#             if distance_to_goal < minimal_distance:
#                 minimal_distance = distance_to_goal
#                 target_path = road_path
#
#     # TODO: road_waypoints可能有错误的点，要根据 waypoint_paths 去导航
#     # 3. 确定左右边界线，决定了能 lattice 是否能往左右两侧画
#     lane_indices, lane_widths, current_lane_index = [], [], None
#     target_path_positions = np.asarray([waypoint.pos[:2] for waypoint in target_path])
#     nearest_path, min_distance = None, 100000000
#     for waypoint_path in waypoint_paths:
#         if waypoint_path[0].lane_index not in lane_indices:
#             near_point_positions = np.asarray([waypoint.pos[:2] for waypoint in waypoint_path[:2]])
#             collision_dists = scipy.spatial.distance.cdist(near_point_positions,
#                                                            target_path_positions)  # [2, path_length]
#             distance = np.mean(np.min(collision_dists, axis=1))
#             if distance < min_distance:
#                 min_distance = distance
#                 nearest_path = waypoint_path
#                 current_lane_index = waypoint_path[0].lane_index
#             lane_indices.append(waypoint_path[0].lane_index)
#             lane_widths.append(waypoint_path[0].lane_width)
#
#     return nearest_path, lane_indices, lane_widths, current_lane_index

def try_to_find_the_path_to_the_goal(goal_position, waypoint_paths, road_waypoints):
    # TODO: road_waypoints可能有错误的点，要根据 waypoint_paths 去导航
    # 3. 确定左右边界线，决定了能 lattice 是否能往左右两侧画
    lane_indices, lane_widths, target_lane_index = [], [], None
    nearest_path, min_distance = None, 100000000
    for waypoint_path in waypoint_paths:
        if waypoint_path[0].lane_index not in lane_indices:
            end_waypoint_pos = waypoint_path[-1].pos[:2]
            distance_to_goal = np.linalg.norm(end_waypoint_pos - goal_position)
            if distance_to_goal < min_distance:
                min_distance = distance_to_goal
                nearest_path = waypoint_path
                target_lane_index = waypoint_path[0].lane_index
            lane_indices.append(waypoint_path[0].lane_index)
            lane_widths.append(waypoint_path[0].lane_width)

    return nearest_path, lane_indices, lane_widths, target_lane_index


def find_the_nearest_way_path_to_ego(ego_info, waypoint_paths):
    # 2. 选择距离当前车最近的路线
    ego_position = np.expand_dims(ego_info.position[:2], 0)
    nearest_path, min_distance = None, 100000000
    # 3. 确定左右边界线，决定了能 lattice 是否能往左右两侧画
    lane_indices, lane_widths, current_lane_index = [], [], None
    for path in waypoint_paths:
        if path[0].lane_index not in lane_indices:
            near_point_positions = np.asarray([waypoint.pos[:2] for waypoint in path[:2]])
            distance_to_lane = np.linalg.norm(near_point_positions - ego_position, axis=1).mean()
            if distance_to_lane < min_distance:
                min_distance = distance_to_lane
                nearest_path = path
                current_lane_index = path[0].lane_index
                # current_lane_width = path[0].lane_width
            lane_indices.append(path[0].lane_index)
            lane_widths.append(path[0].lane_width)
    return nearest_path, lane_indices, lane_widths, current_lane_index


def get_ego_waypoints(ego_info, waypoint_paths, road_waypoints):
    # (1) potential target info
    mission = ego_info.mission
    goal_position = get_goal_position(mission)

    # 1. 先确定是否有到达goal的路线
    force_to_goal = False
    if goal_position:
        nearest_path, lane_indices, lane_widths, target_lane_index = try_to_find_the_path_to_the_goal(
            goal_position, waypoint_paths, road_waypoints)
        distance_to_goal = np.linalg.norm(ego_info.position[:2] - goal_position)
        if distance_to_goal < FORCE_TO_GOAL_DISTANCE:
            force_to_goal = True
    else:
        nearest_path, lane_indices, lane_widths, target_lane_index = find_the_nearest_way_path_to_ego(
            ego_info, waypoint_paths)

    if force_to_goal:
        candidate_lines = [0]
        target_lane_index = 0
    else:
        # %%%%%%%%%%%%%%%%%%%%% 每条车道画2条线 %%%%%%%%%%%%%%%%%%%%%
        candidate_lines = [0, ]
        for lane_index, width in sorted(zip(lane_indices, lane_widths), key=lambda ele: ele[0]):
            candidate_lines += [width / 2] * 2
        target_lane_index = target_lane_index * 2 + 1
        candidate_lines = np.cumsum(candidate_lines)
        candidate_lines = candidate_lines - candidate_lines[target_lane_index]
        candidate_lines[0] += 0.3  # to ensure enough space at the boundary
        candidate_lines[-1] -= 0.3  # to ensure enough space at the boundary
        candidate_lines = candidate_lines[1:-1]  # ignore the boundary
        target_lane_index -= 1  # ignore the boundary

        # %%%%%%%%%%%%%%%%%%%%% 每条车道画4条线 %%%%%%%%%%%%%%%%%%%%%
        # candidate_lines = [0, ]
        # for lane_index, width in sorted(zip(lane_indices, lane_widths), key=lambda ele: ele[0]):
        #     candidate_lines += [width / 4] * 4
        # ego_line_idx = current_lane_index * 4 + 2
        # candidate_lines = np.cumsum(candidate_lines)
        # candidate_lines = candidate_lines - candidate_lines[ego_line_idx]
        # candidate_lines[0] += 0.3  # to ensure enough space at the boundary
        # candidate_lines[-1] -= 0.3  # to ensure enough space at the boundary
        # candidate_lines = candidate_lines[1:-1]  # ignore the boundary
        # ego_line_idx -= 1  # ignore the boundary
    return nearest_path, candidate_lines, target_lane_index, force_to_goal


def get_goal_state(waypoint_path, lookahead_length):
    length = 0
    previous_point = waypoint_path[0]
    for point in waypoint_path[1:]:
        length += np.linalg.norm(point.pos[:2] - previous_point.pos[:2])
        previous_point = point
        if length > lookahead_length:
            break
    # [x, y, max_speed, yaw]
    return [previous_point.pos[0], previous_point.pos[1], previous_point.speed_limit,
            convert_heading(previous_point.heading)], math.floor(length)


def build_paths_using_waypoint_path(waypoint_path, lookahead_length):
    length = 0
    previous_point = waypoint_path[0]
    xs, ys, yaws = [previous_point.pos[0]], [previous_point.pos[1]], [convert_heading(previous_point.heading)]
    for waypoint in waypoint_path[1:]:
        xs.append(waypoint.pos[0])
        ys.append(waypoint.pos[1])
        yaws.append(convert_heading(waypoint.heading))
        length += np.linalg.norm(waypoint.pos[:2] - previous_point.pos[:2])
        previous_point = waypoint
        if length > lookahead_length:
            break
    paths = [[xs, ys, yaws]]
    return paths


def the_speed_is_decreasing(ego_speed, neighbor_speed):
    return ego_speed > neighbor_speed


def distance_to_neighbor_is_far_enough(ego_pos, target_neighbor_id, all_neighbor_info):
    target_neighbor = None
    for neighbor in all_neighbor_info:
        if target_neighbor_id == neighbor.id:
            target_neighbor = neighbor
            break
    if target_neighbor is None:
        return True
    else:
        if np.linalg.norm(ego_pos - target_neighbor.position[:2]) > 20:
            return True
        return False


def is_at_the_crossing(waypoint_path):
    start_yaw = waypoint_path[0].heading
    for point in waypoint_path[1:]:
        yaw = point.heading
        if abs(yaw - start_yaw) >= math.pi / 4:
            return True
    return False


def get_lane_boundary(waypoint_paths):
    min_index, max_index = 10000, -10000
    right_lane, left_lane = None, None
    lane_indices = []
    for waypoint_path in waypoint_paths:
        if waypoint_path[0].lane_index not in lane_indices:
            lane_indices.append(waypoint_path[0].lane_index)

            if waypoint_path[0].lane_index < min_index:
                min_index = waypoint_path[0].lane_index
                right_lane = waypoint_path
            if waypoint_path[0].lane_index > max_index:
                max_index = waypoint_path[0].lane_index
                left_lane = waypoint_path
    return left_lane, right_lane


def clip_the_target_position_by_road_boundary(target_x, target_y, waypoint_paths):
    target_pos = np.array([target_x, target_y])
    left_lane, right_lane = get_lane_boundary(waypoint_paths)
    closest_point_idx_left = np.argmin(
        np.linalg.norm(np.asarray([point.pos[:2] for point in left_lane]) - target_pos, axis=1))
    closest_point_left = left_lane[closest_point_idx_left]

    closest_point_idx_right = np.argmin(
        np.linalg.norm(np.asarray([point.pos[:2] for point in right_lane]) - target_pos, axis=1))
    closest_point_right = right_lane[closest_point_idx_right]

    vector_left_1 = target_pos - closest_point_left.pos[:2]
    theta_left_1 = clip_yaw(np.arctan2(vector_left_1[1], vector_left_1[0]))
    theta_left_0 = convert_heading(closest_point_left.heading)
    if theta_left_1 - theta_left_0 > 0:
        target_x, target_y = closest_point_left.pos[:2]

    vector_right_1 = target_pos - closest_point_right.pos[:2]
    theta_right_1 = clip_yaw(np.arctan2(vector_right_1[1], vector_right_1[0]))
    theta_right_0 = convert_heading(closest_point_right.heading)
    if theta_right_1 - theta_right_0 < 0:
        target_x, target_y = closest_point_right.pos[:2]
    return target_x, target_y


def get_target_at(best_path, ego_position, target_distance):
    cumu_distance = 0
    previous_pos = ego_position
    target_idx = -1
    for x, y in zip(best_path[0], best_path[1]):
        cumu_distance += np.linalg.norm([x - previous_pos[0], y - previous_pos[1]])
        target_idx += 1
        previous_pos = [x, y]
        if cumu_distance >= target_distance:
            break
    return best_path[0][target_idx], best_path[1][target_idx], inverse_heading(best_path[2][target_idx])
