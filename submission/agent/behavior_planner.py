#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：SMARTS 
@File    ：behavior_planner.py
@Author  ：Hao Xiaotian
@Date    ：2022/11/7 1:56 
'''

# State machine states
FOLLOW_LANE = 0
FOLLOW_LEAD_VEHICLE = 1
DECELERATE_TO_STOP = 2


class BehaviouralPlanner:
    def __init__(self):
        self._state = None

    def transit_to_line_tracking(self):
        self.at_crossing = False
        self.decelerate_to_stop = False
        self.line_tracking = True

    def transit_to_at_crossing(self):
        self.line_tracking = False
        self.at_crossing = True

    def transit_to_decelerate_to_stop(self):
        self.line_tracking = False
        self.decelerate_to_stop = True

    def print_state(self, target_velocity, has_collusion_in_current_lane):
        print(
            "\033[1;45m line_tracking={}, at_crossing={}, decelerate_to_stop={}, target_velocity={}, has_collusion_in_current_lane={}\033[0m".format(
                self.line_tracking,
                self.at_crossing,
                self.decelerate_to_stop, target_velocity, has_collusion_in_current_lane
            ))
