import gym
import torch
from pathlib import Path
from typing import Any, Dict

import sys

sys.path.insert(0, str(Path(__file__).parents[0]))
sys.path.insert(0, str(Path(__file__).parents[1] / "train"))

from planner import Planner
from networks import SocialVehiclePredictor


class BasePolicy:
    def act(self, obs: Dict[str, Any]):
        raise NotImplementedError


class ResetNoticeWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(ResetNoticeWrapper, self).__init__(env)
        self.agent_ids = set()

    def observation(self, raw_obs_dict: dict):
        wrapped_obs = {}
        for agent_id, raw_obs in raw_obs_dict.items():
            wrapped_obs[agent_id] = {
                "raw_obs": raw_obs,
                "reset": False,
            }
            if agent_id not in self.agent_ids:
                wrapped_obs[agent_id]["reset"] = True
                self.agent_ids.add(agent_id)
        return wrapped_obs

    def reset(self):
        self.agent_ids = set()
        return super().reset()


def submitted_wrappers():
    return [ResetNoticeWrapper]


class Policy(BasePolicy):
    def __init__(self):
        super().__init__()
        self.env_planners = dict()
        self.sv_predictor = SocialVehiclePredictor().cuda()

        self.sv_predictor.load_state_dict(
            torch.load(Path(__file__).parents[0] / "SVP.pt")
        )

    def act(self, obs: dict):
        if obs is {}:
            return {}

        a = {}
        for agent_id, raw_obs_and_reset_flag in obs.items():
            raw_obs, reset_flag = raw_obs_and_reset_flag["raw_obs"], raw_obs_and_reset_flag["reset"]
            if agent_id not in self.env_planners.keys() or reset_flag:
                self.env_planners[agent_id] = Planner(self.sv_predictor)

            a[agent_id] = self.env_planners[agent_id].get_action(raw_obs)
        return a
