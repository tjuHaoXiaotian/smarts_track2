import sys
from pathlib import Path
from typing import Any, Dict, Optional

import gym
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parents[0]))
sys.path.insert(0, str(Path(__file__).parents[1] / "train"))

from networks import SocialVehiclePredictor
from planner import Planner


def load_config(path: Path) -> Optional[Dict[str, Any]]:
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config

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
        train_config = load_config(Path(__file__).absolute().parents[0] / "config.yaml")
        
        device = train_config["device"]
        self.env_planners = dict()
        self.sv_predictor = SocialVehiclePredictor(device=device).to(device)

        try:
            self.sv_predictor.load_state_dict(
                torch.load(Path(__file__).parents[0] / "SVP.pt", map_location=torch.device(device))
            )
        except:
            pass

    def act(self, obs: dict):
        if obs is {}:
            return {}

        a = {}
        for agent_id, raw_obs_and_reset_flag in obs.items():
            raw_obs, reset_flag = raw_obs_and_reset_flag["raw_obs"], raw_obs_and_reset_flag["reset"]
            if agent_id not in self.env_planners.keys() or reset_flag:
                self.env_planners[agent_id] = Planner(self.sv_predictor)

            try:
                a[agent_id] = self.env_planners[agent_id].get_action(raw_obs)
            except:
                # If any error occurs during planning, just stay still.
                a[agent_id] = [
                    *raw_obs.ego_vehicle_state.position[:2],
                    raw_obs.ego_vehicle_state.heading - 0,
                    0.1
                ]
        return a


if __name__ == "__main__":
    env = gym.make(
        "smarts.env:multi-scenario-v0",
        scenario="3lane_merge_multi_agent",
        sumo_headless=True,
        visdom=False
    )
    for wrapper in submitted_wrappers():
        env = wrapper(env)
    policy = Policy()
    
    obs = env.reset()
    done = {"__all__": False}
    
    while not done["__all__"]:
        act = policy.act(obs)
        obs, reward, done, info = env.step(act)
        