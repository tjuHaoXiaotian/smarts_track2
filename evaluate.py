import random
import sys
from pathlib import Path
from typing import Any, Dict

import gym
import numpy as np
from tqdm import tqdm

# To import submission folder
sys.path.insert(0, str(Path(__file__).parents[0]))

from submission.policy import submitted_wrappers, Policy


def evaluate(config):
    # Make evaluation environments.
    envs_eval = {}
    policy = Policy()
    for scenario in config["scenarios"]:
        env = gym.make(
            "smarts.env:multi-scenario-v0",
            scenario=scenario,
            action_space="TargetPose",
            img_meters=int(config["img_meters"]),
            img_pixels=int(config["img_pixels"]),
            sumo_headless=True,
            # visdom=True  # 启动浏览器，渲染
            visdom=False  # 启动浏览器，渲染
        )
        # Wrap the environment
        for wrapper in submitted_wrappers():
            env = wrapper(env)
        envs_eval[f"{scenario}"] = env

    # Evaluate model for each scenario
    agent_cnt, agent_complete_cnt = 0, 0
    for index, (env_name, env) in enumerate(envs_eval.items()):
        print(f"\n{index}. Evaluating env {env_name}.\n")
        single_agent_cnt, single_agent_complete_cnt = run(env_name=env_name, env=env, config=config, policy=policy)
        agent_cnt += single_agent_cnt
        agent_complete_cnt += single_agent_complete_cnt

    # Close all environments
    for env in envs_eval.values():
        env.close()
    print(
        f'finish all scenarios, agent_cnt:{agent_cnt}, agent_complete_cnt:{agent_complete_cnt}, comp_rate:{agent_complete_cnt / agent_cnt}')


def run(
        env_name,
        env,
        config: Dict[str, Any],
        policy,
):
    agent_cnt, agent_complete_cnt, instance_cnt, instance_complete_cnt = 0, 0, 0, 0

    for _ in tqdm(range(config["eval_episodes"])):
        seed = config["seed"] - instance_cnt
        print("{}: seed={}".format(env_name, seed))
        env.seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        instance_cnt += 1
        obs = env.reset()
        dones = {"__all__": False}

        agent_ids = set()
        agent_ids.update(set(obs.keys()))

        while not dones["__all__"]:
            actions = policy.act(obs)
            obs, rewards, dones, infos = env.step(actions)
            agent_ids.update(set(obs.keys()))
            for agent_id, raw_obs_and_reset_flag in obs.items():
                raw_obs, reset_flag = raw_obs_and_reset_flag["raw_obs"], raw_obs_and_reset_flag["reset"]
                if raw_obs.events.reached_goal:
                    agent_complete_cnt += 1
                if dones["__all__"]:
                    print("step-{}: \033[1;41m {} \033[0m".format(raw_obs.steps_completed, raw_obs.events))
        agent_cnt += len(agent_ids)
    print(
        f'finish a scenario, instance_num:{instance_cnt} agent_num:{agent_cnt}, agent_complete:{agent_complete_cnt}, comp_rate:{agent_complete_cnt / agent_cnt}')
    return agent_cnt, agent_complete_cnt


if __name__ == "__main__":
    ROAD_WAYPOINTS = False
    config = {
        "eval_episodes": 10,
        "seed": 100,
        "scenarios": [
            "1_to_1lane_left_turn_c",
            "1_to_2lane_left_turn_c",
            "1_to_2lane_left_turn_t",

            "3lane_merge_multi_agent",
            "3lane_merge_single_agent",
            "3lane_cruise_multi_agent",
            "3lane_cruise_single_agent",
            # "3lane_cut_in",
            "3lane_overtake",
            # "3lane_merge_single_agent_0",
            # "3lane_merge_single_agent_1",
            # "3lane_merge_single_agent_2",
            # "3lane_cruise_single_agent_1",
            # "3lane_cruise_single_agent_2",
            # "3lane_cruise_single_agent_n",
        ],
        "img_meters": 50,
        "img_pixels": 256,
    }

    evaluate(config)
