
import re
import numpy as np
import os
import pickle

from pathlib import Path

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def prepare_space_wise_info(np_obs, i):
    
    neighbor_info = np.delete(np_obs, i, 0)
    rel_pos = neighbor_info[:, :2] - np_obs[i, :2].reshape(1,-1)
    # dist_mask_id = np.where(np.linalg.norm(rel_pos, 2, 1) < 30)[0]
    dist_mask_id = np.linalg.norm(rel_pos, 2, 1).argsort()[:8]
    if dist_mask_id.size == 0:
        return np.zeros((8, 6))
    else:
        one_step_space_wise_info = np.zeros((8, 6))
        
        one_step_space_wise_info[:dist_mask_id.size, :2] = neighbor_info[dist_mask_id, :2] - np_obs[i, :2].reshape(1,-1)
        one_step_space_wise_info[:dist_mask_id.size,  2] = neighbor_info[dist_mask_id,  2] - np_obs[i,  2].reshape(1,-1)
        one_step_space_wise_info[np.where(one_step_space_wise_info > np.pi)] -= 2 * np.pi
        one_step_space_wise_info[np.where(one_step_space_wise_info <-np.pi)] += 2 * np.pi
        one_step_space_wise_info[:dist_mask_id.size, 3:] = neighbor_info[dist_mask_id, 3:]
        return one_step_space_wise_info
    
    
def prepare_time_wise_info(np_obs, i, t, raw_dataset, id_list, predict_horizon):
    
    one_step_time_wise_info = np.zeros((predict_horizon, 6))
    one_step_time_wise_info[0, 2:] = np_obs[i, 2:]
    valid_length = 0
    for h in range(1, predict_horizon):
        future_neighborhood_vehicle_states = raw_dataset[int(10*t+h)/10].neighborhood_vehicle_states
        future_social_vehicle_ids = [social_vehicle.id for social_vehicle in future_neighborhood_vehicle_states]
        if id_list[i] in future_social_vehicle_ids:
            id_index = future_social_vehicle_ids.index(id_list[i])
            time_rel_pos = future_neighborhood_vehicle_states[id_index].position[:2] - np_obs[i, :2]
            time_rel_heading = future_neighborhood_vehicle_states[id_index].heading  - np_obs[i,  2]
            if time_rel_heading > np.pi: time_rel_heading -= 2 * np.pi
            if time_rel_heading <-np.pi: time_rel_heading += 2 * np.pi
            one_step_time_wise_info[h, :3] = (np.concatenate([
                time_rel_pos, [time_rel_heading]
            ]))
            valid_length += 1
    return one_step_time_wise_info, valid_length

def prepare_predictor_dataset_for_one_file(raw_dataset, predict_horizon: int = 15):
    '''
    np_obs: 
        - position_x
        - position_y
        - heading
        - speed
        - bb_length
        - bb_width
    '''
    raw_dataset_len = len(raw_dataset)
    last_index = raw_dataset_len - predict_horizon
    if not last_index > 0:
        return None, None, None
    
    social_vehicle_trajectories = []
    social_vehicle_ids = []
    
    space_wise_info = [] # (K*T, K-1, D)
    time_wise_info = []  # (K*T,   H, D)
    valid_length = []
    
    for cnt, t in enumerate(raw_dataset.keys()):
        if cnt >= last_index + 1: break

        ego_vehicle_state = raw_dataset[t].ego_vehicle_state
        neighborhood_vehicle_states = raw_dataset[t].neighborhood_vehicle_states
        ego_pos = ego_vehicle_state.position[:2]
        
        np_obs = []
        id_list = []
        for social_vehicle in neighborhood_vehicle_states:
            if (
                'off_lane' in social_vehicle.id or 
                np.linalg.norm(ego_pos - social_vehicle.position[:2]) > 50.0
            ):
                continue
            np_obs.append([
                social_vehicle.position[0],
                social_vehicle.position[1],
                social_vehicle.heading - 0,
                social_vehicle.speed,
                social_vehicle.bounding_box.length,
                social_vehicle.bounding_box.width,
            ])
            id_list.append(social_vehicle.id)
        np_obs = np.array(np_obs, dtype=np.float32)
        social_vehicle_trajectories.append(np_obs.copy())
        social_vehicle_ids.append(id_list.copy())
        
        for i in range(np_obs.shape[0]):
            
            # Calculate space_wise_info
            space_wise_info.append(prepare_space_wise_info(np_obs, i))
                
            # Calculate time_wise_info
            one_step_time_wise_info, one_step_valid_length = prepare_time_wise_info(np_obs, i, t, raw_dataset, id_list, predict_horizon)
            time_wise_info.append(one_step_time_wise_info)
            valid_length.append(one_step_valid_length)
    
    return np.array(space_wise_info), np.array(time_wise_info), np.array(valid_length)


class SocialVehiclePredictorDataset(Dataset):
    def __init__(self, space_wise_data, time_wise_data, valid_length):
        self.space_wise_data = space_wise_data.astype(np.float32)
        self.time_wise_data = time_wise_data.astype(np.float32)
        self.valid_length = valid_length.astype(np.int32)
        
    def __len__(self):
        return self.space_wise_data.shape[0]
    
    def __getitem__(self, index):
        space_wise_data = self.space_wise_data[index]
        time_wise_data = self.time_wise_data[index]
        valid_length = self.valid_length[index]
        return space_wise_data, time_wise_data, valid_length
    
    
def prepare_dataset(input_path):
    scenarios = list()
    for scenario_name in os.listdir(input_path):
        scenarios.append(scenario_name)

    space_wise_data = []
    time_wise_data = []
    valid_length_data = []
    
    for scenario in scenarios[0:len(scenarios)]:
    
        vehicle_ids = list()
        scenario_path = Path(input_path) / scenario
        for filename in os.listdir(scenario_path):
            if filename.endswith(".pkl"):
                match = re.search("(.*).pkl", filename)
                if match is None: 
                    print("No matching pickle file found!")
                    continue
                vehicle_id = match.group(1)
                if vehicle_id not in vehicle_ids:
                    vehicle_ids.append(vehicle_id)

        for id in vehicle_ids[0:len(vehicle_ids)]:
            print(f"Adding data for vehicle id {id} in scenario {scenario}.")

            with open(scenario_path / f"{id}.pkl", "rb") as f:
                vehicle_data = pickle.load(f)
            
            space_wise_info, time_wise_info, valid_length = \
                prepare_predictor_dataset_for_one_file(vehicle_data, 15)
                
            if space_wise_info is None: continue
            
            space_wise_data.append(space_wise_info)
            time_wise_data.append(time_wise_info)
            valid_length_data.append(valid_length)
    
    space_wise_data = np.concatenate(space_wise_data)
    time_wise_data = np.concatenate(time_wise_data)
    valid_length_data = np.concatenate(valid_length_data)
    
    return SocialVehiclePredictorDataset(space_wise_data, time_wise_data, valid_length_data)
    
def prepare_data_loader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
