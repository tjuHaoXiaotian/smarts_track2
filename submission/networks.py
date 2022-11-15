import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parents[0]))

from data_preprocess import prepare_space_wise_info


class MLP(nn.Module):
    def __init__(
            self, input_dim: int, output_dim: int, layers_size: List[int] = [256] * 2,
            activation: nn.Module = nn.ReLU, output_activation: Optional[nn.Module] = None
    ):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_dim, layers_size[0]), activation()]
        for i in range(len(layers_size) - 1):
            layers += [
                nn.Linear(layers_size[i], layers_size[i + 1]),
                activation()
            ]
        layers.append(nn.Linear(layers_size[-1], output_dim))

        if output_activation is not None:
            layers.append(output_activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SocialVehiclePredictor(nn.Module):
    def __init__(
            self,
            d_model=128,
            predict_horizon=15,
            device="cpu"
    ):
        super().__init__()
        self.predict_horizon = predict_horizon
        self.device = device

        # Transformer Encoder
        self.neighbor_encode_layer = MLP(6, d_model, [32, 64])
        self.current_encode_layer = MLP(6, d_model, [32, 64])
        self.future_encode_layer = MLP(3, d_model, [32, 64])

        # self.neighbor_encode_layer = nn.Linear(6, d_model)
        # self.current_encode_layer  = nn.Linear(3, d_model)
        # self.future_encode_layer   = nn.Linear(3, d_model)

        self.positional_encode = nn.parameter.Parameter(
            torch.randn((1, predict_horizon - 1, d_model)), requires_grad=True
        )

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=512, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            transformer_layer, num_layers=2
        )

        self.future_decoder_layer = MLP(d_model, 3, [64, 32])
        # self.future_decoder_layer = nn.Linear(d_model, 3)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, space_wise_data, time_wise_data, valid_lengths):
        neighbor_encode = self.neighbor_encode_layer(space_wise_data)
        current_encode = (
                self.current_encode_layer(time_wise_data[:, 0, :]) +
                self.positional_encode[:, 0, :]
        ).unsqueeze(1)
        future_encode = self.future_encode_layer(time_wise_data[:, 1:-1, :3]) + \
                        self.positional_encode[:, 1:, :]

        transformer_input = torch.cat([
            neighbor_encode,
            current_encode,
            future_encode,
        ], 1)
        src_mask = torch.triu(torch.ones(22, 22), diagonal=1).bool().to(space_wise_data.device)
        src_mask[:, :9] = 0

        loss_mask = (torch.arange(self.predict_horizon - 1).unsqueeze(0).repeat(valid_lengths.shape[0], 1).to(
            space_wise_data.device) < \
                     valid_lengths.reshape(-1, 1)).float().unsqueeze(-1)
        padding_mask = torch.cat([
            torch.zeros((loss_mask.shape[0], 22 - self.predict_horizon + 1), device=loss_mask.device),
            loss_mask.squeeze() - 1,
        ], -1)

        transformer_output = self.transformer(
            transformer_input,
            src_mask,
            padding_mask,
        )[:, -self.predict_horizon + 1:, :]
        state_prediction = self.future_decoder_layer(transformer_output)

        loss = ((state_prediction - time_wise_data[:, 1:, :3]) ** 2)

        loss = (loss * loss_mask).mean()
        return loss

    @torch.no_grad()
    def predict(self, space_wise_data, np_obs):
        neighbor_encode = self.neighbor_encode_layer(space_wise_data)

        time_wise_data = torch.zeros((space_wise_data.shape[0], self.predict_horizon, 6), \
                                     device=space_wise_data.device)
        time_wise_data[:, 0, 2:] = np_obs[:, 2:]

        current_encode = (
                self.current_encode_layer(time_wise_data[:, 0, :]) +
                self.positional_encode[:, 0, :]
        ).unsqueeze(1)

        fixed_transformer_input = torch.cat([
            neighbor_encode,
            current_encode
        ], 1)

        for i in range(self.predict_horizon - 1):
            if i == 0:
                transformer_output = self.transformer(fixed_transformer_input)[:, -1, :]
            else:
                future_encode = (
                        self.future_encode_layer(time_wise_data[:, 1:i + 1, :3]) +
                        self.positional_encode[:, 1:i + 1, :]
                )
                transformer_input = torch.cat([
                    fixed_transformer_input,
                    future_encode
                ], 1)
                transformer_output = self.transformer(transformer_input)[:, -1, :]
            time_wise_data[:, i + 1, :3] = self.future_decoder_layer(transformer_output)
        return time_wise_data[:, :, :3]

    @torch.no_grad()
    def predict_with_raw_obs(self, raw_obs):
        '''Predict the states of neighbors for the next 15 steps.
        
        args:
            raw_obs: The raw observation of a single agent 
                i.e. raw_env.reset()["Agent_0"]
            device: 
            
            
        returns:
            predicted_state: The states of the present and the predicted next 14 steps.
                format: [social_vehicles, 15 states, position&heading]
        '''
        neighborhood_vehicle_states = raw_obs.neighborhood_vehicle_states
        ego_vehicle_state = raw_obs.ego_vehicle_state
        ego_pos = ego_vehicle_state.position[:2]
        space_wise_info, np_obs, vehicle_ids = [], [], []
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
            vehicle_ids.append(social_vehicle.id)

        if len(np_obs) == 0:
            return None
        np_obs = np.array(np_obs, dtype=np.float32)
        for i in range(np_obs.shape[0]):
            # Calculate space_wise_info
            space_wise_info.append(prepare_space_wise_info(np_obs, i))
        space_wise_data = np.array(space_wise_info)
        init_state = np_obs[:, :3]
        time_wise_data = self.predict(
            torch.from_numpy(space_wise_data).to(self.device).float(),
            torch.from_numpy(np_obs).to(self.device).float(),
        )
        time_wise_data[:, 0, -1] = 0.0
        predicted_state = init_state.reshape(-1, 1, 3) + time_wise_data.cpu().numpy()
        predicted_state[:, :, -1] += np.pi / 2
        predicted_state[:, :, -1][np.where(predicted_state[:, :, -1] > np.pi)] -= 2 * np.pi
        predicted_state[:, :, -1][np.where(predicted_state[:, :, -1] < -np.pi)] += 2 * np.pi

        return dict(zip(vehicle_ids, predicted_state))


def prepare_SV_predictor_model(model_file_name="SVP_15.pt", device="cpu"):
    model = SocialVehiclePredictor(device=device).to(device)
    model.load_state_dict(torch.load(
        Path(__file__).parents[1] / model_file_name, map_location=torch.device(device)
    ))
    return model
