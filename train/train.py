import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parents[0]))

from data_preprocess import prepare_data_loader, prepare_dataset
from networks import SocialVehiclePredictor


def load_config(path: Path) -> Optional[Dict[str, Any]]:
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config


def train(input_path, output_path):
    # Get config parameters.
    train_config = load_config(Path(__file__).absolute().parents[0] / "config.yaml")

    n_steps = train_config["n_steps"]
    batch_size = train_config["batch_size"]
    learning_rate = train_config["learning_rate"]
    device = train_config["device"]
    model = SocialVehiclePredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    # * Data preprocessing phase
    try:
        dataset = prepare_dataset(input_path)
        dataloader = prepare_data_loader(dataset, batch_size, True)
    except:
        dataloader = None
        
    if dataloader is None:
        # If size of the dataset is zero or any error occurs during data preprocessing, 
        # just stop training and save a model with random parameters.
        torch.save(model.state_dict(), Path(output_path) / "SVP.pt")
        return

    # * Training phase
    try:
        model.train()
        step = 0
        while step < n_steps:

            for (space_wise_data, time_wise_data, valid_lengths) in dataloader:
                step += 1

                space_wise_data = space_wise_data.to(device)
                time_wise_data = time_wise_data.to(device)
                valid_lengths = valid_lengths.to(device)

                loss = model(space_wise_data, time_wise_data, valid_lengths)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        torch.save(model.state_dict(), Path(output_path) / "SVP.pt")
        
    except:
        # If any error occurs during training, just save a model with random parameters.
        model = SocialVehiclePredictor().to(device)
        torch.save(model.state_dict(), Path(output_path) / "SVP.pt")


def main(args: argparse.Namespace):
    input_path = args.input_dir
    output_path = args.output_dir
    train(input_path, output_path)


if __name__ == "__main__":
    program = Path(__file__).stem
    parser = argparse.ArgumentParser(program)
    parser.add_argument(
        "--input_dir",
        help="The path to the directory containing the offline training data",
        type=str,
        default="/SMARTS/competition/offline_dataset/",
    )
    parser.add_argument(
        "--output_dir",
        help="The path to the directory storing the trained model",
        type=str,
        default="/SMARTS/competition/track2/submission/",
    )

    args = parser.parse_args()

    main(args)
