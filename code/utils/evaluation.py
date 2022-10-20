# Evaluation utils

import argparse
import yaml
from yaml import Loader
import numpy as np
from tqdm import tqdm
from waymo_utils.code.utils import misc


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=False, help="Model to evaluate")
    parser.add_argument("--n-jobs", type=int, required=False, help="Dataloader number of workers")
    parser.add_argument("--data-path", type=str, required=False, help="Path to predictions folder")
    parser.add_argument("--out-path", type=str, required=False, help="Path to save evaluation")
    parser.add_argument("--config", type=str, required=True, help="Config file path")

    args = parser.parse_args()
    return args


def get_config(args):
    path = args.config
    with open(path, 'r') as stream:
        config = yaml.load(stream, Loader)

    # If provided in command line, override defaults

    if args.model_name is not None:
        config["model"]["name"] = args.model_name

    if args.n_jobs is not None:
        config["data_config"]["dataloader_config"]["num_workers"] = args.n_jobs

    if args.data_path is not None:
        config["data_config"]["dataset_config"]["data_path"] = args.data_path

    if args.out_path is not None:
        config["output_config"]["out_path"] = args.out_path

    return config


def minADE(prediction):
    # calculate minADE for 1 target prediction
    n_predictions, _, _ = prediction["coordinates"].shape
    return minADEtopK(prediction, K=n_predictions)


def minADEtopK(prediction, K=1):
    # calculate minADE for 1 target prediction considering only the topK most likely trajectories
    coordinates = prediction["coordinates"]
    assert len(coordinates.shape) == 3  # (n_predictions, num_timesteps, num_features)
    n_predictions, timesteps, _ = coordinates.shape
    if n_predictions != K:
        raise NotImplementedError("Still didn't implement sorting by probability and choosing topK predictions")

    probabilities = prediction["probabilities"]
    gt = prediction["target/future/xy"]
    gt_valid = prediction["target/future/valid"]

    return _compute_minADEs(coordinates, probabilities, gt, gt_valid)


def _init_metric_result(name):
    road_users = ["pedestrian", "vehicle", "cyclist"]

    metric = {}

    for user in road_users:
        metric[user] = {
            name: None,
            "count": 0
        }

    return metric


def averaged_minADE(predictions_dataloader):
    result = _init_metric_result("minADE")
    for prediction in tqdm(predictions_dataloader):
        agent_type = misc.type_to_str(prediction["agent_type"])

        if result[agent_type]["minADE"] is None:
            result[agent_type]["minADE"] = minADE(prediction)
        else:
            result[agent_type]["minADE"] = result[agent_type]["minADE"] + minADE(prediction)
        result[agent_type]["count"] += 1

    for agent_type in result.keys():
        result[agent_type]["minADE"] /= result[agent_type]["count"]

    return result


def _compute_minADEs(coordinates, probabilities, gt, gt_valid):
    # ToDo: Make more efficient
    n_predictions, timesteps, _ = coordinates.shape
    errors = np.linalg.norm(coordinates - gt, axis=2, keepdims=True)
    errors = errors * gt_valid  # make invalid timesteps 0

    ADEs = np.zeros_like(errors)
    minADEs = np.zeros((timesteps,))

    for t in range(timesteps):
        ADEs[:, t] = errors[:, :t + 1].mean(axis=1)
        minADEs[t] = ADEs[:, t].min()

    return minADEs.squeeze()


def minFDE(prediction):
    # calculate minFDE for 1 target prediction
    n_predictions, _, _ = prediction["coordinates"].shape
    return minFDEtopK(prediction, K=n_predictions)


def minFDEtopK(K=1):
    # calculate minFDE for 1 target prediction considering only the topK most likely trajectories
    raise NotImplementedError
