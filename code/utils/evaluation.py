# Evaluation utils

import argparse
import yaml
from yaml import Loader
import numpy as np
from tqdm import tqdm
from waymo_utils.code.utils import misc
# from scenario_mining.utils.parameters.scenario_categories import scenario_catalog


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


def _init_metric_result(metric_name):
    road_users = ["pedestrian", "vehicle", "cyclist", "all"]

    metric = {}

    for user in road_users:
        metric[user] = {
            metric_name: None,
            "count": 0,
            "valid": np.zeros((80,)),
            "agent_id": [],
            "agent_type": [],
            "scenario_id": [],
        }

    return metric


def averaged_minADE(predictions_dataloader):
    result = _init_metric_result("minADE")
    for prediction in tqdm(predictions_dataloader):
        agent_type = misc.type_to_str(prediction["agent_type"])
        prediction_minADE = minADE(prediction)

        if result[agent_type]["minADE"] is None:
            result[agent_type]["minADE"] = prediction_minADE
        else:
            result[agent_type]["minADE"] = result[agent_type]["minADE"] + prediction_minADE
        result[agent_type]["count"] += 1
        result[agent_type]["valid"] += prediction["target/future/valid"].flatten()

        # Add to "all"
        if result["all"]["minADE"] is None:
            result["all"]["minADE"] = prediction_minADE
        else:
            result["all"]["minADE"] = result["all"]["minADE"] + prediction_minADE
        result["all"]["count"] += 1
        result["all"]["valid"] += prediction["target/future/valid"].flatten()

    for agent_type in result.keys():
        result[agent_type]["minADE"] /= result[agent_type]["count"]

    return result


# ToDo: refactor and expand to other metrics and do not hardcode SCs

def _add_error_to_results(results, results_key, prediction):
    agent_type = misc.type_to_str(prediction["agent_type"])
    prediction_minADE = minADE(prediction)

    if results[results_key][agent_type]["minADE"] is None:
        results[results_key][agent_type]["minADE"] = prediction_minADE
    else:
        results[results_key][agent_type]["minADE"] = results[results_key][agent_type]["minADE"] + prediction_minADE
    results[results_key][agent_type]["count"] += 1
    results[results_key][agent_type]["valid"] += prediction["target/future/valid"].flatten()

    # Add to "all"
    if results[results_key]["all"]["minADE"] is None:
        results[results_key]["all"]["minADE"] = prediction_minADE
    else:
        results[results_key]["all"]["minADE"] = results[results_key]["all"]["minADE"] + prediction_minADE
    results[results_key]["all"]["count"] += 1
    results[results_key]["all"]["valid"] += prediction["target/future/valid"].flatten()

def _should_add_to_results(scenario_index, scene_id, agent_id, sc):
    return scene_id in scenario_index and agent_id in scenario_index[scene_id] and sc in scenario_index[scene_id][agent_id]

def averaged_minADE_per_SC(predictions_dataloader, scenario_index):
    results = {
        "overall": _init_metric_result("minADE"),
        "SC1": _init_metric_result("minADE"),
        "SC7": _init_metric_result("minADE"),
        "SC13": _init_metric_result("minADE"),
    }

    #####

    # results_key = "overall"
    for prediction in tqdm(predictions_dataloader):
        scene_id = prediction['scenario_id']
        agent_id = prediction['agent_id']
        if isinstance(scene_id, list):
            scene_id = scene_id[0]
        if isinstance(agent_id, list):
            agent_id = str(agent_id[0])

        _add_error_to_results(results, "overall", prediction)

        for sc in ["SC1", "SC7", "SC13"]:
            if _should_add_to_results(scenario_index, scene_id, agent_id, sc):
                print("Should add to results: scene_id, agent_id, sc = ", scene_id, agent_id, sc)
                _add_error_to_results(results, sc, prediction)

    for k in results.keys():
        for agent_type in results[k].keys():
            if results[k][agent_type]["minADE"] is None:
                continue
            results[k][agent_type]["minADE"] /= results[k][agent_type]["count"]

    return results


def _get_prediction(root, prediction_data):
    result = {}
    for key in prediction_data.keys():
        if root not in key:
            continue
        result[key.replace(root, "")] = prediction_data[key]
    return result


def per_example_minADE(predictions_dataloader):
    result = _init_metric_result("minADE")

    for prediction in tqdm(predictions_dataloader):

        pred = _get_prediction("model/", prediction)
        pred_base = _get_prediction("base/", prediction)
        agent_type = misc.type_to_str(pred["agent_type"])

        prediction_minADE = minADE(pred)
        prediction_base_minADE = minADE(pred_base)

        if result[agent_type]["minADE"] is None:
            result[agent_type]["minADE"] = []

        result[agent_type]["minADE"].append([prediction_minADE, prediction_base_minADE])
        result[agent_type]["agent_type"].append(pred["agent_type"])
        result[agent_type]["scenario_id"].append(pred["scenario_id"])
        result[agent_type]["agent_id"].append(pred["agent_id"])
        result[agent_type]["count"] += 1
        result[agent_type]["valid"] += pred["target/future/valid"].flatten()

        # Add to "all"
        if result["all"]["minADE"] is None:
            result["all"]["minADE"] = []

        result["all"]["minADE"].append([prediction_minADE, prediction_base_minADE])
        result["all"]["agent_type"].append(pred["agent_type"])
        result["all"]["scenario_id"].append(pred["scenario_id"])
        result["all"]["agent_id"].append(pred["agent_id"])
        result["all"]["count"] += 1
        result["all"]["valid"] += pred["target/future/valid"].flatten()

    for agent_type in result.keys():
        result[agent_type]["minADE"] = np.array(result[agent_type]["minADE"])

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
