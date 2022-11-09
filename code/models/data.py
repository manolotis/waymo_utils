import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class PredictionsDataset(Dataset):

    def __init__(self, config, model):
        self._data_path = os.path.join(config["data_config"]["dataset_config"]["data_path"], model["name"])

        self._config = config
        files = os.listdir(self._data_path)
        self._files = [os.path.join(self._data_path, f) for f in files]
        self._files = sorted(self._files)

        if "max_length" in config:
            self._files = self._files[:config["max_length"]]

        assert len(self._files) > 0

    def __len__(self):
        return len(self._files)

    def __getitem__(self, idx):
        np_data = dict(np.load(self._files[idx], allow_pickle=True))

        return np_data

    @staticmethod
    def collate_fn(batch):
        batch_keys = batch[0].keys()
        result_dict = {k: [] for k in batch_keys}

        for sample_num, sample in enumerate(batch):
            for k in batch_keys:
                if not isinstance(sample[k], str) and len(sample[k].shape) == 0:
                    result_dict[k].append(sample[k].item())
                else:
                    result_dict[k].append(sample[k])

        for k, v in result_dict.items():
            if not isinstance(v[0], np.ndarray):
                continue
            # result_dict[k] = torch.Tensor(np.concatenate(v, axis=0))
            result_dict[k] = np.concatenate(v, axis=0)

        result_dict["batch_size"] = len(batch)
        return result_dict


class PredictionsPairDataset(Dataset):

    def __init__(self, config, model):
        self._data_path = os.path.join(config["data_config"]["dataset_config"]["data_path"], model["name"])
        self._data_path_base = os.path.join(config["data_config"]["dataset_config"]["data_path"], model["base"])

        self._config = config
        files = os.listdir(self._data_path)
        files_base = os.listdir(self._data_path_base)
        self._files = [os.path.join(self._data_path, f) for f in files]
        self._files_base = [os.path.join(self._data_path_base, f) for f in files_base]
        self._files = sorted(self._files)
        self._files_base = sorted(self._files_base)

        if "max_length" in config:
            self._files = self._files[:config["max_length"]]

        assert len(self._files) > 0

    def __len__(self):
        return len(self._files)

    def __getitem__(self, idx):
        if self._files[idx].split("/")[-1] != self._files_base[idx].split("/")[-1]:
            print(self._files[idx])
            print(self._files_base[idx])
            raise ValueError("Data record filenames should match")
        np_data = dict(np.load(self._files[idx], allow_pickle=True))
        np_data_base = dict(np.load(self._files_base[idx], allow_pickle=True))

        np_data_new = {}

        for key in np_data.keys():
            if "other" in key:
                continue
            np_data_new[f"model/{key}"] = np_data[key]
            np_data_new[f"base/{key}"] = np_data_base[key]

        return np_data_new

    @staticmethod
    def collate_fn(batch):
        batch_keys = batch[0].keys()
        result_dict = {k: [] for k in batch_keys}

        for sample_num, sample in enumerate(batch):
            for k in batch_keys:
                if not isinstance(sample[k], str) and len(sample[k].shape) == 0:
                    result_dict[k].append(sample[k].item())
                else:
                    result_dict[k].append(sample[k])

        for k, v in result_dict.items():
            if not isinstance(v[0], np.ndarray):
                continue
            # result_dict[k] = torch.Tensor(np.concatenate(v, axis=0))
            result_dict[k] = np.concatenate(v, axis=0)

        result_dict["batch_size"] = len(batch)
        return result_dict


def get_predictions_dataloader(config, model):
    dataset = PredictionsDataset(config, model)
    dataloader = DataLoader(
        dataset, collate_fn=PredictionsDataset.collate_fn, **config["data_config"]["dataloader_config"])
    return dataloader


def get_predictions_pair_dataloader(config, model):
    dataset = PredictionsPairDataset(config, model)
    dataloader = DataLoader(
        dataset, collate_fn=PredictionsPairDataset.collate_fn, **config["data_config"]["dataloader_config"])
    return dataloader
