import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class PredictionsDataset(Dataset):

    def __init__(self, config):
        self._data_path = os.path.join(config["data_config"]["dataset_config"]["data_path"], config["model"]["name"])

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


def get_predictions_dataloader(config):
    dataset = PredictionsDataset(config)
    dataloader = DataLoader(
        dataset, collate_fn=PredictionsDataset.collate_fn, **config["data_config"]["dataloader_config"])
    return dataloader
