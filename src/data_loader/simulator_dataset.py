from torch.utils.data import Dataset
import logging
import os, json
import numpy as np
import torch
import re as regex


class SimulatorDataset(Dataset):
    def __init__(
        self,
        name_dataset,
        data_file_path,
        max_train_size=-1,
        device="cuda",
        lazy_loading=False,
        testing_set=False,
    ):
        # Load the data from the file
        logging.info(f"Loading data from {data_file_path}")

        self.name_dataset = name_dataset
        self.lazy_loading = lazy_loading
        self.device = device
        self.x_params = None
        self.x_sims = None
        self.max_train_size = max_train_size
        self.testing_set = testing_set

        conf_file_name = os.path.basename(data_file_path).replace(
            "data", "args"
        )
        # remove format extension
        conf_file_name = regex.sub(r"\..*", "", conf_file_name)

        conf_file_path = os.path.join(
            os.path.dirname(data_file_path), conf_file_name + ".json"
        )
        if os.path.exists(conf_file_path):
            self.conf_data = json.load(open(conf_file_path, "r"))
            self.conf_all_args = self.conf_data["all_args"]
            self.noisy_samples = self.conf_all_args["n_stoch_samples"]
            if self.noisy_samples == 0:
                self.noisy_samples = 1
        else:
            self.noisy_samples = 1

        if self.lazy_loading:
            init_load_device = "cpu"
        else:
            init_load_device = device

        if data_file_path.endswith(".npz"):
            data_dict = np.load(data_file_path)
        else:
            data_dict = torch.load(
                data_file_path, map_location=init_load_device
            )

        self.x_params = None
        if data_dict["params"] is not None:
            self.x_params = data_dict["params"]

        if name_dataset == "pendulum":
            
            self.x_sims = data_dict["sims"]
        else:
            self.x_sims = data_dict["x"]

        self.init_conds = data_dict["init_conds"]

        if testing_set:
            if self.x_params is not None:
                self.x_params = self.x_params.reshape(
                    (-1, self.noisy_samples) + tuple(self.x_params.shape[1:])
                )
                self.x_sims = self.x_sims.reshape(
                    (-1, self.noisy_samples) + tuple(self.x_sims.shape[1:])
                )
                self.init_conds = self.init_conds.reshape(
                    (-1, self.noisy_samples) + tuple(self.init_conds.shape[1:])
                )

        if self.max_train_size > 0:
            if self.x_params is not None:
                self.x_params = self.x_params[: self.max_train_size]
            self.x_sims = self.x_sims[: self.max_train_size]
            self.init_conds = self.init_conds[: self.max_train_size]

            if self.x_params is not None:
                assert len(self.x_params) == len(
                    self.x_sims
                ), "The number of parameters and simulations must be the same, if both specified."

    def __len__(self):
        return self.x_sims.shape[0]

    def __getitem__(self, index):
        if self.x_params is not None:
            return {
                "init_conds": self.init_conds[index].to(self.device),
                "params": self.x_params[index].to(self.device),
                "x": self.x_sims[index].to(self.device),
            }
        else:
            return {
                "init_conds": self.init_conds[index].to(self.device),
                "x": self.x_sims[index].to(self.device),
                "params": torch.nan,
            }
