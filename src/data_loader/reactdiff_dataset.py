from torch.utils.data import Dataset
import logging
import os, json
import pandas as pd
import torch


class ReactionDiffusionDataset(Dataset):
    def __init__(
        self,
        name_dataset,
        data_file_path,
        device="cuda",
        testing_set=False,
    ):
        # Load the data from the file
        logging.info(f"Loading data from {data_file_path}")
        self.name_dataset = name_dataset
        self.device = device
        self.x_params = None
        self.x_sims = None
        self.testing_set = testing_set
        self.data_dir = data_file_path

        # Load configuration file
        filename = os.path.basename(data_file_path)
        conf_file_path = os.path.join(data_file_path, f"args_{filename}.json")
        self.conf_data = json.load(open(conf_file_path, "r"))["all_args"]
        self.batch_size_data = self.conf_data["batch_size"]
        self.noisy_samples = self.conf_data["n_stoch_samples"]
        if self.noisy_samples == 0:
            self.noisy_samples = 1

        hash_exp_name = self.conf_data["filename"]

        # Load data annotations
        annotation_path = os.path.join(
            data_file_path, f"annotations_{hash_exp_name}.csv"
        )
        self.data_annotations = pd.read_csv(annotation_path)
        self.max_batches = self.data_annotations.idx.max()
        # make pandas index be the idx column
        self.data_annotations.set_index("idx", inplace=True)

    def __len__(self):
        return self.max_batches + 1

    def __getitem__(self, index):

        # Retrieve row from index
        row = self.data_annotations.loc[index]
        data_file_path = os.path.join(self.data_dir, row["filename"])
        data_dict = torch.load(data_file_path, map_location=self.device)
        self.x_params = data_dict["params"]
        self.x_sims = data_dict["x"]
        self.init_conds = data_dict["init_conds"]

        if self.testing_set:
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

        if self.x_params is not None:
            return {
                "init_conds": self.init_conds,
                "params": self.x_params,
                "x": self.x_sims,
            }
        else:
            return {
                "init_conds": self.init_conds,
                "x": self.x_sims,
                "params": torch.nan,
            }
