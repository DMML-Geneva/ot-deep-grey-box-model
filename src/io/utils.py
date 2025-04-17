import json
import os
import yaml
import re

import numpy as np
import torch

# from cloudpathlib import CloudPath
from hydra.core.hydra_config import HydraConfig


def get_hydra_rel_dir():
    if HydraConfig.initialized():
        hydra = HydraConfig.get()
    if HydraConfig.initialized():
        return hydra.runtime.output_dir.replace(hydra.runtime.cwd, "")
    return None


def get_hydra_output_dir():
    if HydraConfig.initialized():
        hydra = HydraConfig.get()
        return hydra.runtime.output_dir
    return None


def dict_to_argparse_input(dictionary):
    arg_list = []
    for key, value in dictionary.items():
        arg_list.append(f"--{key}")
        # check if the value is a list
        if isinstance(value, list):
            arg_list.extend([str(v) for v in value])
        else:
            arg_list.append(str(value))

    return arg_list


# create a function that check if a directory exists and if not create it
def create_dir_if_ne(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def load_yaml(path_file):
    with open(path_file, "r") as file:
        return yaml.safe_load(file)
    return None


def _get_filepath(filename, dirpath, multirun):
    if dirpath is None:  # Get the latest results by default
        dirpath = os.path.dirname(os.path.dirname(ant.__file__))
        if multirun:
            dirpath = os.path.join(dirpath, "multirun")
            latest_date = [
                d
                for d in sorted(os.listdir(dirpath))
                if os.path.isdir(os.path.join(dirpath, d))
            ][-1]
            dirpath = os.path.join(dirpath, latest_date)
            latest_time = [
                d
                for d in sorted(os.listdir(dirpath))
                if os.path.isdir(os.path.join(dirpath, d))
            ][-1]
            dirpath = os.path.join(dirpath, latest_time)
        else:
            dirpath = os.path.join(dirpath, "outputs")
            latest_date = [
                d
                for d in sorted(os.listdir(dirpath))
                if not os.path.isfile(os.path.join(dirpath, d))
            ][-1]
            dirpath = os.path.join(dirpath, latest_date)
            latest_time = [
                d
                for d in sorted(os.listdir(dirpath))
                if not os.path.isfile(os.path.join(dirpath, d))
            ][-1]
            dirpath = os.path.join(dirpath, latest_time)
    if multirun:
        subdirs = [
            d
            for d in sorted(os.listdir(dirpath))
            if os.path.isdir(os.path.join(dirpath, d)) and d != ".submitit"
        ]
        return [os.path.join(dirpath, d, filename) for d in subdirs]
    else:
        return os.path.join(dirpath, filename)


def find_files(dir, regex_str):
    f_res = []
    for root, dirs, files in os.walk(dir):
        for f in files:
            if bool(re.search(regex_str, f)):
                f_res.append(os.path.join(root, f))
    return f_res


class ResultEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if torch.is_tensor(obj):
            return obj.tolist()
        if isinstance(obj, Exception):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


# def s3_upload_from(cloud_upload_path, local_path, s3_client=None):
#    # if the path contains `//`` after `s3://` added due to hydra remove them; sometimes it can cause some issues when uploading the files
#    path = cloud_upload_path.replace("s3://", "").replace("//", "/")
#    path = "s3://" + path
#    cloud_path = CloudPath(path, client=s3_client)
#    return cloud_path.upload_from(local_path, force_overwrite_to_cloud=True)
