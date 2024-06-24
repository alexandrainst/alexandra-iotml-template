"""Definition of the project's pytorch datasets.

File defining the pytorch datasets used for training in this case.
"""

import logging
import os
from collections import OrderedDict
from typing import Any, Dict, List
import numpy as np
import torch
from torch.utils.data import Dataset
from sqlalchemy.sql import text

from {{cookiecutter.library_name}}.utils.sql import load_session
from {{cookiecutter.library_name}}.utils.config import TrainingParams

logger = logging.getLogger("ml_tools.datasets")


class TimeSnippetDataset(Dataset):
    """Custom wrapper on torch's Dataset class."""

    def __init__(
        self,
        static_values: dict[str, Any],
        training_params: TrainingParams,
        dataset_path: str,
    ) -> None:
        """Initialize the dataset class.

        Parameter
        -------

        static_values: dict
            Statistical properties related to an entire dataset.
            These are passed into the normalize_data function

        training_params : TrainingParams
            A dict containing training-specific information
            like the model type and features to be used.

        dataset_path : str
            Path to a directory containing .pt files

        """
        self.dataset_path = dataset_path
        self.static_values = static_values
        self.training_params = training_params

        all_files = []
        for element in os.listdir(dataset_path):
            if "dataset_statistics" in element:
                continue
            all_files.append(element)
        self.snippets = sorted(all_files)

    def __len__(self):
        """Return how many snippet file that exist."""
        return len(self.snippets)

    def __getitem__(self, idx):
        """Define the way we load data into the DataLoader.

        This example simply loads pre-saved torch tensor
        files
        """
        filename = os.path.join(self.dataset_path, self.snippets[idx])

        data = torch.load(filename)

        # Process current datapoint information
        current_pt = data["current_pt"]
        input_data = data["input_data"]

        current_pt = de_nonify(current_pt)
        normalized_input = normalize_data(input_data)

        past = {}
        for k in self.training_params.input_features.values():
            v = torch.Tensor(normalized_input[k])
            past[k] = v

        if self.training_params.training_type == "output_predictor":
            truth_data = data["truth_data"]
            normalized_truth = normalize_data(truth_data)
            future = {}
            for k in self.training_params.output_features.values():
                v = torch.Tensor(normalized_truth[k])
                future[k] = v
        elif self.training_params.training_type == "anomaly_encoder":
            future = 0.0
        else:
            raise Exception("Training type unknown.")

        out = {
            "current_pt": current_pt,
            "input": past,
            "truth": future,
        }
        return out


###############################################################
# Data extraction and preprocessing
def retrieve_data_from_sql(
    sql_table: str, variables: List[str], start_date: str, end_date: str
    ) -> Any:
    """Function that extracts raw data from postgres.

    In this project we produce datasets for a specific
    turbine.

    Parameters
    ---

    sql_table: str
        the name of the table we gather data from

    variables: List[str]
        a list of variables to extract from the table

    start_date: str
        The start of the data period, we are interested in,
        in format "yyyy-mm-dd HH:MM:SS+ZZ"

    end_date: str
        The end of the data period, we are interested in,
        in format "yyyy-mm-dd HH:MM:SS+ZZ"

    """
    session = load_session()

    statement = "SELECT "
    for var in variables:
        statement += f"{var}, "
    statement = statement[:-2]

    statement += f"""
            FROM
                {sql_table}
            WHERE
                time BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY 1;
        """
    statement = text(statement)

    returned = session.execute(text(statement))
    data: OrderedDict[str, Any] = OrderedDict()
    data["month"] = []
    data["day"] = []
    data["weekday"] = []
    data["hour"] = []
    data["minute"] = []

    data["month_s"] = []
    data["weekday_s"] = []
    data["hour_s"] = []
    data["minute_s"] = []

    data["month_c"] = []
    data["weekday_c"] = []
    data["hour_c"] = []
    data["minute_c"] = []
    for i, x in enumerate(returned):
        row = dict(x._mapping)
        for k, v in row.items():
            if i == 0:
                data[k] = []
            data[k].append(v)

            if k == "time":
                data["month"].append(v.month)
                data["day"].append(v.day)
                data["weekday"].append(v.weekday())
                data["hour"].append(v.hour)
                data["minute"].append(v.minute)

                data["month_s"].append(np.sin(2 * np.pi * v.month / 12))
                data["weekday_s"].append(np.sin(2 * np.pi * v.weekday() / 7.0))
                data["hour_s"].append(np.sin(2 * np.pi * v.hour / 24.0))
                data["minute_s"].append(np.sin(2 * np.pi * v.minute / 60.0))

                data["month_c"].append(np.cos(2 * np.pi * v.month / 12))
                data["weekday_c"].append(np.cos(2 * np.pi * v.weekday() / 7.0))
                data["hour_c"].append(np.cos(2 * np.pi * v.hour / 24.0))
                data["minute_c"].append(np.cos(2 * np.pi * v.minute / 60.0))

    return data


def gather_dataset_statistics(df: dict[Any, Any]) -> dict[Any, Any]:
    """Compute statistical properties of individual dataset.

    This function looks at all dataset variables, and saves some of their properties in
    static_values, which can later be used alongside the normalizing function.
    """
    static_values: dict[str, Any] = {}

    for k, v in df.items():
        if k == "time":
            continue
        v = [e if e is not None else 0.0 for e in v]
        v = np.array(v, dtype=float)
        static_values[k] = {
            "mean": np.nanmean(v),
            "std": np.nanstd(v, ddof=1),
            "min": np.nanmin(v),
            "max": np.nanmax(v),
            "median": np.nanmedian(v),
        }

    return static_values


def merge_datasets(df_list: List) -> dict[Any, Any]:
    """Merge several datasets into one big one."""
    keys = df_list[0].keys()
    all_data: dict[Any, Any] = {}
    for k in keys:
        all_data[k] = []

    for df in df_list:
        for k in keys:
            all_data[k] += df[k]

    return all_data


def de_nonify(pydict: Dict[Any, Any]) -> Dict[Any, Any]:
    """Remove Nones from python dictionary."""
    newdict: Dict[Any, Any] = {}

    for k, v in pydict.items():
        newdict[k] = [x if x is not None else -1.0 for x in v]

    return newdict


def produce_snippets(
    df: dict,
    time_window_past: int,
    time_window_future: int,
    include_keys: list | None = None,
    ) -> Any:
    """Take a historical dataset and cut it into snippets of specified dimensions.

    A snippet consists of three elements:

    - the current data point for which the snippet is made
    - a list of the time_window_past previous data points
    - a list of the time_window_future next data points

    The snippet is returned as a Python dict

    Parameters
    ----
    df : dict
        dictionary of the full time series data

    time_window_past: int
        Number of time steps to keep prior to current point

    time_window_future: int
        Number of time steps to keep after to current point

    include_keys: list
        A list of keys from "df" we want to include in the snippet. If
        set to None default to all keys present in "df".

    Returns:
    ---
    snippets : list
        A list of produced snippets of data
    """
    n_data_points = len(df[list(df.keys())[0]])

    if include_keys is None:
        keys_included = list(df.keys())
    else:
        keys_included = include_keys

    i = 0
    snippets: List[Any] = []
    while i < n_data_points:
        if i < time_window_past:
            logger.debug(f"{i}: lower than time window {time_window_past}")
            i += 1
            continue

        if i > (n_data_points - time_window_future):
            logger.debug(
                f"{i}: higher than n_datapoints - time window \
                {n_data_points-time_window_future}"
            )
            break

        current_pt: dict[Any, Any] = {}
        input_data: dict[Any, Any] = {}
        truth_data: dict[Any, Any] = {}

        for k in keys_included:
            v = df[k]
            current_pt[k] = [v[i]]
            input_data[k] = v[(i - time_window_past) : i]
            truth_data[k] = v[(i + 1) : (i + 1 + time_window_future)]
            
        # Stop when the last snippet is empty
        if len(input_data[list(input_data.keys())[0]]) == 0:
            break

        snippet: dict[Any, Any] = {
            "input_data": input_data,
            "truth_data": truth_data,
            "current_pt": current_pt,
        }
        snippets.append(snippet)
        i += 1

    return snippets


def normalize_data(df: dict[Any, Any], static_values: dict[Any, Any]) -> dict[Any, Any]:
    """Normalize data, if needed with static values.

    Use this function to apply simple
    transformation to your data like
    normalizing values between -1 and 1

    Parameters:
    ---

    df : dict
        Raw dataset extracted from retrieve_from_sql

    static_values: dict
        dictionary of precalculated normalization values,
        usually based on the training dataset
    """
    for k, v in df.items():
        if k == "time":
            continue

        # Clear None's and convert to NaN
        v = [e if e is not None else 0.0 for e in v]
        v = np.array(v, dtype=float)

        if k == "some_large_variable":
            v = np.log10(v + 1)

        mu = static_values[k]["mean"]
        sigma = static_values[k]["std"]
        df[k] = (v - mu) / sigma

    return df


#########################
