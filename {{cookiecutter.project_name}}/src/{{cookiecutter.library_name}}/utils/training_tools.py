"""Utilities relevant for training ML models."""

import glob
import logging
import os

import torch
from {{cookiecutter.library_name}}.utils.config import AIModelConfig, MLTrainingConfig, DatasetConfig
from {{cookiecutter.library_name}}.ml_tools.datasets import retrieve_data_from_sql, produce_snippets

logger = logging.getLogger("utils.training_tools")

TRAINING_VERSION = "v0"


def generate_dataset(dataset_config: DatasetConfig) -> None:
    """Produce a dataset of time series snippet that is stored on file."""
    dataset_name = dataset_config.name
    logger.info(f"\n\n---- Creating dataset {dataset_name} ---\n\n")

    for ds_subset in dataset_config.subsets:
        dataset_path = os.path.join(
            f"./training_results/{TRAINING_VERSION}/datasets/",
            f"{dataset_name}_dataset/{ds_subset.name}",
        )

        logger.info(f"generating {dataset_name}...")

        if os.path.isdir(dataset_path) and (
            len(glob.glob(dataset_path + "/*.pt")) != 0
        ):
            logger.info(f"dataset already exists at {dataset_path}")
            continue

        else:
            logger.info(
                f"Path {dataset_path} is empty. generating time series snippets..."
            )
            os.makedirs(dataset_path, exist_ok=True)

        df = retrieve_data_from_sql(
            start_date=ds_subset.time_periods["start"],
            end_date=ds_subset.time_periods["end"],
        )

        snippets = produce_snippets(
            df=df,
            time_window_past=dataset_config.time_window_past,
            time_window_future = dataset_config.time_window_future,
            )

        for i, snip in enumerate(snippets):
            torch.save(
                snip,
                os.path.join(
                    dataset_path,
                    f"{dataset_name}_{ds_subset.name}_sample_{i:0>6d}.pt",
                ),
            )


