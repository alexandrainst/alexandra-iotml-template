"""Script for training a model on the example IOTML data."""

import logging
import os
from typing import Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from {{cookiecutter.library_name}}.ml_tools.datasets import TimeSnippetDataset
from {{cookiecutter.library_name}}.ml_tools.losses import UnitarityLoss, RecoLoss
from {{cookiecutter.library_name}}.ml_tools.models import LinearAE, LSTMCell
from {{cookiecutter.library_name}}.ml_tools.traintest import {{cookiecutter.class_prefix}}TrainAlgo
from {{cookiecutter.library_name}}.utils.config import IoTMLConfig, MLTrainingConfig, DatasetConfig
from {{cookiecutter.library_name}}.utils.training_tools import generate_dataset
from omegaconf import DictConfig
from sklearn.decomposition import PCA

logger = logging.getLogger("train_model")
logger.level = logging.INFO

TRAINING_VERSION = "v2"


def train_model(
    training_config: MLTrainingConfig, dataset_config: DatasetConfig
) -> Any:
    """Training script for a single model."""
    dataset_name = dataset_config.name
    training_params = training_config.training_params
    training_name = training_params.name

    logger.info(f"\n\n---- Training {training_name} on dataset {dataset_name} ---\n\n")

    if training_params.training_type == "anomaly_encoder":
        model_instance = LinearAE(**training_config.aimodel.aimodel_params)
    elif training_params.training_type == "output_predictor":
        model_instance = LSTMCell(
            time_window_past=dataset_config.time_window_past,
            time_window_future=dataset_config.time_window_future,
            input_features=training_params.input_features,
            output_features=training_params.output_features,
            **training_config.aimodel.aimodel_params,
        )
    else:
        raise Exception("Unrecognized model type.")

    if training_params.loss.lower() == "UnitarityLoss":
        loss_instance = UnitarityLoss()
    else:
        loss_instance = RecoLoss()

    traintest = {{cookiecutter.class_prefix}}TrainAlgo(
        model=model_instance,
        training_config=training_config,
        optimizer=torch.optim.Adam(
            model_instance.parameters(), lr=training_params.learning_rate
        ),
        loss_fn=loss_instance,
        device="cuda",
    )

    # add the train and valid dataset to the algo
    dataset_path = os.path.join(
        f"./training_results/{TRAINING_VERSION}/datasets/",
        f"{dataset_name}_dataset/train/",
    )

    train_data = TimeSnippetDataset(
        training_params=training_params, dataset_path=dataset_path
    )
    traintest.add_dataset("train", train_data, batch_size=training_params.batch_size)
    traintest.train(
        dataset_label="train", n_epochs=training_params.n_epochs, autosave=False
    )

    traintest.record_session(
        output_prefix=os.path.join(
            f"./training_results/{TRAINING_VERSION}/trainings/",
            f"{training_name}_{dataset_name}_dataset",
        )
    )
    return traintest.loss_history


def return_pca_inputs(model, dataset):
    """Compute the PCA projection of the data using the latent space of model."""
    train_samples = []
    for i, d in enumerate(dataset):
        # the model is expecting a batch vector,
        # so we need to add an extra dimension at the beginning
        d = torch.unsqueeze(d, dim=0)
        latent_representation = model.encoder(d.to("cuda"))
        array = latent_representation.cpu().detach().numpy()
        train_samples.append(array.flatten())

    all_samples = np.array(train_samples)
    return all_samples


def define_pca_space(model, train_data):
    """Fit the PCA function to a particular dataset and model."""
    all_samples = return_pca_inputs(model=model, dataset=train_data)

    #
    # Determine the PCA of the training sample in latent space
    #
    pca = PCA(n_components=2)
    pca_model = pca.fit(all_samples)
    return pca_model


######################################################################
# Plotting functions
#
def plot_latent_space_pca(ds_name: str, reduced_values: np.ndarray):
    """Plot2D projection of the PCA components using matplotlib."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_ylim([-2, 6])
    ax.set_xlim([-5, 6])
    # plot the training set first
    ax.scatter(
        reduced_values[:, 0], reduced_values[:, 1], label=f"{ds_name} data", alpha=0.3
    )
    ax.legend()
    plt.show()


######################################################################
@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(config: DictConfig) -> None:
    """Main orchestrating function.

    The script will generate 1 or several train+test datasetsset,
    based on the list of defined datasets in the config
    file. It will then train a list of model trainings
    as they are defined in the same config.
    """
    config = IoTMLConfig(config)

    for ds_conf in config.ds:
        # Generate the datasets we need for the ml trainings
        generate_dataset(dataset_config=ds_conf)

        for train_conf in config.ml_trainings:
            # Run the training
            loss_history = train_model(
                training_config=train_conf, dataset_config=ds_conf
            )

            plt.plot(loss_history, "k")
            plt.title(
                f"Loss over iterations - {train_conf.training_params.training_type}"
            )
            plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Train a model on example data")
    datasets = {  # Training phase: start periods with only a few days of opened park
        "train": {"start": "2024-01-01 00:00:00", "end": "2024-01-01 15:00:00"}
    }
    main()
