"""Script for training a model on the example IOTML data."""

import logging
from typing import Any

import hydra
import matplotlib.pyplot as plt
import torch
from iotml_core.ml_tools.config import IoTMLConfig
from iotml_core.ml_tools.datasets import TimeSnippetDataset
from iotml_core.ml_tools.train import TrainAlgo
from iotml_core.utils.data_tools import generate_dataset
from omegaconf import DictConfig

logger = logging.getLogger("train_model")
logger.level = logging.INFO

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train_model(iotml_config: IoTMLConfig) -> Any:
    """Train a single model described by ``iotml_config``."""
    training_config = iotml_config.ml_training
    model_config = iotml_config.aimodel["architecture"]

    logger.info(
        f"\n\n---- Training {iotml_config.training_name} "
        f"on dataset {iotml_config.dataset_name} ---\n\n"
    )

    # Instantiate the model, loss and optimizer straight from the hydra config
    # (each carries a `_target_`; see config/aimodel and config/ml_training).
    model_instance = hydra.utils.instantiate(model_config)
    loss_instance = hydra.utils.instantiate(training_config["loss"])
    opti_instance = hydra.utils.instantiate(
        training_config["optimizer"], params=model_instance.parameters()
    )
    logger.info(f"model instance:\n----\n{model_instance}")
    logger.info(f"loss function:\n----\n{loss_instance}")
    logger.info(f"optimizer:\n----\n{opti_instance}")

    traintest = TrainAlgo(
        model=model_instance,
        iotml_config=iotml_config,
        optimizer=opti_instance,
        loss_fn=loss_instance,
        device=DEVICE,
    )

    n_epochs = 1 if iotml_config.debug else training_config["n_epochs"]
    train_data = TimeSnippetDataset(iotml_config=iotml_config)

    traintest.add_dataset(
        iotml_config.dataset_name,
        train_data,
        batch_size=training_config["batch_size"],
    )
    traintest.train(
        dataset_label=iotml_config.dataset_name, n_epochs=n_epochs, autosave=False
    )
    traintest.record_session()
    return traintest.loss_history


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(config: DictConfig) -> None:
    """Generate the configured dataset, then train a model on it."""
    config = IoTMLConfig(config)

    # Generate the dataset needed for the training (no-op if it already exists)
    generate_dataset(iotml_config=config)

    # Run the training
    loss_history = train_model(iotml_config=config)

    training_type = config.ml_training["training_type"]
    plt.plot(loss_history, "k")
    plt.title(f"Loss over iterations - {training_type}")
    plt.show()


if __name__ == "__main__":
    main()
