"""Preparing datasets for use with the ai model."""

import hydra
from iotml_core.ml_tools.config import IoTMLConfig
from iotml_core.utils.data_tools import generate_dataset
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(config: DictConfig) -> None:
    """Generate the dataset described by the project config."""
    config = IoTMLConfig(config)

    # Generate the dataset we need for the ml trainings
    generate_dataset(iotml_config=config)


if __name__ == "__main__":
    main()
