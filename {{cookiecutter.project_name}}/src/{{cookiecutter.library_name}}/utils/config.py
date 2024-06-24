"""Utility for parsing a project config.

We define here the subsections of the config
and a Parser class to verify whether a config
has all element it is supposed to have

"""

from collections import OrderedDict
from typing import Any, Dict, List

from omegaconf import DictConfig
from pydantic import BaseModel


class AIModelConfig(BaseModel):
    """Information about the model to use in a training."""

    name: str
    aimodel_params: OrderedDict[Any, Any]


class TrainingParams(BaseModel):
    """Parameters pertaining to the training."""

    name: str
    training_type: str
    input_features: OrderedDict[int, str]
    output_features: OrderedDict[int, str]
    learning_rate: float
    batch_size: int
    n_epochs: int
    loss: str


class MLTrainingConfig(BaseModel):
    """Parameters of an ML training."""

    aimodel: AIModelConfig
    training_params: TrainingParams


class DataSubsetConfig(BaseModel):
    """Parameters used in an SQL query."""

    name: str
    time_periods: OrderedDict[Any, Any]


class DatasetConfig(BaseModel):
    """Parameters of the datasets on which trainings are run."""

    name: str
    time_window_past: int
    time_window_future: int
    subsets: List[DataSubsetConfig]


###############################################


class IoTMLConfig:
    """Class for parsing an IoT+ML training config."""

    def __init__(self, config: DictConfig) -> None:
        """Validate and parse the omegaconfig object."""
        self.config = config
        self.validate_config()

    def validate_config(self):
        """Implement some checks to make sure the config makes sense."""
        for train_conf in self.ml_trainings:
            try:
                MLTrainingConfig(**train_conf)
                AIModelConfig(**train_conf["aimodel"])
            except Exception:
                raise Exception("training config badly formatted")

        for dataset in self.ds:
            try:
                DatasetConfig(**dataset)
            except Exception:
                raise Exception("dataset config badly formatted")

            # validate the time windows defined in dataset
            if dataset["time_window_past"] < dataset["time_window_future"]:
                raise Exception(
                    "future time window cannot be larger than the past time window."
                )

        # TODO: check that model input window size <= time_window_past

    @property
    def ml_trainings(self):
        """Return list of ml trainings."""
        return [x.training for x in self.config["ml_trainings"]]

    @property
    def ds(self):
        """Return list of dataset configs."""
        return [x["dataset"] for x in self.config["datasets"]]
