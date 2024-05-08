"""Utility for parsing a project config.

We define here the subsections of the config
and a Parser class to verify whether a config
has all element it is supposed to have

"""

from pydantic import BaseModel, field_validator
from typing import Optional, Dict, Any, List
from omegaconf import DictConfig


class AIModelConfig(BaseModel):
    name: str
    aimodel_params: Dict[Any, Any]

class TrainingParams(BaseModel):
    name: str
    training_type: str
    input_features: Dict[int, str]
    output_features: Dict[int, str]
    learning_rate: float
    batch_size: int
    n_epochs: int
    loss: str

class MLTrainingConfig(BaseModel):
    
    aimodel: AIModelConfig
    training_params: TrainingParams

class DataSubsetConfig(BaseModel):
    name: str
    time_periods: Dict[Any, Any]


class DatasetConfig(BaseModel):
    name: str
    time_window_past: int
    time_window_future: int
    subsets: List[DataSubsetConfig]


###############################################


class IoTMLConfig:

    def __init__(self, config: DictConfig) -> None:
        """Validate and parse the omegaconfig object."""
        self.config = config
        self.validate_config()

    def validate_config(self):
        """implement some checks to make sure the config makes sense."""

        for train_conf in self.ml_trainings:
            try:
                d = MLTrainingConfig(**train_conf)
                m = AIModelConfig(**train_conf["aimodel"])
            except:
                raise Exception("training config badly formatted")
       
        for dataset in self.ds:
            try:
                d = DatasetConfig(**dataset)
            except:
                raise Exception("dataset config badly formatted")

            # validate the time windows defined in dataset
            if dataset["time_window_past"] < dataset["time_window_future"]:
                raise Exception("future time window cannot be larger than the past time window.")

        #TODO: check that model input window size <= time_window_past

    @property
    def ml_trainings(self):
        """return list of ml trainings."""
        return [x.training for x in self.config["ml_trainings"]]


    @property
    def ds(self):
        """return list of dataset configs."""
        return [x["dataset"] for x in self.config["datasets"]]


