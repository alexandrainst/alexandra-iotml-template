"""Script to run an evaluation of the trained models."""

import logging
import os
from pathlib import Path
from typing import Any

import hydra
import matplotlib.pyplot as plt
import torch
from matplotlib.backends.backend_pdf import PdfPages
from {{cookiecutter.library_name}}.ml_tools.datasets import TimeSnippetDataset
from {{cookiecutter.library_name}}.ml_tools.models import LinearAE, LSTMCell
from {{cookiecutter.library_name}}.utils.config import DatasetConfig, IoTMLConfig, MLTrainingConfig
from {{cookiecutter.library_name}}.utils.evaluation_tools import (
    prediction_accuracy,
    summarize_training_accuracy,
)
from {{cookiecutter.library_name}}.utils.plotting_tools import plot_prediction_accuracy, plot_summaries
from omegaconf import DictConfig

project_path = Path(__file__).parents[2]

logger = logging.getLogger("train_model")
logger.level = logging.INFO


TRAINING_VERSION = "v2"


def evaluate_model(
    training_config: MLTrainingConfig, dataset_config: DatasetConfig
) -> Any:
    """Evaluate the performances of a trained ML model.

    Parameters:
    ---

    training_config : MLTrainingConfig
        information about the training performed

    dataset_config: DatasetConfig
        INformation about the dataset used in training

    """
    dataset_name = dataset_config.name
    training_params = training_config.training_params
    training_name = training_params.name
    training_type = training_params.training_type

    logger.info(
        f"\n\n---- Evaluating {training_name} on dataset {dataset_name} ---\n\n"
    )

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

    training_prefix = os.path.join(
        f"./training_results/{TRAINING_VERSION}/trainings/",
        f"{training_name}_{dataset_name}_dataset",
    )
    dataset_path = os.path.join(
        f"./training_results/{TRAINING_VERSION}/datasets/",
        f"{dataset_name}_dataset/test/",
    )

    state_dict = torch.load(training_prefix + ".pt")
    model_instance.load_state_dict(state_dict)
    model_instance.eval()

    if training_type == "output_predictor":
        dataset = TimeSnippetDataset(
            training_params=training_params, dataset_path=dataset_path
        )
        accuracy_results = prediction_accuracy(model=model_instance, dataset=dataset)
        fig = plot_prediction_accuracy(accuracy_results=accuracy_results)
        accuracy_summary = summarize_training_accuracy(
            accuracy_results=accuracy_results
        )

        return fig, accuracy_summary

    else:
        print("NOTHING PLANNED FOR ANOMALY ENCODER")


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(config: DictConfig) -> None:
    """Main orchestrating function."""
    config = IoTMLConfig(config)

    with PdfPages("multipage_pdf.pdf") as pdf:
        summaries = {}
        for ds_conf in config.ds:
            for train_conf in config.ml_trainings:
                training_name = train_conf.training_params.name
                keyword = f"{training_name}_{ds_conf.name}"
                fig, summary = evaluate_model(
                    training_config=train_conf, dataset_config=ds_conf
                )

                pdf.savefig(fig)
                plt.close()
                summaries[keyword] = summary

        feature_wise_summaries = plot_summaries(summaries)

        for f in feature_wise_summaries:
            pdf.savefig(f)
        plt.close()


if __name__ == "__main__":
    main()
