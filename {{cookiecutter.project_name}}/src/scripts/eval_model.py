"""Script to run an evaluation of the trained models."""

import logging
from typing import Any

import hydra
import matplotlib.pyplot as plt
import torch
from iotml_core.ml_tools.config import IoTMLConfig
from iotml_core.ml_tools.datasets import TimeSnippetDataset
from matplotlib.backends.backend_pdf import PdfPages
from omegaconf import DictConfig

from {{cookiecutter.project_name}}.utils.evaluation_tools import (
    prediction_accuracy,
    summarize_training_accuracy,
)
from {{cookiecutter.project_name}}.utils.plotting_tools import (
    plot_prediction_accuracy,
    plot_summaries,
)

logger = logging.getLogger("eval_model")
logger.level = logging.INFO


def evaluate_model(iotml_config: IoTMLConfig) -> Any:
    """Evaluate the performances of a trained ML model."""
    training_type = iotml_config.training_type
    model_config = iotml_config.aimodel["architecture"]

    logger.info(
        f"\n\n---- Evaluating {iotml_config.training_name} "
        f"on dataset {iotml_config.dataset_name} ---\n\n"
    )

    # Rebuild the model architecture from the config and load its weights
    model_instance = hydra.utils.instantiate(model_config)
    model_path = iotml_config.retrieve_model_name_and_path() + ".pt"
    model_instance.load_state_dict(torch.load(model_path))
    model_instance.eval()

    if training_type == "output_predictor":
        dataset = TimeSnippetDataset(iotml_config=iotml_config)
        accuracy_results = prediction_accuracy(model=model_instance, dataset=dataset)
        fig = plot_prediction_accuracy(accuracy_results=accuracy_results)
        accuracy_summary = summarize_training_accuracy(
            accuracy_results=accuracy_results
        )
        return fig, accuracy_summary

    logger.info("No evaluation routine is defined for the anomaly_encoder yet.")
    return None, None


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(config: DictConfig) -> None:
    """Evaluate the configured training and write the plots to a PDF."""
    config = IoTMLConfig(config)

    with PdfPages("multipage_pdf.pdf") as pdf:
        keyword = f"{config.training_name}_{config.dataset_name}"
        fig, summary = evaluate_model(iotml_config=config)

        if fig is None:
            return

        pdf.savefig(fig)
        plt.close()

        for f in plot_summaries({keyword: summary}):
            pdf.savefig(f)
        plt.close()


if __name__ == "__main__":
    main()
