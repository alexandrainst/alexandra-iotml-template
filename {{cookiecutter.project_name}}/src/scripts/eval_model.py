"""Script to run an evaluation of the trained models."""
import glob
import logging
import os
from pathlib import Path
from typing import Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from {{cookiecutter.library_name}}.ml_tools.datasets import {{cookiecutter.class_prefix}}Dataset
from {{cookiecutter.library_name}}.ml_tools.models import {{cookiecutter.class_prefix}}AE, {{cookiecutter.class_prefix}}ARIMA, {{cookiecutter.class_prefix}}LSTM
from {{cookiecutter.library_name}}.utils.evaluation_tools import prediction_accuracy, summarize_training_accuracy
from {{cookiecutter.library_name}}.utils.plotting_tools import plot_prediction_accuracy, plot_summaries
from {{cookiecutter.library_name}}.utils.config import MLTrainingConfig, DatasetConfig, IoTMLConfig


project_path = Path(__file__).parents[2]

logger = logging.getLogger("train_model")
logger.level = logging.INFO


TRAINING_VERSION = "v2"

def evaluate_model(
    training_config: MLTrainingConfig,
    dataset_config: DatasetConfig,
    ) -> None:
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
        model_instance = {{cookiecutter.class_prefix}}AE(**training_config.aimodel.aimodel_params)
    elif training_params.training_type == "output_predictor":
        model_instance = {{cookiecutter.class_prefix}}LSTM(
            time_window_past=dataset_config.time_window_past,
            time_window_future=dataset_config.time_window_future,
            input_features=training_params.input_features,
            output_features=training_params.output_features,
            **training_config.aimodel.aimodel_params)
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

    state_dict = torch.load(training_prefix+".pt")
    model_instance.load_state_dict(state_dict)
    model_instance.eval()

    if training_type == "output_predictor":
        dataset= {{cookiecutter.class_prefix}}Dataset(
            training_params=training_params,
            dataset_path=dataset_path,
        )
        accuracy_results = prediction_accuracy(model=model_instance, dataset=dataset)
        fig = plot_prediction_accuracy(accuracy_results=accuracy_results)
        accuracy_summary = summarize_training_accuracy(accuracy_results=accuracy_results)

        return fig, accuracy_summary

    else:
        print("NOTHING PLANNED FOR ANOMALY ENCODER")

class ModelEvaluator:
    """Evaluation class for a model."""

    def __init__(
        self,
        model,
        model_type: str,
        state_dict_path: str | None = None,
    ) -> None:
        """Provide the trained model information.

        Parameters:
        ---

        model : torch.nn.Module
            architecture of the model trained

        model_type: str
            The type of model we are evaluating

        state_dict_path : str
            path to the ".pt" file storing the trained weights
            of the model
        """
        self.model = model
        if state_dict_path is not None:
            self.model.load_state_dict(torch.load(state_dict_path))
        self.model.eval()
        self.model_type = model_type

    def evaluate(self, dataset):
        """Decide which evaluation to perform based on the model type."""
        if self.model_type == "output_predictor":
            data = self._prediction_accuracy(dataset=dataset)
        else:
            print("NOTHING PLANED FOR ANOMALY ENCODER")
        return data

    def plot_anomaly_latent_space(self, training_set, test_set):
        """Plot latent space visualization.

        We plot the Latent space visualization of the training
        and test dataset, using the PCA decomposition function
        fitted to the train data.

        The goal of this plot is to confirm that the test data
        maps out the same regions of the latent space as the
        training data

        """
        fig, ax = plt.subplots(figsize=(10, 10))

        train_samples = []
        for i, d in enumerate(training_set):
            d = torch.unsqueeze(d, dim=0)
            latent_representation = self.model.encoder(d)
            array = latent_representation.cpu().detach().numpy()
            train_samples.append(array.flatten())

        train_samples = np.array(train_samples)
        pca = PCA(n_components=2)
        pca_model = pca.fit(train_samples)
        reduced_values_train = pca_model.transform(train_samples)

        # plot the training set first
        ax.scatter(
            reduced_values_train[:, 0],
            reduced_values_train[:, 1],
            label="training data",
            alpha=0.3,
        )

        test_samples = []
        for i, d in enumerate(test_set):
            d = torch.unsqueeze(d, dim=0)
            latent_representation = self.model.encoder(d)
            array = latent_representation.cpu().detach().numpy()
            test_samples.append(array.flatten())

        test_samples = np.array(test_samples)
        pca = PCA(n_components=2)
        reduced_values_test = pca_model.transform(test_samples)
        ax.scatter(
            reduced_values_test[:, 0],
            reduced_values_test[:, 1],
            label="test data",
            alpha=0.3,
        )
        ax.legend()
        plt.show()

    def plot_prediction_accuracy(self, data, labels: dict[Any, Any] | None = None):
        """Compare the reconstructed time series with the original data."""
        n_channels = data["real"].shape[0]
        fig, axes = plt.subplots(
            n_channels, figsize=(10, 20 * n_channels), gridspec_kw={"hspace": 1.0}
        )
        fig.suptitle("Output Predictor - Accuracy")

        for i in range(n_channels):
            if n_channels == 1:
                ax = axes
            else:
                ax = axes[i]

            if labels is None:
                label = f"output {i}"
            else:
                label = labels[i]

            ax.set_title(label)
            ax.plot(data["real"][i, :], label="real data")
            ax.plot(data["pred"][i, :], label="prediction")

            handles, lbs = ax.get_legend_handles_labels()
        fig.legend(handles, lbs, loc="center right")
        plt.show()

    def _prediction_accuracy(self, dataset):
        """Evaluate the accuracy of the model's prediction,for a given dataset."""
        self.model.cpu()

        real = []
        pred = []
        for y in dataset:
            all_inputs, outputs = y

            # add batch dimension to inputs
            all_inputs = torch.unsqueeze(all_inputs.cpu(), 0)

            # restore dimensionality of output
            outputs = torch.reshape(
                outputs, (self.model.predict_dims, self.model.predict_window)
            )

            prediction = self.model(all_inputs)
            prediction = torch.reshape(
                prediction, (self.model.predict_dims, self.model.predict_window)
            )

            pred.append(prediction.detach().numpy())
            real.append(outputs.detach().numpy())

        pred = np.hstack(pred)
        real = np.hstack(real)

        difference = (pred - real) / (real) * 100.0

        n_under = sum(difference < 0.0)
        n_over = len(difference) - n_under

        total_under = sum(difference[difference < 0.0])
        avg_under = np.median(difference[difference < 0.0])
        std_under = np.std(difference[difference < 0.0])

        total_over = sum(difference[difference > 0.0])
        return {
            "n_under": n_under,
            "n_over": n_over,
            "total_over": total_over,
            "total_under": total_under,
            "median_under_prediction": avg_under,
            "sigma_under_prediction": std_under,
            "real": real,
            "pred": pred,
            "diff": difference,
        }


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(config: DictConfig) -> None:

    config = IoTMLConfig(config)

    with PdfPages('multipage_pdf.pdf') as pdf:

        summaries = {}
        for ds_conf in config.ds:
            for train_conf in config.ml_trainings:

                training_name = train_conf.training_params.name
                keyword = f"{training_name}_{ds_conf.name}"
                fig, summary = evaluate_model(
                    training_config = train_conf,
                    dataset_config=ds_conf
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
