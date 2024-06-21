"""Test the models and training/testing classes."""
import glob
import logging
import os
from pathlib import Path

import numpy as np
import torch
from {{ cookiecutter.library_name }}.ml_tools.datasets import TimeSnippetDataset
from {{ cookiecutter.library_name }}.ml_tools.losses import RecoLoss, UnitarityLoss
from {{ cookiecutter.library_name }}.ml_tools.models import LinearAE, LSTMCell
from {{ cookiecutter.library_name }}.utils.config import TrainingParams
from {{ cookiecutter.library_name }}.utils.training_tools import generate_dataset
from torch.utils.data import DataLoader

PROJECT_PATH = Path(__file__).parents[1]
logger = logging.getLogger("test.test_ml_components")
TRAINING_VERSION = "v2"



#
# Testing dataset generation.
#
def create_dataset():
    """Create a small dataset made up of train and test samples.

    We also show an example of how you can write down valid
    configs for ML trainings and datasets.

    NOTE: works only if postgres database is connected (ie
    it cannot run in github's automated test pipeline for now.)
    """
    dataset_config = {
        "name": "example_ds",
        "time_window_past": 10,
        "time_window_future": 5,
        "subsets": [
            {
                "name": "train",
                "time_periods": {
                    "start": "2024-01-02 00:00:00+00",
                    "end": "2024-01-02 8:30:00+00",
                },
            },
            {
                "name": "test",
                "time_periods": {
                    "start": "2024-01-02 09:00:00+00",
                    "end": "2024-01-02 9:30:00+00",
                },
            },
        ],
    }

    generate_dataset(dataset_config=dataset_config)

    expected_output_path = os.path.join(
        PROJECT_PATH, f"data/processed/{TRAINING_VERSION}/", "example_ds_dataset/"
    )

    assert (
        len(glob.glob(os.path.join(expected_output_path, "train", "*.pt"))) != 0
    ), f"ERROR: no training dataset found in {expected_output_path}"
    assert (
        len(glob.glob(os.path.join(expected_output_path, "test", "*.pt"))) != 0
    ), f"ERROR: no test dataset found in {expected_output_path}"
    logger.info("PASSED\n-----\n")


def test_read_dataset():
    """Readout the dataset created above."""
    logger.info("\nTesting dataset readout...\n")

    expected_output_path = os.path.join(
        PROJECT_PATH, f"data/processed/{TRAINING_VERSION}/", "example_ds_dataset/"
    )

    training_params = {
        "training_type": "output_predictor",
        "name": "example_training",
        "input_features": {0: "input_p", 1: "input_t"},
        "output_features": {0: "output_p", 1: "output_t"},
        "learning_rate": 0.001,
        "batch_size": 30,
        "n_epochs": 3,
        "loss": "RecoLoss",
    }

    for ds_type in ["test", "train"]:
        logger.debug(f"loading data from {ds_type} set...")
        ds = TimeSnippetDataset(
            training_params=TrainingParams(**training_params),
            dataset_path=os.path.join(expected_output_path, ds_type),
        )
        logger.debug("loading element from dataset...")
        for element in ds:
            logger.debug(f"example data element: {element}")
            p = element["input"]["input_t"]
            logger.debug(f"Past input temperature shape: {p.shape}")
            f = element["truth"]["output_t"]
            logger.debug(f"Future output temperature shape: {f.shape}")
            break

        logger.debug("loading element from a dataloader")
        dl = DataLoader(ds, batch_size=30)

        for batch_element in dl:
            # logger.debug(f"example batch element: {batch_element}")
            e = batch_element["input"]["input_t"]
            logger.debug(f"batched input_t: {e.shape}")
            break

    logger.info("PASSED\n-----\n\n")


#
# Functions to test models
#
def test_ae_model():
    """Initialize a model, and return its output."""
    logger.info("\nTesting Autoencoder model...\n")

    model = LinearAE(
        time_window_past=10,
        input_features={0: "input_p", 1: "input_t"},
        latent_dims=[33],
    )

    logger.debug(f"Encoder weights level 1: {model.encoder.linear1.weight}")
    logger.debug(f"Encoder weights level 2: {model.encoder.linear2.weight}")
    logger.debug(f"Decoder weights level 1: {model.decoder.linear1.weight}")
    logger.debug(f"Decoder weights level 2: {model.decoder.linear2.weight}")

    assert model.encoder.linear1.weight.shape == (
        56,
        2 * 10,
    ), f"ERROR: input-hidden weights dimensions should be \
    (56, input_dims) {model.encoder.linear1.weight.shape}"

    assert model.encoder.linear2.weight.shape == (
        33,
        56,
    ), f"ERROR: input-hidden weights dimensions should be \
    (latent_dims, 56) {model.encoder.linear2.weight.shape}"

    # example of a batched vector of batch size 3
    example_data = {
        "input_p": torch.Tensor(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
                [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            ]
        ),
        "input_t": torch.Tensor(
            [
                [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0],
                [-2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0],
                [-3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0],
            ]
        ),
    }

    batch_vec = example_data["input_t"]
    logger.debug(f"input tensor shape: {batch_vec.shape}")
    output = model(example_data)

    actual_shape = output["input_t"].shape
    assert (
        output["input_t"].shape == example_data["input_t"].shape
    ), f"ERROR: expected output shape per feature: [n_batch, n_time_steps].\
    Actual shape: {actual_shape}"
    logger.info("PASSED\n-----\n\n")


def test_lstm_model():
    """Initialize a model, and return its output."""
    logger.info("\nTesting LSTM model...\n")

    model = LSTMCell(
        time_window_past=10,
        time_window_future=5,
        input_features={0: "input_p", 1: "input_t"},
        output_features={0: "output_p", 1: "output_t"},
        n_hidden=285,
    )

    logger.debug(f"LSTM cell, input-hidden weights: {model.lstm.weight_ih_l0.shape}")
    logger.debug(f"LSTM cell, hidden-hidden weights: {model.lstm.weight_hh_l0.shape}")

    assert model.lstm.weight_ih_l0.shape == (
        4 * 285,
        2,
    ), f"ERROR: input-hidden weights dimensions should be \
    (4*n_hidden, input_dims) {model.lstm.weight_ih_l0.shape}"

    assert model.lstm.weight_hh_l0.shape == (
        4 * 285,
        285,
    ), f"ERROR: input-hidden weights dimensions should be \
    (4*n_hidden, hidden_dims) {model.lstm.weight_hh_l0.shape}"

    # example of a batched vector of batch size 3
    example_data = {
        "input_p": torch.Tensor(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
                [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            ]
        ),
        "input_t": torch.Tensor(
            [
                [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0],
                [-2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0],
                [-3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0],
            ]
        ),
    }

    batch_vec = example_data["input_t"]
    logger.debug(f"input tensor shape: {batch_vec.shape}")
    output = model(example_data)

    actual_shape = output["output_t"].shape
    assert output["output_t"].shape == (
        3,
        5,
    ), f"ERROR: expected output shape pre features: \
    [n_batch, n_time_steps]. Actual shape: {actual_shape}"
    logger.info("PASSED\n-----\n\n")


#
# Test loss functions
#
def test_losses():
    """Evaluate the loss between two tensors."""
    logger.debug("testing the loss functions...")

    # The batch size is 2 in this example
    example_output = {
        "output_meth": torch.Tensor(
            [
                [0.9765, 0.9762, 0.9758, 0.9752, 0.9723],
                [0.9745, 0.9742, 0.9738, 0.9731, 0.9717],
            ]
        ),
        "output_h2": torch.Tensor(
            [
                [0.00113, 0.00113, 0.00114, 0.00113, 0.00114],
                [0.00114, 0.00113, 0.00113, 0.00113, 0.00113],
            ]
        ),
        "output_co": torch.Tensor(
            [
                [0.0349, 0.0349, 0.0349, 0.0348, 0.0348],
                [0.0348, 0.0348, 0.0347, 0.0347, 0.0346],
            ]
        ),
        "output_h2o": torch.Tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]
        ),
        "output_co2": torch.Tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]
        ),
    }

    example_truth = {
        "output_meth": torch.Tensor(
            [[0.97, 0.97, 0.97, 0.97, 0.97], [0.97, 0.97, 0.97, 0.97, 0.97]]
        ),
        "output_h2": torch.Tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]
        ),
        "output_co": torch.Tensor(
            [[0.03, 0.03, 0.03, 0.03, 0.03], [0.03, 0.03, 0.03, 0.03, 0.03]]
        ),
        "output_h2o": torch.Tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]
        ),
        "output_co2": torch.Tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]
        ),
    }

    # calculate the loss manually here:
    true_reco_loss = 0.00022692855
    true_unitarity_loss = 0.0005389155500000022

    loss_reco = RecoLoss()(output_ts=example_output, truth_ts=example_truth)
    loss_unitarity= UnitarityLoss()(output_ts=example_output, truth_ts=example_truth)

    assert np.allclose(
        loss_reco, true_reco_loss, rtol=1e-10
    ), f"ERROR: error in loss function calculation {loss_reco}"

    assert np.allclose(
        loss_unitarity, true_unitarity_loss, rtol=1e-10
    ), f"ERROR: error in loss function calculation {loss_unitarity}"
    logger.info("PASSED\n-----\n\n")
