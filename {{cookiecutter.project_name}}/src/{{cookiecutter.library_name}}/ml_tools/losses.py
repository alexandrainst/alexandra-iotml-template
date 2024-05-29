"""File containing the pytorch customized Loss functions."""

import logging
from collections import OrderedDict
from typing import Any, Dict, List

import torch
import torch.nn as nn
from {{cookiecutter.library_name}}.ml_tools.exceptions import (
    DimensionError,
    MissingRequiredFeatures,
    SnippetsKeyMismatch,
    )

logger = logging.getLogger("ml_tools.losses")


class RecoLoss(nn.Module):
    """Standard squared loss on reconstructed data."""

    def __init__(self):
        """Standard squared loss on reconstructed data."""
        super(RecoLoss, self).__init__()

    def forward(
        self, output_ts: Dict[str, torch.Tensor], truth_ts: Dict[str, torch.Tensor]
    ) -> float:
        """Compare model output with ground truth.

        Parameters:
        ----

        output_ts:  dict
            the output of the model, whose keys are
            feature tensors of shape [n_batch, time_window_future]

        truth_ts:  dict
            the true values against wich the output must be compared.
            Must have the same shape+structure as output_ts.
        """
        if output_ts.keys() != truth_ts.keys():
            raise SnippetsKeyMismatch

        one_element = list(truth_ts.keys())[0]
        if output_ts[one_element].shape != truth_ts[one_element].shape:
            raise DimensionError(
                output_ts[one_element].shape, truth_ts[one_element].shape
            )

        losses = []
        for k in output_ts.keys():
            feature_loss = ((output_ts[k] - truth_ts[k]) ** 2.0)
            losses.append(feature_loss.sum(dim=1))

        loss = torch.stack(losses, dim=0).sum(dim=0)

        # average over batch size
        return loss.mean()


class UnitarityLoss(nn.Module):
    """Loss incorporating physics contraints."""

    def __init__(self, required_features: List ):
        """Loss incorporating physics contraints."""
        super(UnitarityLoss, self).__init__()
        self.required_features = set(required_features)
        self.reco_loss = RecoLoss()

    def forward(
        self, output_ts: Dict[str, torch.Tensor], truth_ts: Dict[str, torch.Tensor],
    ) -> float:
        """Squared loss with + unitarity constraints on some of the features.

        Loss evaluation onyl works if all required features are included
        in the time series snippet.
        """
        output_keys = set(output_ts.keys())

        if not self.required_features.issubset(output_keys):
            raise MissingRequiredFeatures(output_keys, self.required_features)

        reco_part = self.reco_loss(output_ts, truth_ts)

        # unity loss function (mole fractions must sum up to 1.0)
        relevant_outputs = torch.stack([output_ts[k] for k in self.required_features])
        unity_loss = (1.0 - relevant_outputs.sum(dim=0)) ** 2.0
        unity_loss = (unity_loss.sum(dim=1)).mean()

        return reco_part + unity_loss
