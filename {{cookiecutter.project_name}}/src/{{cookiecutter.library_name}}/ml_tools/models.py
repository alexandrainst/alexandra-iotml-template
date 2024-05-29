"""File containing pytorch models.

This file contains model architectures
used in the project. Here are by default
some models which have been previously used

"""

import logging
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from {{ cookiecutter.library_name }}.ml_tools.exceptions import DimensionError
from statsmodels.tsa.arima.model import ARIMA

logger = logging.getLogger(__name__)


#
# Example LSTM Cell
#
class {{ cookiecutter.class_prefix }}LSTM(nn.Module):
    """Multivariate LSTM prediction model."""
    def __init__(
        self,
        time_window_past: int,
        time_window_future: int,
        input_features: Dict[int, Any],
        output_features: Dict[int, Any],
        n_hidden: int | None = None,
    ) -> None:
        """Multivariate LSTM prediction model.

        Parameters:
        ---

        time_window_past: int
            number of time steps taken in the past

        time_window_future: int
            number of time steps taken in the future

        input_features: Dict
            list of input features

        output_features: Dict
            list of output features

        n_hidden: int
            size of the hidden state h
        """
        super({{ cookiecutter.class_prefix }}LSTM, self).__init__()
        self.n_hidden = n_hidden if n_hidden is not None else time_window_past
        self.n_layers = 1

        self.time_window_past = time_window_past
        self.time_window_future = time_window_future

        # Calculate the number of input features
        self.input_features = input_features
        self.output_features = output_features
        self.n_input_features = len(list(input_features.keys()))
        self.n_output_features = len(list(output_features.keys()))

        self.lstm = nn.LSTM(
            input_size=self.n_input_features,
            hidden_size=n_hidden,
            num_layers=self.n_layers,
            batch_first=False,
        )

        # Morph LSTM output into an output of n_predict dimension
        self.linear = nn.Linear(n_hidden, self.n_output_features)

    def shape_dict_to_lstm_input(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Shape the dictionary input into an LSTM input tensor.

        We use the batch_first=False convention used in pytorch.
        x is shaped as a dict with input festures of size:

        [batch_size, time_steps]
        
        and we want to change this into a tensor of size:
        
        [time_window_past, batch_size, n_features]
        """
        one_element = list(x.keys())[0]
        input_dim = x[one_element].shape
        if len(input_dim) != 2:
            raise DimensionError(
                input_dimension=input_dim, required_dimension="[batch_size, time_steps]"
            )
        
        try:
            y = torch.vstack([v[None, :, :] for k, v in x.items()])
            y = torch.transpose(y, 0, 2)
        except Exception as e:
            raise Exception(
                "Dimension problem. input vector must have batch dimension."
            )
        return y

    def reshape_output_to_dict(self, output: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Reshape the model output into a dict.

        This function takes in a tensor of size:

        [time_window_past, batch_size, n_features]

        It returns a dict of output features that have size:

        [batch_size, time_window_future]

        """
        output = output[self.time_window_past - self.time_window_future :, :, :]

        out_dict = {}
        for i, k in self.output_features.items():
            out_dict[k] = output[:, :, i].transpose(0, 1)
        return out_dict

    def forward(self, x: Dict[str, torch.Tensor]):
        """Forward pass on the LSTM cell.

        We implement a couple dimensions checks to ensure that the
        batch and sequence length dimension are not being confused.

        The forward output is a dict of features with size:

        [batch_size, time_window_future]
        """
        x = self.shape_dict_to_lstm_input(x)
       
        # flatten parameter storage for GPU's:
        self.lstm.flatten_parameters()

        # Note: when unspecified, c0 and h0 are initialized to zero in the lstm cell
        output, (final_hidden_state, final_cell_state) = self.lstm(x)

        # convert the n_hidden dimensions to the desired output size:
        x = self.linear(output)
        # reshape into an output dict:
        x = self.reshape_output_to_dict(x)

        return x


#
# Autoencoders for anomaly detection
#
class {{ cookiecutter.class_prefix }}Encoder(nn.Module):
    """Encoder section of autoencoder model."""

    def __init__(self, input_dims, latent_dims):
        """Initialize model."""
        super({{ cookiecutter.class_prefix }}Encoder, self).__init__()
        self.linear1 = nn.Linear(input_dims, 56)
        self.linear2 = nn.Linear(56, latent_dims)

    def forward(self, x):
        """Forward pass on the model."""
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        return self.linear2(x)


class {{ cookiecutter.class_prefix }}Decoder(nn.Module):
    """Decoder section of autoencoder model."""

    def __init__(self, latent_dims, output_dims):
        """Initialize model."""
        super({{ cookiecutter.class_prefix }}Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 56)
        self.linear2 = nn.Linear(56, output_dims)

    def forward(self, x):
        """Forward pass on the model."""
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        return self.linear2(x)


class {{ cookiecutter.class_prefix }}AE(nn.Module):
    """Autoencoder model for anomaly detection."""

    def __init__(self, input_dims, input_window, latent_dims):
        """Initialize model."""
        super({{ cookiecutter.class_prefix }}AE, self).__init__()
        input_size = input_dims * input_window
        self.encoder = {{ cookiecutter.class_prefix }}Encoder(input_size, latent_dims)
        self.decoder = {{ cookiecutter.class_prefix }}Decoder(latent_dims, input_size)

    def forward(self, x):
        """Forward pass on the model."""
        z = self.encoder(x)
        return self.decoder(z)


#
# Rolling ARIMA model
#
class {{ cookiecutter.class_prefix }}ARIMA:
    """Rolling ARIMA Model.

    Here we don't use an ML technique,
    but rather provide a forecast based on
    the statistical properties of the provided snippet.

    The model is however wrapped inside a torch-like object
    to facilitate its use on the same scripts as our ML
    models.
    """

    def __init__(
        self,
        input_window: int,
        predict_window: int,
        p: int,
        d: int,
        q: int,
        predict_variables: dict[Any, Any] | None = None,
        input_variables: dict[Any, Any] | None = None,
    ):
        """Initialize model."""
        super({{ cookiecutter.class_prefix }}ARIMA, self).__init__()
        self.p = p
        self.d = d
        self.q = q
        self.predict_window = predict_window
        self.predict_dims = 1
        self.input_size = input_window

    def eval(self):
        """Dummy function."""
        pass

    def cpu(self):
        """Dummy function."""
        pass

    def forward(self, x):
        """Forward pass is actually just an local eval of ARIMA."""
        x = x.detach().numpy().flatten()
        self.model = ARIMA(x, order=(self.p, self.d, self.q))
        self.fit = self.model.fit()
        return torch.Tensor([self.fit.forecast(steps=self.predict_window)])

    def __call__(self, x):
        """Dummy link to the model's call function."""
        return self.forward(x)
