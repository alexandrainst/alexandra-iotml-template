"""File containing pytorch models.

This file contains model architectures
used in the project. Here are by default
some models which have been previously used

"""
import logging
from collections import OrderedDict
from typing import Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from {{ cookiecutter.library_name }}.ml_tools.exceptions import DimensionError
from statsmodels.tsa.arima.model import ARIMA

logger = logging.getLogger("ml_tools.models")


#
# Example LSTM Cell
#
class LSTMCell(nn.Module):
    """Multivariate LSTM prediction model."""
    def __init__(
        self,
        time_window_past: int,
        time_window_future: int,
        input_features: OrderedDict[int, Any],
        output_features: OrderedDict[int, Any],
        n_hidden: int | None = None,
    ) -> None:
        """Multivariate LSTM prediction model.

        Parameters:
        ---

        time_window_past: int
            number of time steps taken in the past

        time_window_future: int
            number of time steps taken in the future

        input_features: OrderedDict
            list of input features

        output_features: OrderedDict
            list of output features

        n_hidden: int
            size of the hidden state h
        """
        super(LSTMCell, self).__init__()
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

    def shape_dict_to_lstm_input(self, x: OrderedDict[str, torch.Tensor]) -> torch.Tensor:
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

    def reshape_output_to_dict(self, output: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        """Reshape the model output into a dict.

        This function takes in a tensor of size:

        [time_window_past, batch_size, n_features]

        It returns a dict of output features that have size:

        [batch_size, time_window_future]

        """
        output = output[self.time_window_past - self.time_window_future :, :, :]

        out_dict = OrderedDict()
        for i, k in self.output_features.items():
            out_dict[k] = output[:, :, i].transpose(0, 1)
        return out_dict

    def forward(self, x: OrderedDict[str, torch.Tensor])-> OrderedDict[str, torch.Tensor]:
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
class LinearEncoder(nn.Module):
    """Linear encoder of arbitrary depth."""

    def __init__(self, input_dims: int, latent_dims: List[int]):
        """Initialize model.

        Parameters
        ----

        input_dims: int
            the initial size of the feature vector inserted.

        latent_dims: List
            A list of successively smaller dimensions for the
            inner layers of the encoder. The list is by default
            sorted from largest to smallest
        """
        super(LinearEncoder, self).__init__()

        latent_dims = sorted(latent_dims, reverse=True)
        if input_dims<=latent_dims[0]:
            raise Exception("autoencoder input size smaller than first deep layer.")

        linear_layers = OrderedDict()
        in_layer = input_dims
        for i, layer_dims in enumerate(latent_dims):
            layer = nn.Linear(in_layer, layer_dims)
            in_layer = layer_dims
            linear_layers[f"linear_{i}"] = layer
            linear_layers[f"relu_{i}"] = nn.ReLU()

        self.linear_layers = nn.Sequential(linear_layers)


    def forward(self, x):
        """Forward pass on the model."""
        x = torch.flatten(x, start_dim=1)
        x = self.linear_layers(x)
        return x


class LinearDecoder(nn.Module):
    """Linear Decoder of arbitrary depth."""

    def __init__(self, latent_dims: List[int], output_dims: int):
        """Initialize model.

        Parameters
        ----

        latent_dims: List
            A list of successively smaller dimensions for the
            inner layers of the encoder. The list is by default
            sorted from largest to smallest

        output_dims: int
            the final size of the output feature vector
        """
        super(LinearDecoder, self).__init__()
        latent_dims = sorted(latent_dims)
        if output_dims<=latent_dims[-1]:
            raise Exception("autoencoder output size smaller than last deep layer.")
        
        linear_layers = OrderedDict()
        in_layer = latent_dims[0]
        i = 0
        for layer_dims in latent_dims[1:]:
            linear_layers[f"linear_{i}"] = nn.Linear(in_layer, layer_dims)
            linear_layers[f"relu_{i}"] = nn.ReLU()
            in_layer = layer_dims
            i+=1

        linear_layers[f"linear_{i}"] = nn.Linear(in_layer, output_dims)

        self.linear_layers = nn.Sequential(linear_layers)

    def forward(self, x):
        """Forward pass on the model."""
        x = torch.flatten(x, start_dim=1)
        x = self.linear_layers(x)
        return x


class LinearAE(nn.Module):
    """Autoencoder model of arbitrary depth."""

    def __init__(self, 
        time_window_past: int,
        input_features: OrderedDict[int, str],
        latent_dims: List[int]):
        """Autoencoder Model.

        Parameters:
        ---

        time_window_past: int
            number of time steps taken in the past

        input_features: OrderedDict
            list of input features

        latent_dims: List[int]
            dimensional size of the autoencoder bottleneck
        """
        super(LinearAE, self).__init__()
        self.n_features = len(list(input_features.keys()))
        self.input_features = input_features
        self.time_window_past = time_window_past
        self.latent_dims = latent_dims
        self.input_size = self.n_features * self.time_window_past

        self.encoder = LinearEncoder(self.input_size, self.latent_dims)
        self.decoder = LinearDecoder(self.latent_dims, self.input_size)


    def shape_dict_to_ae_input(self, x: OrderedDict[str, torch.Tensor]) -> torch.Tensor:
        """Shape the dictionary input into an autoencoder input tensor.

        We use the batch_first=False convention used in pytorch.
        x is shaped as a dict with input features of size:

        [batch_size, time_window_past]

        and we want to change this into a tensor of size:

        [batch_size, n_features*time_window_past]
        """
        one_element = list(x.keys())[0]
        input_dim = x[one_element].shape
        if len(input_dim) != 2:
            raise DimensionError(
                input_dimension=input_dim,
                required_dimension="[batch_size, time_window_past]",
            )

        try:
            y = torch.hstack([v for k, v in x.items()])
        except Exception:
            raise Exception(
                "Dimension problem. input vector must have batch dimension."
            )
        return y

    def reshape_output_to_dict(self, output: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        """Reshape the model output into a dict.

        This function takes in a tensor of size:

        [batch_size, n_features*time_window_past]

        It returns a dict of output features that have size:

        [batch_size, time_window_future]

        """
        out_dict= OrderedDict()
        for i, k in self.input_features.items():
            out_dict[k] = output[
                :, (i * self.time_window_past) : (i + 1) * self.time_window_past
            ]

        return out_dict

    def forward(self, x):
        """Forward pass on the Autoencoder.

        The forward output is a dict of features with size:

        [batch_size, time_window_past]
        """

        x = self.shape_dict_to_ae_input(x)
        z = self.encoder(x)
        x = self.decoder(z)
        x = self.reshape_output_to_dict(x)

        return x



#
# Variational Autoencoder
#
class LinearVAE(LinearAE):
    """Add Variational component to the LinearAE architecture."""

    def __init__(self,
        time_window_past: int,
        input_features: OrderedDict[int, str],
        latent_dims: List[int]
        ):
        """Initialize model."""
        super().__init__(
            time_window_past=time_window_past,
            input_features=input_features,
            latent_dims=latent_dims
        )

        # Add another encoder layer that maps out stddev for the reparametrization
        self.logvar_encoder = LinearEncoder(self.input_size, latent_dims)
        self.logvar_decoder = LinearDecoder(latent_dims, self.input_size)

    def forward(self, x):
        """The entire pipeline of the VAE.

        encoder -> reparameterization -> decoder.
        """
        mu = self.encoder(x)
        logVar = self.logvar_encoder(x)

        z = self.reparameterize(mu, logVar)
 
        out = self.decoder(z)
        
        return out, mu, logVar

    def reparameterize(self, mu, logVar):
        """Takes in the input mu and logVar and sample the mu + std * eps."""
        if self.training:
            std = torch.exp(logVar/2)
            eps = torch.randn_like(std)
            return mu + std * eps

        return mu
 


#
# Rolling ARIMA model
#
class ARIMAModel:
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
        predict_variables: OrderedDict[Any, Any] | None = None,
        input_variables: OrderedDict[Any, Any] | None = None,
    ):
        """Initialize model."""
        super(ARIMAModel, self).__init__()
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
