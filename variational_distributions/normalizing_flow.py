"""Normalizing Flow Variational Distribution."""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


class NormalizingFlowVariational:
    r"""Normalizing flow variational distribution.

    References:
        [1]: Rezende, Danilo, and Shakir Mohamed. "Variational inference with normalizing flows."
             International conference on machine learning. PMLR, 2015.

    Attributes:
        layers (lst): List of Layer objects.
        num_layers (int): Number of layers
        num_var_params (int): Number of variational parameters
    """

    def __init__(self, layers):
        """Initialize normalizing flow variational distribution.

        Args:
            layers (lst): List of Layer objects.
        """
        self.dimension = layers[0].dimension
        self.layers = layers
        self.num_layers = len(layers)
        self.num_var_params = 0
        for layer in layers:
            self.num_var_params += layer.num_var_params

    @partial(jax.jit, static_argnums=0)
    def forward(self, variational_parameters, x):
        """Forward pass through the network.

        Args:
            variational_parameters (np.array): variational parameters
            x (np.array): standard normal samples

        Returns:
            x (np.array): Transformed samples
            log_pdf (np.array): log pdf of the samples
        """
        params = variational_parameters.reshape(-1)
        x = x.reshape(-1, self.dimension)
        log_pdf = -0.5 * (jnp.log(2.0 * jnp.pi) * self.dimension + np.sum(x**2, axis=1))
        index = 0
        for layer in self.layers:
            layer_params = params[index : index + layer.num_var_params]
            index += layer.num_var_params
            x, log_det = layer.forward(layer_params, x)
            log_pdf = log_pdf - log_det
        return x, log_pdf

    def draw(self, variational_parameters, n_draws=1):
        """Draw *n_draw* samples from the variational distribution.

        Args:
            variational_parameters (np.ndarray): Variational parameters
            n_draws (int): Number of samples to draw

        Returns:
            samples (np.ndarray): Row-wise samples of the variational distribution
        """
        samples = np.random.randn(n_draws, self.dimension)
        return self.forward(variational_parameters, samples)[0]

    @partial(jax.jit, static_argnums=0)
    def logpdf(self, variational_parameters, x):
        """Logpdf evaluated using the variational parameters at samples `x`.

        Args:
            variational_parameters (np.ndarray): Variational parameters
            x (np.ndarray): Row-wise samples

        Returns:
            logpdf (np.ndarray): Row vector of the logpdfs
        """
        x = x.reshape(-1, self.dimension)

        log_pdf = 0
        index = self.num_var_params
        for layer in reversed(self.layers):
            layer_params = variational_parameters[index - layer.num_var_params : index]
            index -= layer.num_var_params
            x, log_det = layer.reverse(layer_params, x)
            log_pdf = log_pdf + log_det
        log_pdf = log_pdf - 0.5 * (jnp.log(2.0 * jnp.pi) * self.dimension + np.sum(x**2, axis=1))
        return log_pdf

    def pdf(self, variational_parameters, x):
        """Pdf of the variational distribution evaluated at samples *x*.

        First computes the logpdf, which is numerically more stable for exponential distributions.

        Args:
            variational_parameters (np.ndarray): Variational parameters
            x (np.ndarray): Row-wise samples

        Returns:
            pdf (np.ndarray): Row vector of the pdfs
        """
        return np.exp(self.logpdf(variational_parameters, x))

    @partial(jax.jit, static_argnums=0)
    def grad_params_logpdf(self, variational_parameters, x):
        """Logpdf gradient w.r.t. the variational parameters.

        Evaluated at samples *x*. Also known as the score function.

        Args:
            variational_parameters (np.ndarray): Variational parameters
            x (np.ndarray): Row-wise samples

        Returns:
            score (np.ndarray): Column-wise scores
        """
        def logpdf(arg1, arg2):
            return self.logpdf(arg1, arg2).reshape()

        return jax.vmap(jax.grad(logpdf, argnums=0), in_axes=(None, 0))(variational_parameters, x)

    @partial(jax.jit, static_argnums=0)
    def total_grad_params_logpdf(self, variational_parameters, standard_normal_sample_batch):
        """Total logpdf reparameterization gradient.

        Total logpdf reparameterization gradient w.r.t. the variational parameters.

        Args:
            variational_parameters (np.ndarray): Variational parameters
            standard_normal_sample_batch (np.ndarray): Standard normal distributed sample batch

        Returns:
            total_grad (np.ndarray): Total Logpdf reparameterization gradient
        """

        def logpdf_forward(arg1, arg2):
            return self.forward(arg1, arg2)[1].reshape()

        return jax.vmap(jax.grad(logpdf_forward, argnums=0), in_axes=(None, 0))(
            variational_parameters, standard_normal_sample_batch)

    @partial(jax.jit, static_argnums=0)
    def grad_sample_logpdf(self, variational_parameters, sample_batch):
        """Computes the gradient of the logpdf w.r.t. *x*.

        Args:
            sample_batch (np.ndarray): Row-wise samples
            variational_parameters (np.ndarray): Variational parameters

        Returns:
            gradients_batch (np.ndarray): Gradients of the log-pdf w.r.t. the
            sample *x*. The first dimension of the array corresponds to
            the different samples. The second dimension to different dimensions
            within one sample. (Third dimension is empty and just added to
            keep slices two dimensional.)
        """

        def logpdf(arg):
            return self.logpdf(variational_parameters, arg).reshape()

        return jax.vmap(jax.grad(logpdf), in_axes=0)(sample_batch)

    @partial(jax.jit, static_argnums=0)
    def grad_params_reparameterization(
        self, variational_parameters, standard_normal_sample_batch, upstream_gradient
    ):
        r"""Calculate the gradient of the reparameterization.

        Args:
            variational_parameters (np.ndarray): Variational parameters
            standard_normal_sample_batch (np.ndarray): Standard normal distributed sample batch
            upstream_gradient (np.array): Upstream gradient

        Returns:
            gradient (np.ndarray): Gradient of the upstream function w.r.t. the variational
                                   parameters.
        """

        def vjp(x, upstream):
            def reparameterization(params):
                return self.forward(params, x)[0].reshape(-1)

            _, f_vjp = jax.vjp(reparameterization, variational_parameters)
            return f_vjp(upstream)

        gradient = jax.vmap(vjp, in_axes=(0, 0))(standard_normal_sample_batch, upstream_gradient)[0]

        return gradient

    def conduct_reparameterization(self, variational_parameters, n_samples):
        """Conduct a reparameterization.

        Args:
            variational_parameters (np.ndarray): Array with variational parameters
            n_samples (int): Number of samples for current batch

        Returns:
            * samples_mat (np.ndarray): Array of actual samples from the
              variational distribution
            * standard_normal_sample_batch (np.ndarray): Standard normal
              distributed sample batch
        """
        standard_normal_samples = np.random.randn(n_samples, self.dimension)
        samples = self.forward(variational_parameters, standard_normal_samples)[0]

        return samples, standard_normal_samples

    def fisher_information_matrix(self, variational_parameters):
        """Compute the fisher information matrix.

        Depends on the variational distribution for the given
        parameterization.

        Args:
            variational_parameters (np.ndarray):  variational parameters (1 x n_params)
        """
        raise NotImplementedError

    def initialize_variational_parameters(self):
        """Initialize variational parameters.

        Returns:
            variational_parameters (np.ndarray):  variational parameters (1 x n_params)
        """
        return np.random.uniform(low=-1.0, high=1.0, size=self.num_var_params)

    def export_dict(self, variational_parameters):
        """Create a dict of the distribution based on the given parameters.

        Args:
            variational_parameters (np.ndarray): Variational parameters

        Returns:
            export_dict (dictionary): Dict containing distribution information
        """
        export_dict = {
            "type": "normalizing_flow",
            "variational_parameters": variational_parameters,
        }
        return export_dict


class PlanarLayer:
    """Planar layer.

    References:
        [1]: Rezende, Danilo, and Shakir Mohamed. "Variational inference with normalizing flows."
             International conference on machine learning. PMLR, 2015.

    Attributes:
        dimension (int): Dimensionality of the distribution
        num_var_params (int): Number of variational parameters for the layer
        activation (obj): Activation function object
    """

    def __init__(self, dimension, activation):
        """Initialize the planar layer.

        Args:
            dimension (int): Dimensionality of the distribution
            activation (obj): Activation function object
        """
        self.dimension = dimension
        self.num_var_params = 2 * dimension + 1
        self.activation = activation

    @partial(jax.jit, static_argnums=0)
    def forward(self, params, inputs):
        """Forward pass through the layer.

        Args:
            params (np.array): Variational parameters of the layer
            inputs (np.array): Input samples

        Returns:
            outputs (np.array): Transformed samples
            log_det (np.array): Log determinant of the transformation
        """
        weights = params[: self.dimension]
        scaler = params[self.dimension : -1]
        bias = params[-1]

        linear = jnp.sum(weights[jnp.newaxis, :] * inputs, axis=1) + bias
        inner = jnp.sum(weights * scaler)

        scaler_ = scaler + (jnp.log(1 + jnp.exp(inner)) - 1 - inner) * weights / jnp.sum(
            weights**2
        )  # constraint w.T * u > -1

        outputs = (
            inputs + scaler_[jnp.newaxis, :] * self.activation.evaluate(linear)[:, jnp.newaxis]
        )
        log_det = jnp.log(jnp.abs(1 + jnp.sum(weights * scaler_) * self.activation.grad(linear)))

        return outputs, log_det

    @partial(jax.jit, static_argnums=0)
    def reverse(self, params, outputs):
        """Reverse pass through the layer.

        Args:
            params (np.array): Variational parameters of the layer
            outputs (np.array): Output samples

        Returns:
            inputs (np.array): Back-transformed samples
            log_det (np.array): Log determinant of the transformation
        """
        if not isinstance(self.activation, LeakyRelu):
            raise ValueError("Reverse is not defined for ", str(self.activation))

        weights = params[: self.dimension]
        scaler = params[self.dimension : -1]
        bias = params[-1]

        linear = jnp.sum(weights[jnp.newaxis, :] * outputs, axis=1) + bias
        inner = jnp.sum(weights * scaler)

        scaler_ = scaler + (jnp.log(1 + jnp.exp(inner)) - 1 - inner) * weights / jnp.sum(
            weights**2
        )

        a = (linear < 0) * (self.activation.negative_slope - 1.0) + 1.0

        scaler_ = jnp.outer(a, scaler_)
        inner_ = jnp.sum(weights[jnp.newaxis, :] * scaler_, axis=1)

        inputs = outputs - scaler_ * (linear / (1 + inner_))[:, jnp.newaxis]
        log_det = -jnp.log(jnp.abs(1 + inner_))

        return inputs, log_det


class MeanFieldNormalLayer:
    """Mean-Field Normal Layer.

    Attributes:
        dimension (int): Dimensionality of the distribution
        num_var_params (int): Number of variational parameters for the layer
    """

    def __init__(self, dimension):
        """Initialize the mean-field normal layer.

        Args:
            dimension (int): Dimensionality of the distribution
        """
        self.dimension = dimension
        self.num_var_params = 2 * dimension

    def forward(self, params, inputs):
        """Forward pass through the layer.

        Args:
            params (np.array): Variational parameters of the layer
            inputs (np.array): Input samples

        Returns:
            outputs (np.array): Transformed samples
            log_det (np.array): Log determinant of the transformation
        """
        mean = params[: self.dimension][jnp.newaxis, :]
        log_std = params[self.dimension :][jnp.newaxis, :]
        std = jnp.exp(log_std)
        outputs = inputs * std + mean
        log_det = jnp.sum(log_std)

        return outputs, log_det

    def reverse(self, params, outputs):
        """Reverse pass through the layer.

        Args:
            params (np.array): Variational parameters of the layer
            outputs (np.array): Output samples

        Returns:
            inputs (np.array): Back-transformed samples
            log_det (np.array): Log determinant of the transformation
        """
        mean = params[: self.dimension][jnp.newaxis, :]
        log_std = params[self.dimension : -1][jnp.newaxis, :]
        std = jnp.exp(log_std)
        inputs = (outputs - mean) / std
        log_det = -jnp.sum(log_std)

        return inputs, log_det


class FullRankNormalLayer:
    """Full-Rank Normal Layer.

    Attributes:
        dimension (int): Dimensionality of the distribution
        num_var_params (int): Number of variational parameters for the layer
    """

    def __init__(self, dimension):
        """Initialize the full-rank normal layer.

        Args:
            dimension (int): Dimensionality of the distribution
        """
        self.dimension = dimension
        self.num_var_params = (dimension * (dimension + 1)) // 2 + dimension

    def forward(self, params, inputs):
        """Forward pass through the layer.

        Args:
            params (np.array): Variational parameters of the layer
            inputs (np.array): Input samples

        Returns:
            outputs (np.array): Transformed samples
            log_det (np.array): Log determinant of the transformation
        """
        mean = params[: self.dimension][jnp.newaxis, :]
        cholesky_vector = params[self.dimension :]
        idx = jnp.tril_indices(self.dimension, k=0, m=self.dimension)
        cholesky = jnp.zeros((self.dimension, self.dimension)).at[idx].set(cholesky_vector)

        outputs = jnp.dot(inputs, cholesky.T) + mean
        log_det = jnp.sum(jnp.log(jnp.abs(jnp.diag(cholesky))))

        return outputs, log_det


class LeakyRelu:
    """Leaky Relu activation function.

    Attributes:
        negative_slope (float): Slope for negative input values
    """

    def __init__(self, negative_slope=0.2):
        """Initialize LeakyRelu object.

        Args:
            negative_slope (float, opt): Slope for negative input values
        """
        self.negative_slope = negative_slope

    def evaluate(self, x):
        """Evaluate activation at x.

        Args:
            x (array_like): Input array

        Returns:
            jnp.array: activation LeakyRelu(x)
        """
        return jnp.where(x >= 0, x, self.negative_slope * x)

    def grad(self, x):
        """Evaluate gradient of Leaky Relu at x.

        Args:
            x (array_like): Input array

        Returns:
            jnp.array: Gradient of Leaky Relu at x
        """
        return jnp.where(x >= 0, jnp.ones(x.shape), self.negative_slope * jnp.ones(x.shape))


class Tanh:
    """Tanh activation function."""

    @staticmethod
    def evaluate(x):
        """Evaluate tanh at x.

        Args:
            x (array_like): Input array

        Returns:
            jnp.array: activation tanh(x)
        """
        return jnp.tanh(x)

    @staticmethod
    def grad(x):
        """Evaluate gradient of tanh at x.

        Args:
            x (array_like): Input array

        Returns:
            jnp.array: Gradient of tanh at x
        """
        return 1 - jnp.tanh(x) ** 2
