"""Adamax optimizer."""

import numpy as np

from stochastic_optimizers.stochastic_optimizer import StochasticOptimizer
from stochastic_optimizers.exponential_averaging import ExponentialAveraging


class Adamax(StochasticOptimizer):
    r"""Adamax stochastic optimizer [1]. *eps* added to avoid division by zero.

    References:
        [1] Kingma and Ba. "Adam: A Method for Stochastic Optimization".  ICLR 2015. 2015.

    Attributes:
        beta_1 (float): :math:`\beta_1` parameter as described in [1].
        beta_2 (float): :math:`\beta_2` parameter as described in [1].
        m (ExponentialAveragingObject): Exponential average of the gradient.
        u (np.array): Maximum gradient momentum.
        eps (float): Nugget term to avoid a division by values close to zero.
    """

    _name = "Adamax Stochastic Optimizer"

    def __init__(
        self,
        learning_rate,
        optimization_type,
        beta_1=0.9,
        beta_2=0.999,
        eps=1e-8,
    ):
        """Initialize optimizer.

        Args:
            learning_rate (float): Learning rate for the optimizer
            optimization_type (str): "max" in case of maximization and "min" for minimization
            beta_1 (float): :math:`beta_1` parameter as described in [1]
            beta_2 (float): :math:`beta_1` parameter as described in [1]
            eps (float): Nugget term to avoid a division by values close to zero
        """
        super().__init__(
            learning_rate=learning_rate,
            optimization_type=optimization_type,
        )
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.m = ExponentialAveraging(coefficient=beta_1)
        self.u = 0
        self.eps = eps

    def scheme_specific_gradient(self, gradient):
        """Adamax gradient computation.

        Args:
            gradient (np.array): Gradient

        Returns:
            gradient (np.array): Adam gradient
        """
        if self.iteration == 0:
            self.m.current_average = np.zeros(gradient.shape)
            self.u = np.zeros(gradient.shape)

        m_hat = self.m.update_average(gradient)
        m_hat /= 1 - self.beta_1 ** (self.iteration + 1)
        abs_grad = np.abs(gradient)
        self.u = np.maximum(self.beta_2 * self.u, abs_grad)
        gradient = m_hat / (self.u + self.eps)
        return gradient
