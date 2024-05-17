"""RMSprop optimizer."""

import numpy as np

from stochastic_optimizers.stochastic_optimizer import StochasticOptimizer
from stochastic_optimizers.exponential_averaging import ExponentialAveraging


class RMSprop(StochasticOptimizer):
    r"""RMSprop stochastic optimizer [1].

    References:
        [1] Tieleman and Hinton. "Lecture 6.5-rmsprop: Divide the gradient by a running average of
            its recent magnitude". Coursera. 2012.

    Attributes:
        beta (float):  :math:`\beta` parameter as described in [1].
        v (ExponentialAveragingObject): Exponential average of the gradient momentum.
        eps (float): Nugget term to avoid a division by values close to zero.
    """

    _name = "RMSprop Stochastic Optimizer"

    def __init__(
        self,
        learning_rate,
        optimization_type,
        beta=0.999,
        eps=1e-8,
    ):
        """Initialize optimizer.

        Args:
            learning_rate (float): Learning rate for the optimizer
            optimization_type (str): "max" in case of maximization and "min" for minimization
            beta (float): :math:`beta` parameter as described in [1]
            eps (float): Nugget term to avoid a division by values close to zero
        """
        super().__init__(
            learning_rate=learning_rate,
            optimization_type=optimization_type,
        )
        self.beta = beta
        self.v = ExponentialAveraging(coefficient=beta)
        self.eps = eps

    def scheme_specific_gradient(self, gradient):
        """Rmsprop gradient computation.

        Args:
            gradient (np.array): Gradient

        Returns:
            gradient (np.array): RMSprop gradient
        """
        if self.iteration == 0:
            self.v.current_average = np.zeros(gradient.shape)

        v_hat = self.v.update_average(gradient**2)
        v_hat /= 1 - self.beta ** (self.iteration + 1)
        gradient = gradient / (v_hat**0.5 + self.eps)
        return gradient
