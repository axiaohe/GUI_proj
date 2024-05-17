"""SGD optimizer."""

from stochastic_optimizers.stochastic_optimizer import StochasticOptimizer


class SGD(StochasticOptimizer):
    """Stochastic gradient descent optimizer."""

    _name = "SGD Stochastic Optimizer"

    def __init__(
        self,
        learning_rate,
        optimization_type,
    ):
        """Initialize optimizer.

        Args:
            learning_rate (float): Learning rate for the optimizer
            optimization_type (str): "max" in case of maximization and "min" for minimization
        """
        super().__init__(
            learning_rate=learning_rate,
            optimization_type=optimization_type,
        )

    def scheme_specific_gradient(self, gradient):
        """SGD gradient computation.

        Args:
            gradient (np.array): Gradient

        Returns:
            gradient (np.array): SGD gradient
        """
        return gradient
