"""Stochastic optimizer."""
import abc


class StochasticOptimizer(metaclass=abc.ABCMeta):
    """Base class for stochastic optimizers.

    The optimizers are implemented as generators. This increases the modularity of this class,
    since an object can be used in different settings.

    Attributes:
        learning_rate (float): Learning rate for the optimizer.
        precoefficient (int): Is 1 in case of maximization and -1 for minimization.
        iteration (int): Number of iterations done in the optimization so far.
    """

    _name = "Stochastic Optimizer"

    def __init__(
        self,
        learning_rate,
        optimization_type,
    ):
        """Initialize stochastic optimizer.

        Args:
            learning_rate (float): Learning rate for the optimizer
            optimization_type (str): "max" in case of maximization and "min" for minimization
        """
        self.precoefficient = {"min": -1, "max": 1}[optimization_type]
        self.learning_rate = learning_rate
        self.iteration = 0

    @abc.abstractmethod
    def scheme_specific_gradient(self, gradient):
        """Scheme specific gradient computation.

        Here the gradient is transformed according to the desired stochastic optimization approach.

        Args:
            gradient (np.array): Current gradient
        """


    def step(self, variational_parameters, gradient):
        """Optimization step.

        Args:
            variational_parameters (np.array): Current variational parameters
            gradient (np.array): Current gradient

        Returns:
            variational_parameters (np.array): Updated variational parameters
        """
        gradient = self.scheme_specific_gradient(gradient)
        variational_parameters += self.precoefficient * self.learning_rate * gradient
        self.iteration += 1
        return variational_parameters