"""Exponential Averaging."""

import numpy as np



class ExponentialAveraging:
    r"""Exponential averaging.

    :math:`x^{(0)}_{avg}=x^{(0)}`

    :math:`x^{(j)}_{avg}= \alpha x^{(j-1)}_{avg}+(1-\alpha)x^{(j)}`

    Is also sometimes referred to as exponential smoothing.

    Attributes:
        coefficient (float): Coefficient in (0,1) for the average.
        current_average (np.array): Current average value.
    """

    def __init__(self, coefficient):
        """Initialize exponential averaging object.

        Args:
            coefficient (float): Coefficient in (0,1) for the average
        """
        if coefficient < 0 or coefficient > 1:
            raise ValueError("Coefficient for exponential averaging needs to be in (0,1)")
        super().__init__()
        self.coefficient = coefficient
        self.current_average = None

    def update_average(self, new_value):
        """Compute the actual average.

        Args:
            new_value (np.array): New observation for the averaging

        Returns:
            Current average value
        """
        if isinstance(new_value, (float, int)):
            new_value = np.array(new_value)
        if self.current_average is not None:
            self.current_average = self.average_computation(new_value)
        else:
            # If it is the first observation
            self.current_average = new_value.copy()
        return self.current_average.copy()

    def average_computation(self, new_value):
        """Compute the exponential average.

        Args:
            new_value (float or np.ndarray): New value to update the average.

        Returns:
            current_average (np.ndarray): Returns the current average
        """
        current_average = (
            self.coefficient * self.current_average + (1 - self.coefficient) * new_value
        )
        return current_average