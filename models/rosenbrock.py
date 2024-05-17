import numpy as np


class RosenbrockModel:
    def __init__(self):
        self.a = 0
        self.b = 10

    def evaluate(self, samples):
        x = samples[:, 0]
        y = samples[:, 1]
        potential = (self.a - x) ** 2 + self.b * (y - x ** 2) ** 2
        return -1 * potential

    def evaluate_and_gradient(self, samples):
        x = samples[:, 0]
        y = samples[:, 1]
        potential = (self.a - x) ** 2 + self.b * (y - x ** 2) ** 2

        dp_dx = -2 * (self.a - x) - 4 * x * self.b * (y - x ** 2)
        dp_dy = 2 * self.b * (y - x ** 2)
        grad_potential = np.vstack([dp_dx, dp_dy]).T

        return -1 * potential, - grad_potential

