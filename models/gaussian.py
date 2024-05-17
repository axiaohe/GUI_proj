import numpy as np


class GaussianModel:
    def __init__(self):
        covariance = np.array([[1.0, 0.2], [0.2, 0.1]])
        self.precision = np.linalg.inv(covariance)
        self.logpdf_const = -1 / 2 * (np.log(2.0 * np.pi) * 2 + np.linalg.slogdet(covariance)[1])
        self.mean = np.array([-1, 0.5])

    def evaluate(self, samples):
        dist = samples - self.mean.reshape(1, -1)
        logpdf = self.logpdf_const - 0.5 * (np.dot(dist, self.precision) * dist).sum(axis=1)
        return logpdf

    def evaluate_and_gradient(self, samples):
        logpdf = self.evaluate(samples)
        grad_logpdf = np.dot(self.mean.reshape(1, -1) - samples, self.precision)
        return logpdf, grad_logpdf

