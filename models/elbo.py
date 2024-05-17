import numpy as np


class ELBOModel:
    def __init__(self, model, variational_distribution, n_samples_per_iter):
        """Initialize ELBO model.

        Args:
            model (Model): log joint probability model
            variational_distribution (obj): Variational distribution
            n_samples_per_iter (int): Batch size per iteration
        """
        self.model = model
        self.variational_distribution = variational_distribution
        self.n_samples_per_iter = n_samples_per_iter

    def evaluate_and_gradient(self, variational_parameters):
        """Compute ELBO and ELBO gradient with reparameterization trick.

        Args:
            variational_parameters (np.ndarray): variational parameters

        Returns:
            elbo (np.ndarray): ELBO
            elbo_gradient (np.ndarray): ELBO gradient
        """
        # update variational_params
        variational_params = variational_parameters.reshape(-1)

        (
            sample_batch,
            standard_normal_sample_batch,
        ) = self.variational_distribution.conduct_reparameterization(
            variational_params, self.n_samples_per_iter
        )

        log_prob_joint, grad_log_prob_joint = self.model.evaluate_and_gradient(sample_batch)

        total_grad_variational_batch = self.variational_distribution.total_grad_params_logpdf(
            variational_params, standard_normal_sample_batch
        )

        sample_elbo_grad = self.variational_distribution.grad_params_reparameterization(
            variational_params,
            standard_normal_sample_batch,
            upstream_gradient=grad_log_prob_joint
        ) - total_grad_variational_batch

        # MC estimate of elbo gradient
        grad_elbo = np.mean(sample_elbo_grad, axis=0)

        logpdf_variational = self.variational_distribution.forward(
            variational_params, standard_normal_sample_batch)[1]
        elbo = np.mean(log_prob_joint - logpdf_variational)
        return elbo, grad_elbo