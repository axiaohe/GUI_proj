import numpy as np
import matplotlib.pyplot as plt

from models import GaussianModel, RezendeModel, RosenbrockModel, ELBOModel
from stochastic_optimizers import Adam, Adamax, RMSprop, SGD
from variational_distributions.normalizing_flow import (
    NormalizingFlowVariational, PlanarLayer, Tanh, LeakyRelu, FullRankNormalLayer,
    MeanFieldNormalLayer
)

# Data equivalent: GaussianModel, RezendeModel, RosenbrockModel
# Layer options: PlanarLayer Tanh, PlanarLayer LeakyRelu, FullRankNormalLayer, MeanFieldNormalLayer
# Optimizer options: Adam, Adamax, RMSprop, SGD



def main():
    np.random.seed(2)
    num_dim = 2

    layers = [
        MeanFieldNormalLayer(num_dim),
        PlanarLayer(num_dim, Tanh()),
        FullRankNormalLayer(num_dim),
        PlanarLayer(num_dim, LeakyRelu()),
    ]

    layers += [PlanarLayer(num_dim, Tanh()) for _ in range(12)]
    # layers = [PlanarLayer(num_dim, LeakyRelu()) for _ in range(12)]
    # layers = [FullRankNormalLayer(num_dim)]

    variational_distribution = NormalizingFlowVariational(layers=layers)

    model = GaussianModel()
    elbo_model = ELBOModel(
        model=model,
        variational_distribution=variational_distribution,
        n_samples_per_iter=8   # batch size
    )

    optimizer = Adam(
        learning_rate=1e-3,
        optimization_type='max'   # this is fixed
    )

    history = {'variational_parameters': [], 'elbo': []}

    variational_parameters = variational_distribution.initialize_variational_parameters()

    max_iter = 100_000
    for i in range(max_iter):
        elbo, elbo_gradient = elbo_model.evaluate_and_gradient(variational_parameters)
        history['variational_parameters'].append(variational_parameters.copy())
        history['elbo'].append(elbo.copy())

        variational_parameters = optimizer.step(variational_parameters, elbo_gradient)

        if i % 100 == 0:
            print('Iteration', i)

            ################################################################################
            #################################   Plotting   #################################
            ################################################################################

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

            xlin = np.linspace(-2, 2, 100)
            ylin = np.linspace(-2, 3, 100)
            X, Y = np.meshgrid(xlin, ylin)
            positions = np.vstack([X.ravel(), Y.ravel()]).T

            true_pdf = np.exp(model.evaluate(positions))

            levels = 10

            try:
                variational_pdf = np.exp(variational_distribution.logpdf(variational_parameters,
                                                                         positions))
                ax1.tricontourf(positions[:, 0], positions[:, 1], variational_pdf, levels=levels)
                ax1.tricontour(positions[:, 0], positions[:, 1], variational_pdf, colors='w',
                               linewidths=1, levels=levels)
            except ValueError:
                samples = variational_distribution.draw(variational_parameters, n_draws=100_000)
                ax1.hist2d(samples[:, 0].flatten(), samples[:, 1].flatten(), (100, 100),
                           range=[[-2, 2], [-2, 3]])

            ax2.tricontourf(positions[:, 0], positions[:, 1], true_pdf, levels=levels)
            ax2.tricontour(positions[:, 0], positions[:, 1], true_pdf, colors='w', linewidths=1,
                           levels=levels)

            plt.show()


if __name__ == "__main__":
    main()