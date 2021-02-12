import torch
from algorithms.single_model_algorithm import SingleModelAlgorithm
from models.initializer import initialize_model
from wilds.common.utils import conditional_hsic

class ERM_HSIC(SingleModelAlgorithm):
    def __init__(self, config, d_out, grouper, loss,
            metric, n_train_steps):
        model = initialize_model(config, d_out).to(config.device)

        # initialize module
        super().__init__(
            config=config,
            model=model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )

        self.num_domains = self.grouper.cardinality.item()
        self.beta = config.hsic_beta
        self.d_out = d_out

    def objective(self, results):
        c = results['g']
        z = results['features']
        y_true = results['y_true']

        c = torch.eye(self.num_domains, device=c.device)[c]
        y = torch.eye(self.d_out, device=y_true.device)[y_true]

        hsic = conditional_hsic(z, c, y, batch_size=None)

        return self.loss.compute(results['y_pred'], results['y_true'], return_dict=False) + self.beta * hsic
