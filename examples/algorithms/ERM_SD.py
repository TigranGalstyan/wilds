from algorithms.single_model_algorithm import SingleModelAlgorithm
from models.initializer import initialize_model


class ERM_SD(SingleModelAlgorithm):
    def __init__(self, config, d_out, grouper, loss, metric, n_train_steps):
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
        self.lamb = config.sd_penalty_lamb

    def objective(self, results):
        y_true = results['y_true']
        y_pred = results['y_pred']

        penalty = (y_pred ** 2).mean()

        return self.loss.compute(y_pred, y_true, return_dict=False) + self.lamb * penalty
