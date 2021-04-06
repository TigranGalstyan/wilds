import torch.autograd

from algorithms.single_model_algorithm import SingleModelAlgorithm
from models.initializer import initialize_model
from models.domain_classifiers import initialize_domain_classifier


class ReverseGradLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lamb):
        ctx.alpha = lamb
        return x

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class DANN(SingleModelAlgorithm):
    """ Unsupervised Domain Adaptation by Backpropagation.

       Original paper:
           @inproceedings{ganin2015unsupervised,
             title={Unsupervised domain adaptation by backpropagation},
             author={Ganin, Yaroslav and Lempitsky, Victor},
             booktitle={International conference on machine learning},
             pages={1180--1189},
             year={2015},
             organization={PMLR}
           }
   """
    def __init__(self, config, d_out, grouper, loss,
            metric, n_train_steps):
        model = initialize_model(config, d_out).to(config.device)

        self.num_domains = grouper.cardinality.item()
        self.lamb = config.dann_lamb
        self.dc_name = config.dann_dc_name
        self.d_out = d_out

        # initialize domain classifier
        model.domain_classifier = initialize_domain_classifier(name=self.dc_name,
                                                               num_features=model.n_outputs,
                                                               num_domains=self.num_domains).to(config.device)

        # initialize module
        super().__init__(
            config=config,
            model=model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )

    def process_batch(self, batch):
        ret = super().process_batch(batch)
        features = ret['features']
        features = ReverseGradLayer.apply(features, self.lamb)
        ret['domain_pred'] = self.model.domain_classifier(features)
        return ret

    def objective(self, results):
        y_loss = self.loss.compute(results['y_pred'], results['y_true'], return_dict=False)
        domain_loss = self.loss.compute(results['domain_pred'], results['g'], return_dict=False)
        return y_loss + domain_loss
