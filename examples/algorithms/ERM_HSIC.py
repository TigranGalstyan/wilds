import re

import numpy as np
import torch
import torch.nn.functional as F
import torch.autograd

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


class ERM_HSIC_GradPenalty(SingleModelAlgorithm):
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
        self.beta = config.hsic_beta
        self.lamb = config.grad_penalty_lamb
        self.key_regex = '.*'  # TODO: config.key_regex
        self.d_out = d_out

    def objective(self, results):
        c = results['g']
        z = results['features']
        y_true = results['y_true']

        c = torch.eye(self.num_domains, device=c.device)[c]
        y = torch.eye(self.d_out, device=y_true.device)[y_true]

        hsic = conditional_hsic(z, c, y, batch_size=None)

        # compute the gradient penalty
        if self.is_training:
            example_losses = F.cross_entropy(input=results['y_pred'], target=results['y_true'],
                                             reduction='none')
            avg_gradients = [None] * self.num_domains
            for domain_idx in range(self.num_domains):
                mask = results['g'] == domain_idx
                if mask.sum() == 0:
                    continue
                domain_avg_loss = torch.mean(example_losses[mask])
                g = torch.autograd.grad(domain_avg_loss, list(self.parameters()),
                                        retain_graph=True, create_graph=True)
                avg_gradients[domain_idx] = g

            grad_penalty = 0.0
            param_names = dict(self.named_parameters()).keys()
            for k, name in enumerate(param_names):
                if re.match(self.key_regex, name) is None:
                    continue
                g = [avg_gradients[domain_idx][k] for domain_idx in range(self.num_domains)
                        if avg_gradients[domain_idx] is not None]
                actual_num_domains = len(g)
                g = torch.stack(g).reshape((actual_num_domains, -1))

                # compute mean of pairwise gradient similarities, i.e., mean_{i<j} ||g_i - g_j||^2
                term1 = actual_num_domains * torch.sum(g**2, dim=1).sum(dim=0)
                term2 = torch.sum(torch.sum(g, dim=0)**2, dim=0)
                grad_penalty += 2.0 / (actual_num_domains * (actual_num_domains - 1)) * (term1 - term2)
        else:
            grad_penalty = 0.0

        return self.loss.compute(results['y_pred'], results['y_true'], return_dict=False) + self.beta * hsic \
            + self.lamb * grad_penalty
