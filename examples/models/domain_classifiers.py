import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, num_features, num_domains, hidden_sizes):
        super(MLP, self).__init__()
        self.num_features = num_features
        self.num_domains = num_domains
        self.hidden_sizes = hidden_sizes

        layers = []
        prev_size = num_features
        for dim in hidden_sizes:
            layers.append(torch.nn.Linear(prev_size, dim))
            layers.append(torch.nn.ReLU(inplace=True))
            prev_size = dim
        layers.append(torch.nn.Linear(prev_size, num_domains))
        self.classifier = torch.nn.Sequential(tuple(layers))

    def forward(self, x):
        return self.classifier(x)


_container = {}

def register_parser(_container, parser_name):
    def decorator(parser_fn):
        _container[parser_name] = parser_fn

        def wrapper(*args, **kwargs):
            return parser_fn(*args, **kwargs)

        return wrapper

    return decorator


@register_parser(_container, 'mnist_simple_1hidden')
def mnist_simple_1hidden(**kwargs):
    num_domains = kwargs.get('num_domains', 4)
    return MLP(num_features=180, num_domains=num_domains, hidden_sizes=[128])


def initialize_domain_classifier(name, **kwargs):
    if name not in _container:
        return ValueError(f"No domain classifier is found with name '{name}'")
    return _container[name](**kwargs)
