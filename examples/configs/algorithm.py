algorithm_defaults = {
    'ERM': {
        'train_loader': 'standard',
        'uniform_over_groups': False,
        'eval_loader': 'standard',
    },
    'ERM_HSIC': {
        'train_loader': 'standard',
        'uniform_over_groups': False,
        'eval_loader': 'standard',
        'hsic_beta': 1.0,
    },
    'ERM_SD': {
        'train_loader': 'standard',
        'uniform_over_groups': False,
        'eval_loader': 'standard',
        'sd_penalty_lamb': 0.01,
    },
    'ERM_HSIC_SD': {
        'train_loader': 'standard',
        'uniform_over_groups': False,
        'eval_loader': 'standard',
        'sd_penalty_lamb': 0.01,
        'hsic_beta': 1.0,
    },
    'ERM_HSIC_GradPenalty': {
        'train_loader': 'standard',
        'uniform_over_groups': False,
        'eval_loader': 'standard',
        'hsic_beta': 1.0,
        'grad_penalty_lamb': 1.0,
        'params_regex': '.*'
    },
    'groupDRO': {
        'train_loader': 'standard',
        'uniform_over_groups': True,
        'distinct_groups': True,
        'eval_loader': 'standard',
        'group_dro_step_size': 0.01,
    },
    'deepCORAL': {
        'train_loader': 'group',
        'uniform_over_groups': True,
        'distinct_groups': True,
        'eval_loader': 'standard',
        'coral_penalty_weight': 1.,
    },
    'IRM': {
        'train_loader': 'group',
        'uniform_over_groups': True,
        'distinct_groups': True,
        'eval_loader': 'standard',
        'irm_lambda': 100.,
        'irm_penalty_anneal_iters': 500,
    }
}
