import os
from tqdm import tqdm
import torch
from utils import save
import torch.autograd.profiler as profiler
from wilds.common.utils import conditional_hsic

def log_results(algorithm, dataset, general_logger, epoch, batch_idx):
    if algorithm.has_log:
        log = algorithm.get_log()
        log['epoch'] = epoch
        log['batch'] = batch_idx
        dataset['algo_logger'].log(log)
        if dataset['verbose']:
            general_logger.write(algorithm.get_pretty_log_str())
        algorithm.reset_log()

def run_epoch(algorithm, dataset, general_logger, epoch, config, train):
    if dataset['verbose']:
        general_logger.write(f"\n{dataset['name']}:\n")

    if train:
        algorithm.train()
    else:
        algorithm.eval()
    # process = psutil.Process(os.getpid())

    # process = psutil.Process(os.getpid())

    # Not preallocating memory is slower
    # but makes it easier to handle different types of data loaders
    # (which might not return exactly the same number of examples per epoch)
    epoch_y_true = []
    epoch_y_pred = []
    epoch_metadata = []

    # Using enumerate(iterator) can sometimes leak memory in some environments (!)
    # so we manually increment batch_idx
    batch_idx = 0
    iterator = tqdm(dataset['loader']) if config.progress_bar else dataset['loader']

    for batch in iterator:
        if train:
            batch_results = algorithm.update(batch)
        else:
            batch_results = algorithm.evaluate(batch)

        # These tensors are already detached, but we need to clone them again
        # Otherwise they don't get garbage collected properly in some versions
        # The subsequent detach is just for safety
        # (they should already be detached in batch_results)
        epoch_y_true.append(batch_results['y_true'].clone().detach())
        epoch_y_pred.append(batch_results['y_pred'].clone().detach())
        epoch_metadata.append(batch_results['metadata'].clone().detach())

        if train and (batch_idx+1) % config.log_every==0:
            log_results(algorithm, dataset, general_logger, epoch, batch_idx)
            # mem = process.memory_info().rss
            # print(f'Mem: {mem / 1024 / 1024:6.1f}M')

        batch_idx += 1

    results, results_str = dataset['dataset'].eval(
        torch.cat(epoch_y_pred),
        torch.cat(epoch_y_true),
        torch.cat(epoch_metadata))

    if config.scheduler_metric_split==dataset['split']:
        algorithm.step_schedulers(
            is_epoch=True,
            metrics=results,
            log_access=(not train))

    # log after updating the scheduler in case it needs to access the internal logs
    log_results(algorithm, dataset, general_logger, epoch, batch_idx)

    results['epoch'] = epoch
    dataset['eval_logger'].log(results)
    if dataset['verbose']:
        general_logger.write('Epoch eval:\n')
        general_logger.write(results_str)

    return results


def train(algorithm, datasets, general_logger, config, epoch_offset, best_val_metric):
    for epoch in range(epoch_offset, config.n_epochs):
        general_logger.write('\nEpoch [%d]:\n' % epoch)

        # First run training
        run_epoch(algorithm, datasets['train'], general_logger, epoch, config, train=True)

        # Then run val
        val_results = run_epoch(algorithm, datasets['val'], general_logger, epoch, config, train=False)
        curr_val_metric = val_results[config.val_metric]
        general_logger.write(f'Validation {config.val_metric}: {curr_val_metric:.3f}\n')

        # Then run everything else
        if config.evaluate_all_splits:
            additional_splits = [split for split in datasets.keys() if split not in ['train','val']]
        else:
            additional_splits = config.eval_splits
        for split in additional_splits:
            run_epoch(algorithm, datasets[split], general_logger, epoch, config, train=False)

        if best_val_metric is None:
            is_best = True
        else:
            if config.val_metric_decreasing:
                is_best = curr_val_metric < best_val_metric
            else:
                is_best = curr_val_metric > best_val_metric
        if is_best:
            best_val_metric = curr_val_metric

        if config.save_step is not None and (epoch + 1) % config.save_step == 0:
            save(algorithm, epoch, best_val_metric, os.path.join(config.log_dir, '%d_model.pth' % epoch))
        if config.save_last:
            save(algorithm, epoch, best_val_metric, os.path.join(config.log_dir, 'last_model.pth'))
        if config.save_best and is_best:
            save(algorithm, epoch, best_val_metric, os.path.join(config.log_dir, 'best_model.pth'))
            general_logger.write(f'Best model saved at epoch {epoch}\n')

        general_logger.write('\n')


def evaluate(algorithm, datasets, epoch, general_logger, config):
    algorithm.eval()
    overall_results = {}
    z_splits = {}
    y_splits = {}
    c_splits = {}
    for split, dataset in list(datasets.items()):
        if (not config.evaluate_all_splits) and (split not in config.eval_splits):
            continue
        epoch_y_true = []
        epoch_y_pred = []
        epoch_z = []
        epoch_metadata = []
        iterator = tqdm(dataset['loader']) if config.progress_bar else dataset['loader']
        for batch in iterator:
            batch_results = algorithm.evaluate(batch)
            epoch_y_true.append(batch_results['y_true'].clone().detach())
            epoch_y_pred.append(batch_results['y_pred'].clone().detach())
            epoch_z.append(batch_results['features'].clone().detach())
            epoch_metadata.append(batch_results['metadata'].clone().detach())

        epoch_y_pred, epoch_y_true, epoch_z, epoch_metadata =\
            torch.cat(epoch_y_pred), torch.cat(epoch_y_true), torch.cat(epoch_z), torch.cat(epoch_metadata)

        results, results_str = dataset['dataset'].eval(
            epoch_y_pred,
            epoch_y_true,
            epoch_metadata)

        if config.dataset in ['camelyon17', 'cmnist', 'cmnist4', 'cmnist7', 'cmnist28', 'cmnist10', 'vlcs']:
            num_envs = 4 if config.dataset == 'vlcs' else 5
            if config.dataset == 'camelyon17':
                metadata = torch.tensor(epoch_metadata / 10, dtype=torch.long) # slides => hospitals
            else:
                metadata = epoch_metadata
            c = dataset['dataset'].dataset._eval_grouper.metadata_to_group(metadata)
            c_splits[split] = c
            c = torch.eye(num_envs, device=c.device)[c]
            if config.dataset == 'camelyon17':
                n_classes = 2
            elif config.dataset == 'cmnist10':
                n_classes = 10
            elif config.dataset == 'vlcs':
                n_classes = 5
            else:
                n_classes = 3
            y = torch.eye(n_classes, device=epoch_y_true.device)[epoch_y_true]
            z_splits[split] = epoch_z
            y_splits[split] = y
            if split == 'train':
                hsic_mean, hsic_std = conditional_hsic(epoch_z, c, y)
                results['hsic_std'] = hsic_std
                results['hsic_mean'] = hsic_mean
                results_str += 'HSIC std_mean: {:.4f} {:.4f}\n'.format(hsic_mean, hsic_std)
            else:
                results['hsic_std'] = 0
                results['hsic_mean'] = 0

        results['epoch'] = epoch
        dataset['eval_logger'].log(results)
        general_logger.write(f'Eval split {split} at epoch {epoch}:\n')
        general_logger.write(results_str)
        overall_results[split] = results

    if config.dataset in ['camelyon17', 'cmnist', 'cmnist4', 'cmnist7', 'cmnist28', 'vlcs', 'cmnist10']:

        if config.save_z:
            torch.save(z_splits, os.path.join(config.log_dir, f'z_splits_epoch_{epoch}.pt'))
            torch.save(y_splits, os.path.join(config.log_dir, f'y_splits_epoch_{epoch}.pt'))
            torch.save(c_splits, os.path.join(config.log_dir, f'c_splits_epoch_{epoch}.pt'))

        # z = z_splits['train'][c_splits['train']!=0]
        # y = y_splits['train'][c_splits['train']!=0]
        # c = c_splits['train'][c_splits['train']!=0]
        #
        # num_envs = 4 if config.dataset == 'vlcs' else 5
        # c = torch.eye(num_envs, device=c.device)[c]

        # if (not config.evaluate_all_splits) and ('val'  in config.eval_splits):
        #     c_val = torch.eye(num_envs, device=c.device)[c_splits['val']]
        #     hsic_val_mean, hsic_val_std = conditional_hsic(torch.cat([z, z_splits['val']]),
        #                                                    torch.cat([y, y_splits['val']]), torch.cat([c, c_val]))
        #     general_logger.write("Hsic between hospitals {}: {:.4f} {:.4f}\n".format(
        #         '1 3 4' if config.dataset == 'camelyon17' else '1 2 3', hsic_val_mean, hsic_val_std))
        #     datasets['val']['eval_logger'].log({'hsic_mean': hsic_val_mean, 'hsic_std': hsic_val_std})
        #     overall_results['val']['hsic_mean'] = hsic_val_mean
        #     overall_results['val']['hsic_std'] = hsic_val_std
        #
        #
        # if (not config.evaluate_all_splits) and ('test'  in config.eval_splits):
        #     c_test = torch.eye(num_envs, device=c.device)[c_splits['test']]
        #     hsic_test_mean, hsic_test_std = conditional_hsic(torch.cat([z, z_splits['test']]),
        #                                                      torch.cat([y, y_splits['test']]), torch.cat([c, c_test]))
        #     general_logger.write("Hsic between hospitals {}: {:.4f} {:.4f}\n".format(
        #         '2 3 4' if config.dataset == 'camelyon17' else '1 2 4', hsic_test_mean, hsic_test_std))
        #     datasets['test']['eval_logger'].log({'hsic_mean': hsic_test_mean, 'hsic_std': hsic_test_std})
        #     overall_results['test']['hsic_mean'] = hsic_test_mean
        #     overall_results['test']['hsic_std'] = hsic_test_std

    return overall_results, z_splits, y_splits, c_splits
