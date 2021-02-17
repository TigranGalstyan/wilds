import os
import argparse
import pickle

import matplotlib.pyplot as plt
from glob import glob

G_labels = ['$e_{G0}$', '$e_{G1}$', '$e_{G2}$', '$e_{G3}$']
I_labels = ['$d_{I0}$', '$d_{I1}$', '$d_{I2}$']
edgecolor = ['#00500070', '#400090']

K_pal = [
            '#ff5622',
            '#ff9800',
            '#9d27b0',
            '#673ab7',
            '#3f51b5',
            '#2096f3',
            '#00bbd4',
            '#009688',
            '#8bc34a',
            '#cedd39',
            '#ffeb3b',
            '#ffc107',
            '#f34236',
            '#e81e63'
        ] * 2
alpha = (0.8, 0.6)


def save_plot(epoch_errors, x_lim, y_lim, filename):
    x = sorted(list(epoch_errors.keys()))

    fig, ax = plt.subplots(2,1, figsize=(4.2, 5))
    ax[0].stackplot(
        x,
        [epoch_errors[k]['EG0'] for k in x],
        [epoch_errors[k]['EG1'] for k in x],
        [epoch_errors[k]['EG2'] for k in x],
        [epoch_errors[k]['EG3'] for k in x],
        edgecolor= edgecolor[0],
        labels = G_labels,
        colors = K_pal[7:7+4],
        alpha = alpha[0]
    )
    ax[0].legend()
    ax[0].set_ylabel("Error")
    ax[1].stackplot(
        x,
        [epoch_errors[k]['EI0'] for k in x],
        [epoch_errors[k]['EI1'] for k in x],
        [epoch_errors[k]['EI2'] for k in x],
        edgecolor=edgecolor[1],
        labels = I_labels,
        colors = K_pal[2:5],
        alpha = alpha[1]
    )
    ax[1].legend()
    ax[1].set_ylabel("Distinguishability")
    for i in range(2):
        ax[i].set_xlabel('Epoch')
        ax[i].set_xlim(x_lim)
        ax[i].set_ylim(y_lim[i])
    plt.tight_layout()
    plt.savefig(filename)

def failures_to_errors(logistics, num_train_domains, num_val_domains):
    e = {}

    e['EG0'] = 1 - logistics['G0']
    e['EG1'] = 1 - logistics['G1']
    e['EG2'] = 1 - logistics['G2']
    e['EG3'] = 1 - logistics['G3']

    e['EG3'] -= e['EG2']
    e['EG2'] -= e['EG1']
    e['EG1'] -= e['EG0']

    e['EI0'] = logistics['I0'] - 1 / num_train_domains
    e['EI1'] = logistics['I1'] - 1 / num_train_domains  # + num_val_domains)
    e['EI2'] = logistics['I2'] - 1 / num_train_domains  # + num_val_domains)

    e['EI2'] -= e['EI1']
    e['EI1'] -= e['EI0']

    return e

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', required=True)
    parser.add_argument('--dataset', default='camelyon17')

    args = parser.parse_args()

    if args.dataset == 'camelyon17':
        num_train_domains = 3
        num_val_domains = 1
        x_lim = (0, 9)
        y_lim = [(0, 0.5), (0, 0.65)]
    elif args.dataset.startswith('cmnist'):
        num_train_domains = 3
        num_val_domains = 1
        x_lim = (0, 29)
        y_lim = [(0, 0.70), (0, 0.65)]
    else:
        raise ValueError('Incompatible dataset specified.')


    epoch_errors = {}
    logistics_files = sorted(glob(os.path.join(args.log_dir, 'tests_epoch_*.pkl')))
    for logistics_file in logistics_files:
        epoch = int(logistics_file.split('_')[-1].split('.')[0])

        with(open(logistics_file, "rb")) as f:
            logistics = pickle.load(f)

        epoch_errors[epoch] = failures_to_errors(logistics, num_train_domains, num_val_domains)

    save_plot(epoch_errors, x_lim, y_lim, os.path.join(args.log_dir, 'errors.pdf'))

if __name__=='__main__':
    main()
