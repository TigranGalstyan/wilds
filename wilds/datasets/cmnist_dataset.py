import os
import torch
import pandas as pd
from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.all_metrics import Accuracy

class CMNISTDataset(WILDSDataset):
    """
    This is a modified version of the MNIST dataset.

    Supported `split_scheme`:
        'official'

    Input (x):
        28x28 colored images from MNIST dataset.

    Label (y):
        y is the digit in the image. on of [5, 6, 9].

    Metadata:
        Each image is colored by one of the generated colors.

    """

    def __init__(self, root_dir='data', download=False, split_scheme='official'):
        self._dataset_name = 'cmnist'
        self._version = '1.0'
        self._data_dir = self.initialize_data_dir(root_dir, download)
        self._original_resolution = (28, 28)

        # Read in metadata
        self._metadata_df = pd.read_csv(
            os.path.join(self._data_dir, 'metadata.csv'),
            index_col=0,
            # dtype={'patient': 'str'}
        )

        # Get the y values
        self._y_array = torch.LongTensor(self._metadata_df['digit'].values)
        self._y_array = (self._y_array == 6) + (self._y_array == 9) * 2
        self._y_size = 3
        self._n_classes = 3

        # Get filenames
        self._input_array = [
            f'images/env_{env}/digit_{digit}/{image}.pt'
            for image, digit, env in
            self._metadata_df.loc[:, ['image', 'digit', 'env']].itertuples(index=False, name=None)]

        # Extract splits
        # Note that the hospital numbering here is different from what's in the paper,
        # where to avoid confusing readers we used a 1-indexed scheme and just labeled the test hospital as 5.
        # Here, the numbers are 0-indexed.
        test_env = 4
        val_env = 3

        self._split_dict = {
            'train': 0,
            'id_val': 1,
            'test': 2,
            'val': 3
        }
        self._split_names = {
            'train': 'Train',
            'id_val': 'Validation (ID)',
            'test': 'Test',
            'val': 'Validation (OOD)',
        }
        envs = self._metadata_df['env'].values.astype('long')
        val_env_mask = (self._metadata_df['env'] == val_env)
        test_env_mask = (self._metadata_df['env'] == test_env)
        self._metadata_df.loc[val_env_mask, 'split'] = self.split_dict['val']
        self._metadata_df.loc[test_env_mask, 'split'] = self.split_dict['test']

        self._split_scheme = split_scheme
        if self._split_scheme != 'official':
            raise ValueError(f'Split scheme {self._split_scheme} not recognized')

        self._split_array = self._metadata_df['split'].values

        self._metadata_array = torch.stack(
            [torch.LongTensor(envs),
             self._y_array],
            dim=1)
        self._metadata_fields = ['env', 'y']

        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=['env'])

        self._metric = Accuracy()

        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
        """
        Returns x for a given idx.
        """
        img_filename = os.path.join(
            self.data_dir,
            self._input_array[idx])
        x = torch.load(img_filename)
        return x

    def eval(self, y_pred, y_true, metadata):
        return self.standard_group_eval(
            self._metric,
            self._eval_grouper,
            y_pred, y_true, metadata)
