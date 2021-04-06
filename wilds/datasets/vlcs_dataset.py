import os
import torch
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.all_metrics import Accuracy

class VLCSDataset(WILDSDataset):
    """
    VLCS Dataset from Domainbed
    """

    def __init__(self, root_dir='data', download=False, split_scheme='official'):
        self._dataset_name = 'vlcs'
        self._version = '1.0'
        # self._download_url = 'https://drive.google.com/uc?id=1skwblH1_okBwxWxmRsp9_qi15hyPpxg8'
        self._data_dir = self.initialize_data_dir(root_dir, download)
        self._resolution = (224, 224)

        # Read in metadata
        self._metadata_df = pd.read_csv(
            os.path.join(self._data_dir, 'metadata.csv'),
            index_col=0
        )

        # Get the y values
        self._label_map = {
            'bird': 0,
            'car': 1,
            'chair': 2,
            'dog': 3,
            'person': 4
        }
        self._label_array = self._metadata_df['label'].values
        self._y_array = torch.LongTensor([self._label_map[y] for y in self._label_array])
        self._y_size = 1
        self._n_classes = 5

        # Get filenames
        self._input_array = [
            f'{env}/{label}/{image}'
            for image, label, env in
            self._metadata_df.loc[:, ['image', 'label', 'env']].itertuples(index=False, name=None)]

        test_env = ''  #'VOC2007'
        val_env = 'VOC2007'

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

        env_map = {
            'SUN09': 0,
            'LabelMe': 1,
            'Caltech101': 2,
            'VOC2007': 3
        }
        env_names = self._metadata_df['env'].values
        envs = [env_map[name] for name in env_names]

        val_env_mask = (self._metadata_df['env'] == val_env)
        test_env_mask = (self._metadata_df['env'] == test_env)
        self._metadata_df.loc[val_env_mask, 'split'] = self.split_dict['val']
        self._metadata_df.loc[test_env_mask, 'split'] = self.split_dict['test']

        self._split_scheme = split_scheme
        if self._split_scheme != 'official':
            raise ValueError(f'Split scheme {self._split_scheme} not recognized')
        self._split_array = self._metadata_df['split'].values

        self._metadata_array = torch.stack(
            (torch.LongTensor(envs),
             self._y_array),
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
       x = Image.open(img_filename).convert('RGB').resize(self._resolution)
       return x

    def eval(self, y_pred, y_true, metadata):
        return self.standard_group_eval(
            self._metric,
            self._eval_grouper,
            y_pred, y_true, metadata)
