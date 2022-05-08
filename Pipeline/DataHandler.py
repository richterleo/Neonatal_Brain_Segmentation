import csv
import os
import random
import tempfile
import typing

from BaseTypes import NpDecoder, Collector
from copy import deepcopy
from Hyperparams import label_dispersion_factor
from pathlib import Path
from monai.data import DataLoader, Dataset, CacheNTransDataset
from monai.data.utils import pad_list_data_collate
from Transforms import create_train_val_transform, create_test_transform

class TrainCollector(Collector):
    '''Class for loading data and creating train/val/test and young/old split.

    Args:
        root_dir (Union[str, pathlib.Path]): root directory
        db_path (pathlib.Path): path to database
        result_dir (pathlib.Path): path to results folder
        t1_dir (pathlib.Path): path to T1w images in database
        t2_dir (pathlib.Path): path to T2w images in database
        label_dir (pathlib.Path): path to labels in database
        cache_dir (pathlib.Path): path to cache folder for hiding labels
        t1_list (List): list of paths to T1w images
        t2_list (List): list of paths to T2w images
        label_list (List): list of paths to labels
        meta_data_list (List): list of paths to meta_data
        meta_data_list_dicts (List): list of meta data dicts
        data_dict (List): list of dicts containing T1w, T2w, label and meta data per sample
        val_test_split (float): proportion of images in train set. rest of samples is equally distributed over val and test set
    '''
    def __init__(self, root_dir, result_dir, mode, val_test_split=0.8, db='dHCP'):
        super().__init__(root_dir, result_dir, db)
        self.mode = mode
        if mode == 'labelBudgeting':
            # define dir to cache transformed samples
            self.cache_dir = self.result_dir / 'cache_dir'
        self.val_test_split = val_test_split

    def create_sets(self):
        '''Creates train val test datasets from presaved subject ids.
        
        Returns:
            train_dict (List): list of dicts for train samples
            val_dict (List): list of dicts for val samples  
        '''
        
        id_path = self.root_dir / 'IDs'
        train_path = id_path / 'train_ids.csv'
        val_path = id_path / 'val_ids.csv'
        test_path = id_path / 'test_ids.csv'  

        if not (train_path.is_file() and val_path.is_file() and test_path.is_file()):
            self.create_train_val_test_ids()

        with open(train_path, 'r') as f:
            train_ids = f.read()
        train_ids = train_ids.strip()
        train_id_list = train_ids.split(',')

        train_data_dict = [item for item in self.data_dict if item['meta_data']['id'] in train_id_list]

        with open(val_path, 'r') as f:
            val_ids = f.read()
        val_ids = val_ids.strip()
        val_id_list = val_ids.split(',')

        val_data_dict = [item for item in self.data_dict if item['meta_data']['id'] in val_id_list]

        return train_data_dict, val_data_dict

    def create_train_val_test_ids(self):
        '''Creates fixed and random train, val, test indices and saves them down.'''

        assert self.mode in ['baseline', 'agePrediction', 'labelBudgeting'], 'method cannot be used in mode <<transfer>>'

        id_list = [item['meta_data']['id'] for item in self.data_dict]
        id_list_shuffle = random.shuffle(id_list)

        # balance out total number of slices by decreasing training samples in baseline conditions
        if self.mode == 'baseline' or self.mode == 'agePrediction':
            num_samples = int(label_dispersion_factor * len(id_list)) # default value: 30%
        elif self.mode == 'labelBudgeting':
            num_samples = len(id_list)

        train_stop = int(num_samples*self.val_test_split)
        val_stop = int(num_samples*((1-self.val_test_split)/2))

        train_ids = id_list_shuffle[:train_stop]
        val_ids = id_list_shuffle[train_stop:val_stop]
        test_ids = id_list_shuffle[val_stop:num_samples]

        id_path = self.root_dir / 'IDs'
        train_path = id_path / 'train_ids.csv'
        val_path = id_path / 'val_ids.csv'
        test_path = id_path / 'test_ids.csv' 

        Path(id_path).mkdir(parents=True, exist_ok=True)

        with open(train_path,'w') as train_file:
            wr = csv.writer(train_file, dialect='excel')
            wr.writerow(train_ids)

        with open(val_path,'w') as val_file:
            wr = csv.writer(val_file, dialect='excel')
            wr.writerow(val_ids)

        with open(test_path,'w') as test_file:
            wr = csv.writer(test_file, dialect='excel')
            wr.writerow(test_ids)

    def get_loaders(self, pixdim, roi_size, batch_size, slicing_mode = None, selection_mode=None):
        if self.mode == 'transfer':
            train_dict = self.create_old()

            train_transform, _ = create_train_val_transform(pixdim, roi_size, False)
            train_ds = Dataset(train_dict, transform=train_transform)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

            return train_loader
        
        elif self.mode == 'labelBudgeting':
            train_dict, val_dict = self.create_sets()

            train_transform, val_transform = create_train_val_transform(pixdim, roi_size, True, 
                                            slicing_mode=slicing_mode, selection_mode=selection_mode) # inserts label hide transform
            train_ds = CacheNTransDataset(train_dict, transform=train_transform, cache_n_trans=5, cache_dir=self.cache_dir)
            val_ds = Dataset(val_dict, transform=val_transform)

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=pad_list_data_collate)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, collate_fn=pad_list_data_collate)

            return train_loader, val_loader
        
        else:
            train_dict, val_dict = self.create_sets()

            train_transform, val_transform = create_train_val_transform(pixdim, roi_size, False)            
            train_ds = Dataset(train_dict, transform=train_transform)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

            val_ds = Dataset(val_dict, transform=val_transform)
            val_loader = DataLoader(val_ds, batch_Size=batch_size, shuffle=True)

            return train_loader, val_loader

    def create_old(self):
        '''Creates dataset made up of old subjects for transfer learning.
        
        Returns:
            list of sample dicts from old subjects
        '''

        id_path = self.root_dir / 'IDs'
        old_path = id_path / 'old_ids.csv'
        young_train_path = id_path / 'young_train_ids.csv'
        young_test_path = id_path / 'young_test_ids.csv'  

        # check whether all files exist, else we have to recreate everything
        if not (old_path.is_file() and young_train_path.is_file() and young_test_path.is_file()):
            self.create_old_young_ids()

        with open(old_path, 'r') as f:
            old_ids = f.read()
        old_ids = old_ids.strip()
        old_id_list = old_ids.split(',')

        old_data_dict = [item for item in self.data_dict if item['meta_data']['id'] in old_id_list]

        return old_data_dict

    def create_young(self):
        '''Creates dataset made up of young subjects for transfer learning.
        
        Returns:
            list of sample dicts from young subjects
        '''
        
        id_path = self.root_dir / 'IDs'
        old_path = id_path / 'old_ids.csv'
        young_train_path = id_path / 'young_train_ids.csv'
        young_test_path = id_path / 'young_test_ids.csv'  

        # check whether all files exist, else we have to recreate everything
        if not (old_path.is_file() and young_train_path.is_file() and young_test_path.is_file()):
            self.create_old_young_ids()

        with open(young_train_path, 'r') as f:
            young_train_ids = f.read()
        young_train_ids = young_train_ids.strip()
        young_train_id_list = young_train_ids.split(',')

        young_train_data_dict = [item for item in self.data_dict if item['meta_data']['id'] in young_train_id_list]

        return young_train_data_dict


class TestCollector(Collector):

    def __init__(self, root_dir, result_dir, db='dHCP'):
        super().__init__(root_dir, result_dir, db)

    def create_sets(self):
        '''Creates test set from presaved subject ids.
        
        Returns:
            list of test sample dicts 
        '''
        
        id_path = self.root_dir / 'IDs'
        train_path = id_path / 'train_ids.csv'
        val_path = id_path / 'val_ids.csv'
        test_path = id_path / 'test_ids.csv'  

        if not (train_path.is_file() and val_path.is_file() and test_path.is_file()):
            self.create_train_val_test_ids()

        with open(test_path, 'r') as f:
            test_ids = f.read()
        test_ids = test_ids.strip()
        test_id_list = test_ids.split(',')

        test_data_dict = [item for item in self.data_dict if item['meta_data']['id'] in test_id_list]
        
        return test_data_dict

    def create_young(self):
        '''Creates test set for transfer learning from presaved subject ids.
        
        Returns:
            list of test sample for young subjects dicts 
        '''
        id_path = self.root_dir / 'IDs'
        old_path = id_path / 'old_ids.csv'
        young_train_path = id_path / 'young_train_ids.csv'
        young_test_path = id_path / 'young_test_ids.csv'  

        # check whether all files exist, else we have to recreate everything
        if not (old_path.is_file() and young_train_path.is_file() and young_test_path.is_file()):
            self.create_old_young_ids()

        with open(young_train_path, 'r') as f:
            young_train_ids = f.read()
        young_train_ids = young_train_ids.strip()
        young_train_id_list = young_train_ids.split(',')

        young_train_data_dict = [item for item in self.data_dict if item['meta_data']['id'] in young_train_id_list]

        with open(young_test_path, 'r') as f:
            young_test_ids = f.read()
        young_test_ids = young_test_ids.strip()
        young_test_id_list = young_test_ids.split(',')

        young_test_data_dict = [item for item in self.data_dict if item['meta_data']['id'] in young_test_id_list]

        return young_train_data_dict, young_test_data_dict


    def get_loaders(self, pixdim, roi_size, batch_size):
        
        if self.mode == 'transfer':
            train_dict, test_dict = self.create_young()

            train_transform, _ = create_train_val_transform(pixdim, roi_size, False) # no labelBudgeting
            test_transform = create_test_transform(pixdim)
            train_ds = Dataset(train_dict, transform=train_transform)
            test_ds = Dataset(test_dict, transform=test_transform)

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=1, shuffle=False) # test loader always gets batch_size = 1

            return train_loader, test_loader
        
        else:
            test_dict = self.create_sets()
            test_transform = create_test_transform(pixdim)

            test_ds = Dataset(test_dict, transform=test_transform)
            test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

            return test_loader

if __name__ == '__main__':
    os.environ["MONAI_DATA_DIRECTORY"] = 'Pipeline'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    directory = os.environ.get("MONAI_DATA_DIRECTORY")

    root_dir = tempfile.mkdtemp() if directory is None else directory

    result_dir = Path(root_dir) / r'results/baseline_results1647892348'

    datacollector = TrainCollector(root_dir, result_dir, False)
    print(datacollector.data_dict)
