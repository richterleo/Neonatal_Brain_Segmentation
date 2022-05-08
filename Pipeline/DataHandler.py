import os
import tempfile
import typing

from BaseTypes import NpDecoder, Collector
from pathlib import Path

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
    def __init__(self, root_dir, result_dir, hide_labels, val_test_split=0.8, db='dHCP'):
        super().__init__(root_dir, result_dir, db)
        self.hide_labels = hide_labels
        if hide_labels:
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

        with open(young_test_path, 'r') as f:
            young_test_ids = f.read()
        young_test_ids = young_test_ids.strip()
        young_test_id_list = young_test_ids.split(',')

        young_test_data_dict = [item for item in self.data_dict if item['meta_data']['id'] in young_test_id_list]

        return young_test_data_dict

if __name__ == '__main__':
    os.environ["MONAI_DATA_DIRECTORY"] = 'Pipeline'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    directory = os.environ.get("MONAI_DATA_DIRECTORY")

    root_dir = tempfile.mkdtemp() if directory is None else directory

    result_dir = Path(root_dir) / r'results/baseline_results1647892348'

    datacollector = TrainCollector(root_dir, result_dir, False)
    print(datacollector.data_dict)
