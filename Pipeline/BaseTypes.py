import csv
import json
import numpy as np
import os
import random
import tempfile
import time

from abc import ABC, abstractmethod
from AgeTools import AgeDynUNet
from datetime import datetime
from Hyperparams import modes
from monai.networks.nets import DynUNet 
from pathlib import Path


class ModeIncompatibleError(Exception):
    '''Raised when incompatible mode is chosen

    Attributes:
        mode (str): training mode
        message (str): error message
    '''

    def __init__(self, mode, message= "Invalid mode was selected."):
        self.mode = mode
        self.message = message
        super().__init__(self.message)

class MissingModeError(Exception):
    '''Raised when slicing_mode or selection_mode is missing

    Attributes:

        message (str): error message
    '''

    def __init__(self,  message= "slicing_mode and selection_mode must be given"):
        self.message = message
        super().__init__(self.message)


class Logger(ABC):
    '''Abstract base class for ResultLogger and InferenceLogger'''

    def __init__(self, mode, session_info='', monai_data_dir='Pipeline', random_seed = 0):

        super().__init__()
        self.check_valid_mode(mode, modes)
        self.mode = mode
        self.root_dir = Path(self.get_root_dir(monai_data_dir))
        self.start = time.time()
        self.meta_info = {"session_date": str(datetime.today()), "mode": self.mode,
                            "session_info": session_info, "random_seed": random_seed}
        self.end = None

    def get_root_dir(self, monai_data_dir):
        '''Sets environment variables'''

        os.environ["MONAI_DATA_DIRECTORY"] = monai_data_dir
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        directory = os.environ.get("MONAI_DATA_DIRECTORY")
        return tempfile.mkdtemp() if directory is None else directory

    @abstractmethod
    def create_result_folder(self):
        '''Creates unique results folder'''
        pass

    @abstractmethod
    def populate_hyperparams(self):
        pass

    @abstractmethod
    def populate_results(self):
        pass

    @staticmethod
    def check_valid_mode(mode, modes):
        '''Checks if mode is valid'''
        if mode not in modes:
                raise ModeIncompatibleError(mode)

    @abstractmethod
    def log_tcs(self):
        pass

    @abstractmethod
    def save_info(self):
        pass

    def define_model(self, kernels, strides):
        if self.mode == 'agePrediction':
            model = AgeDynUNet(
                        spatial_dims=3,
                        in_channels=2,
                        out_channels=10,
                        kernel_size=kernels,
                        strides=strides,
                        upsample_kernel_size=strides[1:],
                        deep_supervision=True,
                        deep_supr_num = 3,
                        res_block=True)
        else:
            model  = DynUNet(
                    spatial_dims=3,
                    in_channels=2,
                    out_channels=10,
                    kernel_size=kernels,
                    strides=strides, #2
                    upsample_kernel_size=strides[1:],
                    deep_supervision=True,
                    deep_supr_num = 3,
                    res_block=True)

        return model

    def restart_clock(self):
        '''Restarts clock when training starts'''
        self.start = time.time()
    
    def stop_clock(self):
        '''Stores time from class initialization to current process.'''

        t1 = time.time()
        self.end = str(round(t1))
        self.meta_info["duration"] = str(t1-self.start)


class NpDecoder(json.JSONDecoder):
    '''Decoder class for handling meta data.'''

    def default(self, obj):
        if isinstance(obj, int):
            return np.integer(obj)
        elif isinstance(obj, float):
            return np.floating(obj)
        elif isinstance(obj, list):
            return np.ndarray(obj)
        elif isinstance(obj, bool):
            return np.bool_(obj)
        else:
            return super(NpDecoder, self).default(obj)


class Collector(ABC):

    def __init__(self, root_dir, result_dir, db='dHCP'):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.db_path = self.root_dir / db # Pipeline/dHCP
        self.result_dir = Path(result_dir)
        self.t1_dir = self.db_path / 'T1w'
        self.t2_dir = self.db_path / 'T2w'
        self.label_dir = self.db_path / 'labels'
        self.meta_data_dir = self.db_path / 'meta_data'

        self.t1_list = sorted([os.path.join(self.t1_dir, f) for f in os.listdir(self.t1_dir)])
        self.t2_list = sorted([os.path.join(self.t2_dir, f) for f in os.listdir(self.t2_dir)])
        self.label_list = sorted([os.path.join(self.label_dir, f) for f in os.listdir(self.label_dir)])
        self.meta_data_list = sorted([os.path.join(self.meta_data_dir, f) for f in os.listdir(self.meta_data_dir)])
        assert len(self.t1_list) == len(self.t2_list) == len(self.label_list), "Number of train and label files does not align"

        self.meta_data_list_dicts = self.read_meta_data()
        self.data_dict = [{"t1_image": t1_img, "t2_image": t2_img, "label": label, "meta_data": meta_data} 
                        for t1_img, t2_img, label, meta_data in zip(self.t1_list, self.t2_list, self.label_list, self.meta_data_list_dicts)]

    def read_meta_data(self):
        '''Read json encoded meta data into dicts.
        
        Returns:
            dict of meta data per subject and session     
        '''

        meta_data_list_dicts = []
        for meta_file in self.meta_data_list:
            try: 
                with open(meta_file, 'r') as f:
                    file_name = meta_file.split('/')[-1]
                    id = file_name.split('_meta_data.json')[0]
                    meta_data = json.load(f)
                    meta_data["id"] = id
                    meta_data_list_dicts.append(meta_data)
            except json.JSONDecodeError:
                print(f"Skipping {meta_file} because of broken encoding")

        return meta_data_list_dicts

    @abstractmethod
    def create_sets(self):
        pass

    @abstractmethod
    def get_loaders(self):
        pass

    def create_train_val_test_ids(self, mode):
        '''Creates fixed and random train, val, test indices and saves them down.'''

        assert mode in ['baseline', 'agePrediction', 'labelBudgeting'], 'method cannot be used in mode <<transfer>>'

        id_list = [item['meta_data']['id'] for item in self.data_dict]
        id_list_shuffle = random.shuffle(id_list)

        if mode == 'baseline' or mode == 'agePrediction':
            num_samples = int(0.3 * len(id_list))
        elif mode == 'labelBudgeting':
            num_samples = int(0.9 * len(id_list))

        train_stop = int(num_samples*self.val_test_split)
        val_stop = int(num_samples*((1-self.val_test_split)/2))

        train_ids = id_list_shuffle[:train_stop]
        val_ids = id_list_shuffle[train_stop:val_stop]
        test_ids = id_list_shuffle[val_stop:]

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

    def create_old_young_ids(self, thresh_young=120, thresh_old=120, young_split=0.5):
        '''Creates fixed ids for old and young population and saves them down.'''

        age_dict = {item['id']: item['scan_age'] for item in self.meta_data_list_dicts}
        age_sorted_ids  = {k: v for k, v in sorted(age_dict.items(), key=lambda item: item[1])}
        age_sorted_list = list(age_sorted_ids.keys())

        old_ids = random.shuffle(age_sorted_list[-thresh_old:])
        young_ids = random.shuffle(age_sorted_list[:thresh_young])
        young_train_stop = int(thresh_young*young_split)
        young_train_ids = young_ids[:young_train_stop]
        young_test_ids = young_ids[young_train_stop:]

        id_path = self.root_dir / 'IDs'
        old_path = id_path / 'old_ids.csv'
        young_train_path = id_path / 'young_train_ids.csv'
        young_test_path = id_path / 'young_test_ids.csv'  

        Path(id_path).mkdir(parents=True, exist_ok=True)

        with open(old_path,'w') as old_file:
            wr = csv.writer(old_file, dialect='excel')
            wr.writerow(old_ids)

        with open(young_train_path,'w') as young_train_file:
            wr = csv.writer(young_train_file, dialect='excel')
            wr.writerow(young_train_ids)
        
        with open(young_test_path, 'w') as young_test_file:
            wr = csv.writer(young_test_file, dialect='excel')
            wr.writerow(young_test_ids)