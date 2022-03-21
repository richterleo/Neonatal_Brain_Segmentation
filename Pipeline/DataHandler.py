import json
import numpy as np
import os

from pathlib import Path

class NpDecoder(json.JSONDecoder):
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

class DataCollector:

    def __init__(self, root_dir, result_dir, hide_labels, db='dHCP'):
        self.db_path = root_dir / db # Pipeline/dHCP
        self.result_dir = result_dir
        self.t1_dir = self.db_path / 'TW1_full'
        self.t2_dir = self.db_path / 'TW2_full'
        self.label_dir = self.db_path / 'labels_full'
        self.meta_data_dir = self.db_path / 'meta_data_full'
        if hide_labels:
            self.cache_dir = self.result_dir / 'cache_dir'

        self.t1_list = sorted([os.path.join(self.t1_dir, f) for f in os.listdir(self.t1_dir)])
        self.t2_list = sorted([os.path.join(self.t2_dir, f) for f in os.listdir(self.t2_dir)])
        self.label_list = sorted([os.path.join(self.label_dir, f) for f in os.listdir(self.label_dir)])
        self.meta_data_list = sorted([os.path.join(self.meta_data_dir, f) for f in os.listdir(self.meta_data_dir)])
        assert len(self.t1_list) == len(self.t2_list) == len(self.label_list) == len(self.meta_data_list), "number of train and label files does not align"

        self.meta_data_list_dicts = self.read_meta_data()
        self.data_dict = [{"t1_image": t1_img, "t2_image": t2_img, "label": label, "meta_data": meta_data} 
                        for t1_img, t2_img, label, meta_data in zip(self.t1_list, self.t2_list, self.label_list, self.meta_data_list_dicts)]
    
    def read_meta_data(self):
        '''Read json encoded meta data into dicts
        
        Returns:
            dict[]: '''
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

