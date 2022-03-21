from dHCP_transforms import ConvertToMultiChannelBasedOnDHCPClassesd
from dHCP_utils import create_indices, get_kernels_strides
import monai
import pandas as pd
import re
import time
import os
import shutil
import random
from datetime import datetime
from typing import Callable, List, Mapping, Optional, Sequence, Tuple, Union
import json

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from monai.transforms.inverse import InvertibleTransform
from monai.apps import DecathlonDataset, download_and_extract, extractall
from monai.config import print_config, DtypeLike, KeysCollection
from monai.data import DataLoader, Dataset, decollate_batch, CacheNTransDataset
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet, DynUNet   
from monai.transforms import (
    Activations,
    AsDiscrete,
    CenterSpatialCropd,
    Compose,
    LoadImaged,
    NormalizeIntensityd,
    Spacingd,
    ToTensord,
    AddChanneld,
    Resized,
    EnsureTyped,
    EnsureType,
    ConcatItemsd,
    DeleteItemsd,
)
from monai.transforms.transform import Transform
from monai.utils import set_determinism
from monai.utils.misc import ensure_tuple_rep
from monai.inferers import sliding_window_inference
from monai.networks.layers.factories import Act, Norm
from monai.data import NiftiSaver
from monai.transforms.utils import map_classes_to_indices

from torchvision import transforms

import torch

t0 = round(time.time())

os.environ["MONAI_DATA_DIRECTORY"] = 'dHCP_Training' #set the environment variable
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory

print(f"Root dir is {root_dir}.")

""" Save Important Information and Set Hyperparameters """

mode = "Segmentations_Baseline_Combined"
result_dir = root_dir + '/results/' + mode + '_results' + str(t0)
os.mkdir(result_dir)

# initialize dictionaries to save down meta data and results
meta_info_dict = {"session_date": str(datetime.today())}
meta_info_dict["session_content"] = "Save segmentation sample of the best baseline model, combine classes."

hyperparam_dict = {}
results= {}

saves_dict = {}

# Set parameters
prop_of_whole = 1
batch_size = 1
model_size = 'big' # network architecture must match the one that one we want to test
# path_load_model = 'dHCP_Training/results/test_Train_on_old_results1631032916/best_metric_model_epoch_1.pth'
path_load_model = 'dHCP_Training/results/DynUnet_results1630829112/best_metric_model_epoch_59.pth'

# get dependent params
if model_size == 'big':
    roi_size = [128, 128, 128]
    pixdim = [0.5, 0.5, 0.5]
elif model_size == 'small':
    roi_size = [96, 96, 96]
    pixdim = [0.6, 0.6, 0.6]    
kernels, strides = get_kernels_strides(roi_size, pixdim)

hyperparam_dict["roi_size"] = str(roi_size)
hyperparam_dict["pixdim"] = str(pixdim)
hyperparam_dict["pretrained_model_path"] = path_load_model
random_seed = 0

""" Set Random Seed for Reproducibility"""

set_determinism(seed=random_seed)

""" Load Data into Dicts """

# load in meta data from json files
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

data_dir = os.path.join(root_dir, 'backup_dHCP/testset') #'dHCP_Training/backup_dHCP'
t1_dir = os.path.join(data_dir, 'T1w')
t2_dir = os.path.join(data_dir, 'T2w') 
label_dir = os.path.join(data_dir, 'labels')
meta_data_dir = os.path.join(data_dir, 'meta_data') 

# Create list of files
t1_list = sorted([os.path.join(t1_dir, file) for file in os.listdir(t1_dir)])
t2_list = sorted([os.path.join(t2_dir, file) for file in os.listdir(t2_dir)])
label_list = sorted([os.path.join(label_dir, file) for file in os.listdir(label_dir)])
meta_data_list = sorted([os.path.join(meta_data_dir, file) for file in os.listdir(meta_data_dir)])
meta_data_list_dicts = []

for meta_file in meta_data_list:
  try: 
    with open(meta_file, 'r') as f:
      file_name = meta_file.split('/')[-1]
      id = file_name.split('_meta_data.json')[0]
      meta_data = json.load(f)
      meta_data["id"] = id
      meta_data_list_dicts.append(meta_data)
  except json.JSONDecodeError:
    print(f"Skipping {meta_file} because of broken encoding")

# Create basic training data dict
data_dict = [{"t1_image": t1_image, "t2_image": t2_image, "label": label, "meta_data": meta_data} for t1_image, t2_image, label, meta_data in zip(t1_list, t2_list, label_list, meta_data_list_dicts)]

assert len(t1_list) == len(t2_list) == len(label_list) == len(meta_data_list_dicts)

""" Define Transformations"""

transform= Compose(
    [
        LoadImaged(keys=["t1_image", "t2_image", "label"]), #[217, 290, 290]
        ConvertToMultiChannelBasedOnDHCPClassesd(keys="label"), #(10, 217, 290, 290)
        AddChanneld(keys=["t1_image", "t2_image"]), #(2, 217, 290, 290)
        Spacingd(keys=["t1_image", "t2_image", "label"], pixdim=pixdim, mode=("bilinear", "bilinear", "nearest")),
        NormalizeIntensityd(keys=["t1_image", "t2_image"], nonzero=True, channel_wise=True),
        ConcatItemsd(keys=["t1_image", "t2_image"], name="image"),
        DeleteItemsd(keys=["t1_image", "t2_image"]),
        EnsureTyped(keys=["image", "label"]),
    ]
)


''' Save down a segmentation of a young sample, an old sample, and a medium-aged sample'''

# list of scan ages
scan_ages = [item["scan_age"] for item in meta_data_list_dicts]

median_age = np.median(scan_ages)
avg_age = np.mean(scan_ages)
print(f"Median Scan Age is {median_age} and Average Scan Age is {avg_age}.")
# small dataset: median scan age is40.57 and average scan age is 39.69
# whole dataset: median scan age is 40.57 and average scan age is 39.56

# split age into 3 buckets and see how many subjects are in each bucket
age_bins = np.linspace(np.min(scan_ages), np.max(scan_ages), 4)              
age_df = pd.DataFrame(data=scan_ages, columns=["scan_ages"])
age_df["bucket"] = pd.cut(age_df.scan_ages, age_bins)

distribution, buckets = np.histogram(scan_ages, age_bins)
print(f"Young: under {buckets[1]}, old: after {buckets[2]}") #young: under 36.95, old: over 39

random.seed(0)

young_neonates = [item for item in data_dict if item['meta_data']['scan_age'] < buckets[1]]
medium_neonates = [item for item in data_dict if (item['meta_data']['scan_age'] > buckets[1] and item['meta_data']['scan_age'] < buckets[2])]
old_neonates = [item for item in data_dict if item['meta_data']['scan_age'] > buckets[2]]
print(f"We have {len(young_neonates)} samples in the young age group, {len(medium_neonates)} samples in the medium age group and {len(old_neonates)} samples in the old age group.")

# choose one example in each age group to evaluate on
young_sample = [young_neonates[0]]
print(f"young sample: id: {young_sample[0]['meta_data']['id']}, age: {young_sample[0]['meta_data']['scan_age']}")

medium_sample = [medium_neonates[0]]
print(f"medium sample: id: {medium_sample[0]['meta_data']['id']}, age: {medium_sample[0]['meta_data']['scan_age']}")

old_sample = [old_neonates[0]]
print(f"old sample: id: {old_sample[0]['meta_data']['id']}, age: {old_sample[0]['meta_data']['scan_age']}")

saves_dict['young_sample_id'] = young_sample[0]['meta_data']['id']
saves_dict['young_sample_scan_age'] = young_sample[0]['meta_data']['scan_age']
saves_dict['young_sample_t1_path'] = young_sample[0]['t1_image']
saves_dict['young_sample_t2_path'] = young_sample[0]['t2_image']
saves_dict['young_sample_label_path'] = young_sample[0]['label']

saves_dict['medium_sample_id'] = medium_sample[0]['meta_data']['id']
saves_dict['medium_sample_scan_age'] = medium_sample[0]['meta_data']['scan_age']
saves_dict['medium_sample_t1_path'] = medium_sample[0]['t1_image']
saves_dict['medium_sample_t2_path'] = medium_sample[0]['t2_image']
saves_dict['medium_sample_label_path'] = medium_sample[0]['label']

saves_dict['old_sample_id'] = old_sample[0]['meta_data']['id']
saves_dict['old_sample_scan_age'] = old_sample[0]['meta_data']['scan_age']
saves_dict['old_sample_t1_path'] = old_sample[0]['t1_image']
saves_dict['old_sample_t2_path'] = old_sample[0]['t2_image']
saves_dict['old_sample_label_path'] = old_sample[0]['label']

test_data = young_sample + medium_sample + old_sample
test_ds = Dataset(test_data, transform=transform)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

'''Load model that we want to evaluate on the test set'''

device = torch.device("cuda:0")
model = DynUNet(
    spatial_dims=3,
    in_channels=2,
    out_channels=10,
    kernel_size=kernels,
    strides=strides, #2
    upsample_kernel_size=strides[1:],
    deep_supervision=True,
    deep_supr_num = 3,
    res_block=True
).to(device)

model.load_state_dict(torch.load(path_load_model))

post_trans = Compose(
    [EnsureType(), Activations(sigmoid=True)
    #, AsDiscrete(threshold_values=True)
])

''' Define sliding window inference function '''

def inference(input):
    
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=(128, 128, 128),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
            # mode='gaussian'
        )

    return _compute(input)


model.eval()


niftisaver = NiftiSaver(output_dir=result_dir, output_postfix = 'baseline_seg', separate_folder=False)
#niftilabelsaver = NiftiSaver(output_dir=result_dir, output_postfix='label', separate_folder=False)

with torch.no_grad():
    for test_data in test_loader:
        test_inputs, test_labels = ( #label is list of len batch_size. item has shape [10, 217, 290, 290]
        test_data["image"].to(device),
        test_data["label"].to(device),
        )
        test_outputs = inference(test_inputs)
        test_outputs = [post_trans(i) for i in decollate_batch(test_outputs)] #this is a list with len batch_size. items have shape [10, 217, 290, 290]

        test_output_comb = torch.argmax(test_outputs[0], dim=0)
        test_output_comb = test_output_comb.unsqueeze(0)
        # meta_data = {'original_affine': test_data['label_meta_dict']['original_affine'][0], 'filename_or_obj': test_data['meta_data']['id'][0]}
        # meta_dict = {test_data['label_meta_dict'].keys

        meta_data = {label_key: test_data['label_meta_dict'][label_key][0] for label_key in test_data['label_meta_dict'].keys()}
        meta_data['filename_or_obj']= test_data['meta_data']['id'][0]
        # print(meta_dict)
        # meta_dict["filename_or_obj"] = test_data['meta_data']['id'][0]
        # print(test_data['meta_data']['id'][0])
        niftisaver.save(test_output_comb, meta_data)
        # niftilabelsaver.save(test_labels[0], meta_dict)






