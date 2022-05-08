from Transforms import ConvertToMultiChannelBasedOnDHCPClassesd
from Utils import create_indices, get_kernels_strides
from AgeDynUnet import AgeDynUNet
from sliding_window_inference_age import sliding_window_inference_age
import pandas as pd
import time
import os
from datetime import datetime
from typing import Callable, List, Mapping, Optional, Sequence, Tuple, Union
import json

import numpy as np
from monai.transforms.inverse import InvertibleTransform
from monai.apps import DecathlonDataset, download_and_extract, extractall
from monai.config import print_config, DtypeLike, KeysCollection
from monai.data import DataLoader, Dataset, decollate_batch, CacheNTransDataset
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric, MSEMetric, MAEMetric
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

from torchvision import transforms

import torch

t0 = round(time.time())

os.environ["MONAI_DATA_DIRECTORY"] = 'dHCP_Training' #set the environment variable
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory

print(f"Root dir is {root_dir}.")

""" Save Important Information and Set Hyperparameters """

mode = "AgeUNet_Evaluation"
result_dir = root_dir + '/results/' + mode + '_results' + str(t0)
os.mkdir(result_dir)

# initialize dictionaries to save down meta data and results
meta_info_dict = {"session_date": str(datetime.today())}
meta_info_dict["session_content"] = "Test AgeUNet"

hyperparam_dict = {}

results= {}

# Set parameters
prop_of_whole = 1
batch_size = 1
model_size = 'big' # network architecture must match the one that one we want to test
# path_load_model = 'dHCP_Training/results/test_Train_on_old_results1631032916/best_metric_model_epoch_1.pth'
path_load_model = 'dHCP_Training/results/Age_Segmentation_Dynunet_results1631399790/best_metric_model_epoch_57.pth'

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
data_dict_full = [{"t1_image": t1_image, "t2_image": t2_image, "label": label, "meta_data": meta_data} for t1_image, t2_image, label, meta_data in zip(t1_list, t2_list, label_list, meta_data_list_dicts)]

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


""" Create Dataloaders """

# with open('dHCP_Training/Young_test_ids.csv', 'r') as f:
#     young_test_ids = f.read()

# young_test_ids = young_test_ids.strip()
# young_test_id_list = young_test_ids.split(',')

#test_dict = [item for item in data_dict_full if item['meta_data']['id'] in young_test_id_list]
test_dict = data_dict_full

test_ds = Dataset(test_dict, transform=transform)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

print(f"Number of test images: {len(test_ds)}")

'''Load model that we want to evaluate on the test set'''

device = torch.device("cuda:0")
model = AgeDynUNet(
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
    [EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold_values=True)]
)

''' Define sliding window inference function '''

def inference(input):
    
    def _compute(input):
        return sliding_window_inference_age(
            inputs=input,
            roi_size=(128, 128, 128),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
            # mode='gaussian'
        )

    return _compute(input)

''' Define Dice metric'''

dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
mse_metric = MSEMetric()
mae_metric = MAEMetric()


model.eval()

metric_values = []
metric_values_BG = []
metric_values_CSF = []
metric_values_cGM = []
metric_values_WM = []
metric_values_bg = []
metric_values_Ventricles = []
metric_values_Cerebellum = []
metric_values_dGM = []
metric_values_Brainstem = []
metric_values_Hippocampus = []

step = 0
with torch.no_grad():
    for test_data in test_loader:
        test_inputs, test_labels, test_meta_data = (
        test_data["image"].to(device),
        test_data["label"].to(device),
        test_data['meta_data']
        )
        test_outputs, age_pred = inference(test_inputs)
        age_pred = torch.unsqueeze(torch.unsqueeze(age_pred, 0),0)
        age = torch.unsqueeze(test_meta_data['scan_age'], 0).cpu()
        mae_metric(y_pred=age_pred, y=age)
        mse_metric(y_pred=age_pred, y=age)
        test_outputs = [post_trans(i) for i in decollate_batch(test_outputs)]
        dice_metric(y_pred=test_outputs, y=test_labels)
        dice_metric_batch(y_pred=test_outputs, y=test_labels)
        step +=1
        print(f'step {step}')

    results['metric'] = dice_metric.aggregate().item()
    results['mse_metric'] = mse_metric.aggregate().item()
    results['mae_metric'] = mae_metric.aggregate().item()

    metric_batch = dice_metric_batch.aggregate()
    results['metric_BG'] = metric_batch[0].item()
    results['metric_CSF'] = metric_batch[1].item()
    results['metric_cGM'] = metric_batch[2].item()
    results['metric_WM'] = metric_batch[3].item()
    results['metric_bg'] = metric_batch[4].item()
    results['metric_Ventricles'] = metric_batch[5].item()
    results['metric_Cerebellum'] = metric_batch[6].item()
    results['metric_dGM'] = metric_batch[7].item()
    results['metric_Brainstem'] = metric_batch[8].item()
    results['metric_Hippocampus'] = metric_batch[9].item()

    dice_metric.reset()
    dice_metric_batch.reset()


t1 = round(time.time())
duration = t1 - t0
results['test_time'] = duration

print(results['metric'])

df_meta_info_dict = pd.DataFrame(meta_info_dict, index=[0])
df_meta_info_dict.to_csv(os.path.join(result_dir, 'meta_info.csv'), sep="\t", index=False)

df_results = pd.DataFrame(results, index=[0])
df_results.to_csv(os.path.join(result_dir, 'results.csv'), sep="\t", index=False)

df_hyperparam_dict = pd.DataFrame(hyperparam_dict, index=[0])
df_hyperparam_dict.to_csv(os.path.join(result_dir, 'hyperparams.csv'), sep="\t", index=False)


