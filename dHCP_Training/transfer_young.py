from dHCP_transforms import ConvertToMultiChannelBasedOnDHCPClassesd, HideLabelsd, HideLabels
from dHCP_utils import create_indices, get_kernels_strides, set_parameter_requires_grad
import monai
import pandas as pd
import re
import time
import os
import shutil
from collections import Counter, OrderedDict
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
    AsChannelFirstd,
    AsDiscrete,
    CenterSpatialCropd,
    Compose,
    LoadImage,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    ToTensord,
    AddChanneld,
    Resized,
    EnsureTyped,
    EnsureType,
    ResizeWithPadOrCropd,
    ConcatItemsd,
    DeleteItemsd,
    Rand3DElasticd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    CropForegroundd,
    SpatialPadd
)
from monai.transforms.transform import Transform
from monai.utils import set_determinism
from monai.utils.misc import ensure_tuple_rep
from monai.networks.layers.factories import Act, Norm
from monai.inferers import sliding_window_inference

from torchvision import transforms

import torch

t0 = round(time.time())

os.environ["MONAI_DATA_DIRECTORY"] = 'dHCP_Training' #set the environment variable
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory

print(f"Root dir is {root_dir}.")

""" Save Important Information and Set Hyperparameters """

mode = "Finetuning_youngest_dataset"
result_dir = root_dir + '/results/' + mode + '_results' + str(t0)
os.mkdir(result_dir)

# initialize dictionaries to save down meta data and results
meta_info_dict = {"session_date": str(datetime.today())}
meta_info_dict["session_content"] = "Finetuning"

hyperparam_dict = {}

results_dict= {"epoch_loss": [], "mean_dice": [], "dice_BG": [], "dice_CSF": [],
                "dice_cGM": [], "dice_WM": [], "dice_bg": [], "dice_Ventricles":[], 
                "dice_Cerebellum": [], "dice_dGM": [], "dice_Brainstem": [], 
                "dice_Hippocampus": [], "mean_dice_crucial_structures": [], 
                "best_mean_dice": [], "best_epoch":[], "training_time": None}

results= {}


# Set parameters
transfer_lr_weight = 0.1
max_epochs = 10
random_seed = 25
batch_size = 2
model_size = 'big' #{'big', 'small'}
path_load_model = 'dHCP_Training/results/Pretrain_on_oldest_results1631247676/best_metric_model_epoch_16.pth'
strategy = 'fine_tuning' #{'no_fine_tuning', 'fine_tuning', 'shallow', 'medium', 'deep'}

# Get dependent parameters
lr = 1e-2 * transfer_lr_weight
if model_size == 'big':
    roi_size = [128, 128, 128]
    pixdim = [0.5, 0.5, 0.5]
elif model_size == 'small':
    roi_size = [96, 96, 96]
    pixdim = [0.6, 0.6, 0.6]    
kernels, strides = get_kernels_strides(roi_size, pixdim)

hyperparam_dict["max_epochs"] = max_epochs
hyperparam_dict["roi_size"] = str(roi_size)
hyperparam_dict["pixdim"] = str(pixdim)
hyperparam_dict["pretrained_model_path"] = path_load_model
hyperparam_dict["strategy"] = strategy

meta_info_dict["random_seed"] = random_seed

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

data_dir = os.path.join(root_dir, 'backup_dHCP') #'dHCP_Training/backup_dHCP'
t1_dir = os.path.join(data_dir, 'T1w_full')
t2_dir = os.path.join(data_dir, 'T2w_full') 
label_dir = os.path.join(data_dir, 'labels_full')
meta_data_dir = os.path.join(data_dir, 'meta_data_full') 

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

""" Create train and test set """

with open('dHCP_Training/Young_train_ids.csv', 'r') as f:
    young_train_ids = f.read()

young_train_ids = young_train_ids.strip()
young_train_id_list = young_train_ids.split(',')

with open('dHCP_Training/Young_test_ids.csv', 'r') as f:
    young_test_ids = f.read()

young_test_ids = young_test_ids.strip()
young_test_id_list = young_test_ids.split(',')

train_dict = [item for item in data_dict_full if item['meta_data']['id'] in young_train_id_list]
test_dict = [item for item in data_dict_full if item['meta_data']['id'] in young_test_id_list]

print(f"We use the {len(train_dict)} images for training and {len(test_dict)} for testing.")

""" Define Transformations"""

train_transform= Compose(
    [
        LoadImaged(keys=["t1_image", "t2_image", "label"]), #[217, 290, 290]
        ConvertToMultiChannelBasedOnDHCPClassesd(keys="label"), #(10, 217, 290, 290)
        AddChanneld(keys=["t1_image", "t2_image"]), #(2, 217, 290, 290)
        Spacingd(keys=["t1_image", "t2_image", "label"], pixdim=pixdim, mode=("bilinear", "bilinear", "nearest")),
        CropForegroundd(keys=["t1_image", "t2_image", "label"], source_key="t2_image", select_fn=lambda x: x>1, margin=0),
        ConcatItemsd(keys=["t1_image", "t2_image"], name="image"),
        DeleteItemsd(keys=["t1_image", "t2_image"]),
        RandSpatialCropd(
            keys=["image", "label"], roi_size=roi_size, random_size=False, random_center=True
        ), # [192, 192, 192]
        SpatialPadd(keys=["image", "label"], spatial_size=roi_size),
        RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=2),
        Rand3DElasticd(
            keys=["image", "label"],
            mode=("bilinear", "nearest"),
            prob=0.24,
            sigma_range=(5, 8),
            magnitude_range=(40, 80),
            translate_range=(20, 20, 20),
            rotate_range=(np.pi / 36, np.pi / 36, np.pi),
            scale_range=(0.15, 0.15, 0.15),
            padding_mode="reflection",
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandGaussianNoised(keys=["image"], std=0.01, prob=0.13),
        RandGaussianSmoothd(
            keys=["image"],
            sigma_x=(0.5, 1.15),
            sigma_y=(0.5, 1.15),
            sigma_z=(0.5, 1.15),
            prob=0.13,
        ),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.24),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.24),
        EnsureTyped(keys=["image", "label"]),
    ]
)


test_transform= Compose(
    [
        LoadImaged(keys=["t1_image", "t2_image", "label"]), #[217, 290, 290]
        ConvertToMultiChannelBasedOnDHCPClassesd(keys="label"), #(10, 217, 290, 290)
        AddChanneld(keys=["t1_image", "t2_image"]), #(2, 217, 290, 290)
        Spacingd(keys=["t1_image", "t2_image", "label"], pixdim=pixdim, mode=("bilinear", "bilinear", "nearest")),
        CenterSpatialCropd(keys=["t1_image", "t2_image", "label"], roi_size=[217, 290, 290]),
        SpatialPadd(keys=["t1_image", "t2_image", "label"], spatial_size=[217, 290, 290]),
        NormalizeIntensityd(keys=["t1_image", "t2_image"], nonzero=True, channel_wise=True),
        ConcatItemsd(keys=["t1_image", "t2_image"], name="image"),
        DeleteItemsd(keys=["t1_image", "t2_image"]),
        EnsureTyped(keys=["image", "label"]),
    ]
)



""" Create Dataloaders """


#train_ds_hide_labels = CacheNTransDataset(train_dict, transform=train_transform, cache_n_trans=0, cache_dir = cache_dir)
train_ds = Dataset(train_dict, transform=train_transform)
test_ds = Dataset(test_dict, transform=test_transform)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)


"""Load model from as pre-trained model on old age group"""


# Use nnU-Net model with best params 
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


''' Load pre-trained model'''

#model.load_state_dict(torch.load(path_load_model))

''' Different levels of fine tuning'''

param_names_to_update, params_to_update = set_parameter_requires_grad(model, strategy, model_size)

''' Define optimizer, loss function etc'''

loss_function = DiceCELoss(to_onehot_y=False, sigmoid = True, squared_pred=True) 
optimizer = torch.optim.SGD(
    params_to_update, #only train params for which requires_grad is True
    lr=lr,
    momentum=0.99,
    weight_decay=3e-5,
    nesterov=True,
)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (1 - epoch / max_epochs) ** 0.9)
dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
post_trans = Compose(
    [EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold_values=True)]
)

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

""" Training Loop"""

training_start_time = round(time.time()) #use for saving to different model

for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0

    for batch_data in train_loader:
        step += 1
    
        inputs, labels, meta_data = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
            batch_data["meta_data"]
        )
        optimizer.zero_grad()
        outputs = model(inputs)

        image_id = meta_data["id"]
        subj_age = meta_data["scan_age"]
        subj_age = subj_age.tolist()

        # deep supervision makes output tuple
        outputs = torch.unbind(outputs, dim=1) #this is a tuple of len 4 with shape [2,10,96,96,96]

        # compute deep supervision loss 
        loss = sum(0.5 ** i * loss_function(output, labels)
                for i, output in enumerate(outputs))
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(
            f"{step}/{len(train_ds) // train_loader.batch_size}"
            f", train_loss: {loss.item():.4f}" 
            f", image_ids: {image_id}"
            f", subject ages: {subj_age}"
        )
    epoch_loss /= step
    results_dict["epoch_loss"].append(epoch_loss)

    scheduler.step()    
    print(f"epoch {epoch + 1} average loss: {epoch_loss}")

''' Evaluate on the test set'''

model.eval()

step = 0
with torch.no_grad():
    for test_data in test_loader:
        test_inputs, test_labels = (
        test_data["image"].to(device),
        test_data["label"].to(device),
        )
        test_outputs = inference(test_inputs)
        test_outputs = [post_trans(i) for i in decollate_batch(test_outputs)]
        dice_metric(y_pred=test_outputs, y=test_labels)
        dice_metric_batch(y_pred=test_outputs, y=test_labels)
        step +=1
        print(f'step {step}')

    results['metric'] = dice_metric.aggregate().item()

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

torch.save(
    model.state_dict(),
    os.path.join(result_dir, "model_" + str(strategy) + ".pth"),
)
print("saved new best metric model")


''' Save results'''

end_time = round(time.time()) 
training_time = end_time - training_start_time
results_dict["training_time"] = training_time
print(f"Training time: {training_time}s")

t1 = round(time.time())
duration = t1 - t0
meta_info_dict["duration"] = duration
print(f"Time of execution: {duration}")

df_meta_info_dict = pd.DataFrame(meta_info_dict, index=[0])
df_meta_info_dict.to_csv(os.path.join(result_dir, 'meta_info.csv'), sep="\t", index=False)

# df_results_dict = pd.DataFrame(results_dict)
# df_results_dict.to_csv(os.path.join(result_dir, 'results.csv'), sep="\t", index_label="epoch")

df_results = pd.DataFrame(results, index=[0])
df_results.to_csv(os.path.join(result_dir, 'results.csv'), sep="\t", index=False)

df_hyperparam_dict = pd.DataFrame(hyperparam_dict, index=[0])
df_hyperparam_dict.to_csv(os.path.join(result_dir, 'hyperparams.csv'), sep="\t", index=False)

# df_analysis_dict = pd.DataFrame(analysis_dict)
# df_analysis_dict.to_csv(os.path.join(result_dir, 'loss_analysis.csv'), sep="\t", index=False)



