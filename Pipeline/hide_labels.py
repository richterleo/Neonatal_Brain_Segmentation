from dHCP_transforms import ConvertToMultiChannelBasedOnDHCPClassesd, HideLabelsd, HideLabels
from dHCP_utils import create_indices, get_kernels_strides, get_slices_from_matrix
import monai
import pandas as pd
import re
import time
import os
import shutil
import tempfile
import tqdm
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
from monai.data import DataLoader, Dataset, decollate_batch, CacheNTransDataset, PersistentDataset
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet, DynUNet   
from monai.data.utils import pad_list_data_collate
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
    CropForegroundd
)
from monai.transforms.transform import Transform
from monai.utils import set_determinism
from monai.utils.misc import ensure_tuple_rep
from monai.networks.layers.factories import Act, Norm

from torchvision import transforms

import torch

t0 = round(time.time())

os.environ["MONAI_DATA_DIRECTORY"] = 'dHCP_Training' #set the environment variable
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
print(f"Root dir is {root_dir}.")

""" Save important information and set hyperparameters"""

mode = "Hide_Labels"
result_dir = root_dir + '/results/' + mode + '_results' + str(t0)
os.mkdir(result_dir)

#initialize dictionaries
meta_info_dict = {"session_date": str(datetime.today())}
meta_info_dict["session_content"] = "Random Slicing, Random Labels"

hyperparam_dict = {}
hyperparam_dict["modalities"] = "t1 and t2"

results_dict= {"epoch_loss": [], "mean_dice": [], "dice_BG": [], "dice_CSF": [],
                "dice_cGM": [], "dice_WM": [], "dice_bg": [], "dice_Ventricles":[], 
                "dice_Cerebellum": [], "dice_dGM": [], "dice_Brainstem": [], 
                "dice_Hippocampus": [], "mean_dice_crucial_structures": [], 
                "best_mean_dice": [], "best_epoch":[], "training_time": None}

analysis_dict = {"loss": [], "image_id": [], "subj_age": [], "epoch": [], "step": []}

#Set parameters
lr = 1e-2
max_epochs = 20
hide_labels = True
random_seed = 0 # does not affect label hiding
prop_of_whole = 1
batch_size = 2
age_group = 'whole'
slicing_mode = "random"
selection_mode = "random"
roi_size = [128, 128, 128]
pixdim = [0.5, 0.5, 0.5]

kernels, strides = get_kernels_strides(roi_size, pixdim)

hyperparam_dict["learning rate"] = lr
hyperparam_dict["max_epochs"] = max_epochs
hyperparam_dict["hide_labels"] = hide_labels
if not hide_labels:
    hyperparam_dict["slicing_mode"] = hyperparam_dict["selection_mode"] = None
else:
    hyperparam_dict["slicing_mode"] = slicing_mode
    hyperparam_dict["selection_mode"] = selection_mode
hyperparam_dict["batch_size"] = batch_size
hyperparam_dict["age_group"] = age_group

meta_info_dict["random_seed"] = random_seed


""" Set Random Seed for Reproducibility"""

set_determinism(seed=random_seed)

""" Load in Data"""


#Create directories

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


data_dir = os.path.join(root_dir, 'backup_dHCP') 
t1_dir = os.path.join(data_dir, 'T1w_full')
t2_dir = os.path.join(data_dir, 'T2w_full')
label_dir = os.path.join(data_dir, 'labels_full')
meta_data_dir = os.path.join(data_dir, 'meta_data_full')
cache_dir = os.path.join(result_dir, 'cache_dir')
os.mkdir(cache_dir)

#Create list of files
t1_list = sorted([os.path.join(t1_dir, file) for file in os.listdir(t1_dir)])
t2_list = sorted([os.path.join(t2_dir, file) for file in os.listdir(t2_dir)])
label_list = sorted([os.path.join(label_dir, file) for file in os.listdir(label_dir)])
meta_data_list = sorted([os.path.join(meta_data_dir, file) for file in os.listdir(meta_data_dir)])
meta_data_list_dicts = []

assert len(t1_list) == len(t2_list) == len(label_list) 

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


#Create basic training data dict
data_dict_full = [{"t1_image": t1_image, "t2_image": t2_image, "label": label, "meta_data": meta_data} for t1_image, t2_image, label, meta_data in zip(t1_list, t2_list, label_list, meta_data_list_dicts)]
# 709 entries

#We now want to use more data, so that in we have the same amount of annotated data as in the case with whole brains. need to recover list of ids
with open('dHCP_Training/three_fifths_of_data.csv', 'r') as f:
    id_list_recovered = f.read()

id_list_recovered = id_list_recovered.strip()
id_list_recovered = id_list_recovered.split(',')

data_dict = [dict_item for dict_item in data_dict_full if dict_item['meta_data']['id'] in id_list_recovered]
data_dict_list_ids = [item['meta_data']['id'] for item in data_dict]

print(f"We have {len(t1_list)} T1w training images, {len(t2_list)} T2w images, {len(label_list)} labels and {len(meta_data_list_dicts)} meta_data files in total.")
print(f"data dict consists of {len(data_dict)} samples")
#709 images and 709 labels 

""" Define Transformations"""

#Define transforms
visualisation_transform = Compose(
    [
        LoadImaged(keys=["t1_image", "t2_image" "label"]), #[217, 290, 290]
        AddChanneld(keys=["t1_image", "t2_image"]), #(2, 217, 290, 290)
        ConcatItemsd(keys=["t1_image", "t2_image"], name="image"),
        DeleteItemsd(keys=["t1_image", "t2_image"]),
        ToTensord(keys=["image", "label"]),
    ]
)

train_transform_hide_labels= Compose(
    [
        LoadImaged(keys=["t1_image", "t2_image", "label"]), #[217, 290, 290]
        ConvertToMultiChannelBasedOnDHCPClassesd(keys="label"), #(10, 217, 290, 290)
        AddChanneld(keys=["t1_image", "t2_image"]), #(2, 217, 290, 290)
        Spacingd(keys=["t1_image", "t2_image", "label"], pixdim=pixdim, mode=("bilinear", "bilinear", "nearest")),
        HideLabelsd(keys="label"),
        CropForegroundd(keys=["t1_image", "t2_image", "label", "label_slice_matrix"], source_key="t2_image", select_fn=lambda x: x>1, margin=0),
        ConcatItemsd(keys=["t1_image", "t2_image"], name="image"),
        DeleteItemsd(keys=["t1_image", "t2_image"]),
        RandSpatialCropd(
            keys=["image", "label", "label_slice_matrix"], roi_size=roi_size, random_size=False, random_center=True
        ), # [192, 192, 192]
        RandFlipd(keys=["image", "label", "label_slice_matrix"], prob=0.1, spatial_axis=0),
        RandFlipd(keys=["image", "label", "label_slice_matrix"], prob=0.1, spatial_axis=1),
        RandFlipd(keys=["image", "label", "label_slice_matrix"], prob=0.1, spatial_axis=2),
        Rand3DElasticd(
            keys=["image", "label", "label_slice_matrix"],
            mode=("bilinear", "nearest", "nearest"),
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


val_transform = Compose(
    [
        LoadImaged(keys=["t1_image", "t2_image", "label"]), #[217, 290, 290]
        ConvertToMultiChannelBasedOnDHCPClassesd(keys="label"),
        AddChanneld(keys=["t1_image", "t2_image"]), #(2, 217, 290, 290)
        Spacingd(keys=["t1_image", "t2_image", "label"], pixdim=pixdim),
        ConcatItemsd(keys=["t1_image", "t2_image"], name="image"),
        DeleteItemsd(keys=["t1_image", "t2_image"]),
        RandSpatialCropd(
            keys=["image", "label"], roi_size=roi_size, random_size=False, random_center=True
        ), # [192, 192, 192]
        #Resized(keys=["image", "label"], spatial_size=[64, 64, 64]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"]),
     
    ]
)



''' Construction of train/val set'''

#Create Train Test Val Split and Dataloader
train_indices, val_indices, test_indices, train_dict, val_dict, test_dict = create_indices(data_dict, prop_of_whole=prop_of_whole, val_frac=0.1, test_frac=0.1)


#train_ds_hide_labels = CacheNTransDataset(train_dict, transform=train_transform, cache_n_trans=0, cache_dir = cache_dir)
if hide_labels:
    train_ds = CacheNTransDataset(train_dict, transform=train_transform_hide_labels, cache_n_trans=5, cache_dir=cache_dir)
else:
    train_ds = Dataset(train_dict, transform=train_transform)
val_ds = Dataset(val_dict, transform=val_transform)

hyperparam_dict["size_training_set"] = len(train_ds)
hyperparam_dict["size_val_set"] = len(val_ds)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=pad_list_data_collate)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, collate_fn=pad_list_data_collate)


"""Creating the Model"""
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
loss_function = DiceCELoss(to_onehot_y=False, sigmoid = True, squared_pred=True) 
optimizer = torch.optim.SGD(
    model.parameters(),
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

hyperparam_dict["network_architecture"] = type(model).__name__
hyperparam_dict["loss_function"] = type(loss_function).__name__
hyperparam_dict["scheduler"] = type(scheduler).__name__


""" Training Loop"""

training_start_time= round(time.time()) #use for saving to different model

val_interval = 1
best_metric = -1
best_metric_epoch = -1

for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0

    for batch_data in train_loader:
        step += 1
        
        if hide_labels:
            inputs, labels, meta_data, label_slice_matrix, location = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
                batch_data["meta_data"],
                batch_data["label_slice_matrix"].to(device), 
                batch_data["label_slice_dict"]["location"].to(device)
            )
            optimizer.zero_grad()
            outputs = model(inputs) #shape of outputs is [B, 4, 10, 96, 96, 96]

            image_id = meta_data["id"]
            subj_age = meta_data["scan_age"]
            subj_age = subj_age.tolist()

            # aux function returns sliced outputs and labels with compatible shapes. 
            # outputs and labels are lists with len batch_size 
            # cannot be stacked because of non-matching dimensions (esp. if slicing_mode = "random", slicing axis is then randomly chosen)
            outputs, labels = get_slices_from_matrix(label_slice_matrix, location, outputs, labels)

            # in this case we sum the loss 
            loss = 0

            for output, label in zip(outputs, labels):
                output = torch.unbind(output, dim=0) #this is now a tuple

                loss += sum(
                0.5 ** i * loss_function(outp.unsqueeze(0), label.unsqueeze(0)) #need to unsqueeze since loss function expects [B, H, W, D]
                for i, outp in enumerate(output)
                )
                      
        else:

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

            outputs = torch.unbind(outputs, dim=1)

            loss = sum(
                    0.5 ** i * loss_function(output, labels)
                    for i, output in enumerate(outputs)
                )
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(
            f"{step}/{len(train_ds) // train_loader.batch_size}"
            f", train_loss: {loss.item():.4f}" 
            f", image_id: {image_id}"
            f", subject age: {subj_age}"
        )
        analysis_dict["loss"].append(loss.item())
        analysis_dict["image_id"].append(image_id)
        analysis_dict["subj_age"].append(subj_age)
        analysis_dict["step"].append(step)
        analysis_dict["epoch"].append(epoch)
    epoch_loss /= step
    results_dict["epoch_loss"].append(epoch_loss)

    scheduler.step()    
    print(f"epoch {epoch + 1} average loss: {epoch_loss}")


    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():

            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )

                val_outputs = model(val_inputs)
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                val_labels = val_labels.byte()
                dice_metric(y_pred=val_outputs, y=val_labels)
                dice_metric_batch(y_pred=val_outputs, y=val_labels)

            metric = dice_metric.aggregate().item()
            results_dict["mean_dice"].append(metric)
            metric_batch = dice_metric_batch.aggregate()

            metric0 = metric_batch[0].item()
            results_dict["dice_BG"].append(metric0)
            metric1 = metric_batch[1].item()
            results_dict["dice_CSF"].append(metric1)
            metric2 = metric_batch[2].item()
            results_dict["dice_cGM"].append(metric2)
            metric3 = metric_batch[3].item()
            results_dict["dice_WM"].append(metric3)
            metric4 = metric_batch[4].item()
            results_dict["dice_bg"].append(metric4)
            metric5 = metric_batch[5].item()
            results_dict["dice_Ventricles"].append(metric5)
            metric6 = metric_batch[6].item()
            results_dict["dice_Cerebellum"].append(metric6)
            metric7 = metric_batch[7].item()
            results_dict["dice_dGM"].append(metric7)
            metric8 = metric_batch[8].item()
            results_dict["dice_Brainstem"].append(metric8)
            metric9 = metric_batch[9].item()
            results_dict["dice_Hippocampus"].append(metric9)

            results_dict["mean_dice_crucial_structures"].append(np.mean([metric1, metric2, metric3, metric6, metric7, metric8, metric9]))

            dice_metric.reset()
            dice_metric_batch.reset()

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1

                torch.save(
                    model.state_dict(),
                    os.path.join(result_dir, "best_metric_model_epoch_" + str(epoch) + ".pth"),
                )
                print("saved new best metric model")
            
            results_dict["best_mean_dice"].append(best_metric)
            results_dict["best_epoch"].append(best_metric_epoch-1)
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f" background: {metric0:.4f} CSF: {metric1:.4f} cGM: {metric2:.4f}"
                f" WM: {metric3:.4f} BG: {metric4:.4f} Ventricle: {metric5:.4f}"
                f" Cerebellum: {metric6:.4f} dGM: {metric7:.4f} Brainstem: {metric8:.4f}"
                f" Hippocampus: {metric9:.4f}"
                f"\nbest mean dice: {best_metric:.4f}"
                f" at epoch: {best_metric_epoch}"
            )

end_time = round(time.time()) 
training_time = end_time - training_start_time
results_dict["training_time"] = training_time

print(f"From start to finish it took {training_time}s")

plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Epoch Average Loss")
x = [i + 1 for i in range(len(results_dict["epoch_loss"]))]
y = results_dict["epoch_loss"]
plt.xlabel("epoch")
plt.plot(x, y, color="red")
plt.subplot(1, 2, 2)
plt.title("Val Mean Dice")
x = [val_interval * (i + 1) for i in range(len(results_dict["mean_dice"]))]
y = results_dict["mean_dice"]
plt.xlabel("epoch")
plt.plot(x, y, color="green")
plt.savefig(result_dir + "/whole_brain_baseline_dice.png")
plt.show()

plt.figure("train", (18, 6))
plt.subplot(2, 5, 1)
plt.title("Val Mean Dice Background")
x = [val_interval * (i + 1) for i in range(len(results_dict["dice_BG"]))]
y = results_dict["dice_BG"]
plt.xlabel("epoch")
plt.plot(x, y, color="blue")

plt.subplot(2, 5, 2)
plt.title("Val Mean Dice CSF")
x = [val_interval * (i + 1) for i in range(len(results_dict["dice_CSF"]))]
y = results_dict["dice_CSF"]
plt.xlabel("epoch")
plt.plot(x, y, color="brown")

plt.subplot(2, 5, 3)
plt.title("Val Mean Dice cGM")
x = [val_interval * (i + 1) for i in range(len(results_dict["dice_cGM"]))]
y = results_dict["dice_cGM"]
plt.xlabel("epoch")
plt.plot(x, y, color="purple")

plt.subplot(2, 5, 4)
plt.title("Val Mean Dice WM")
x = [val_interval * (i + 1) for i in range(len(results_dict["dice_WM"]))]
y = results_dict["dice_WM"]
plt.xlabel("epoch")
plt.plot(x, y, color="yellow")

plt.subplot(2, 5, 5)
plt.title("Val Mean Dice bg")
x = [val_interval * (i + 1) for i in range(len(results_dict["dice_bg"]))]
y = results_dict["dice_bg"]
plt.xlabel("epoch")
plt.plot(x, y, color="red")

plt.subplot(2, 5, 6)
plt.title("Val Mean Dice Ventricle")
x = [val_interval * (i + 1) for i in range(len(results_dict["dice_Ventricles"]))]
y = results_dict["dice_Ventricles"]
plt.xlabel("epoch")
plt.plot(x, y, color="magenta")

plt.subplot(2, 5, 7)
plt.title("Val Mean Dice Cerebellum")
x = [val_interval * (i + 1) for i in range(len(results_dict["dice_Cerebellum"]))]
y = results_dict["dice_Cerebellum"]
plt.xlabel("epoch")
plt.plot(x, y, color="indigo")

plt.subplot(2, 5, 8)
plt.title("Val Mean Dice dGM")
x = [val_interval * (i + 1) for i in range(len(results_dict["dice_dGM"]))]
y = results_dict["dice_dGM"]
plt.xlabel("epoch")
plt.plot(x, y, color="cornflowerblue")

plt.subplot(2, 5, 9)
plt.title("Val Mean Dice Brainstem")
x = [val_interval * (i + 1) for i in range(len(results_dict["dice_Brainstem"]))]
y = results_dict["dice_Brainstem"]
plt.xlabel("epoch")
plt.plot(x, y, color="salmon")

plt.subplot(2, 5, 10)
plt.title("Val Mean Dice Hippocampus")
x = [val_interval * (i + 1) for i in range(len(results_dict["dice_Hippocampus"]))]
y = results_dict["dice_Hippocampus"]
plt.xlabel("epoch")
plt.plot(x, y, color="peru")

plt.tight_layout()
plt.savefig(result_dir + "/subplots.png")
plt.show()



t1 = round(time.time())
duration = t1 - t0
meta_info_dict["duration"] = duration
print(f"Time of execution: {duration}")

df_results_dict = pd.DataFrame(results_dict)
df_results_dict.to_csv(os.path.join(result_dir, 'results.csv'), sep="\t", index_label="epoch")

df_meta_info_dict = pd.DataFrame(meta_info_dict, index=[0])
df_meta_info_dict.to_csv(os.path.join(result_dir, 'meta_info.csv'), sep="\t", index=False)

df_hyperparam_dict = pd.DataFrame(hyperparam_dict, index=[0])
df_hyperparam_dict.to_csv(os.path.join(result_dir, 'hyperparams.csv'), sep="\t", index=False)


# df_analysis_dict = pd.DataFrame(analysis_dict)
# df_analysis_dict.to_csv(os.path.join(result_dir, 'loss_analysis.csv'), sep="\t", index=False)