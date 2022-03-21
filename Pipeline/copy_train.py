import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import pandas as pd
import random
import tempfile
import time
import torch

from datetime import datetime
from DataHandler import DataCollector
from Logging import resultsLogger
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
from pathlib import Path
from Plotting import resultPlotter
from torchvision import transforms
from transforms import ConvertToMultiChannelBasedOnDHCPClassesd, HideLabelsd, HideLabels
from typing import Callable, List, Mapping, Optional, Sequence, Tuple, Union
from utils import create_indices, get_kernels_strides


# Creates new directories for saving down results
resultlogger = resultsLogger('baseline', "only for testing")
resultlogger.create_result_folder()

# set random seed for reproducability
set_determinism(seed=resultlogger.meta_info["random_seed"])

# load data into dicts
datacollector = DataCollector(resultlogger.root_dir, resultlogger.result_dir, resultlogger.hyperparams["hide_labels"])


age_bins = np.linspace(np.min(scan_ages), np.max(scan_ages), 10)              
age_df = pd.DataFrame(data=scan_ages, columns=["scan_ages"])
age_df["bucket"] = pd.cut(age_df.scan_ages, age_bins)

# print(np.histogram(scan_ages, age_bins))
# for the small dataset: [ 1,  0,  4, 10, 16, 14, 40, 35, 22] --> use the 6th bucket as cutoff


""" Age Analysis"""


""" Define Transformations"""

visualisation_transform = Compose(
    [
        LoadImaged(keys=["t1_image", "t2_image", "label"]), # (217, 290, 290) orig size
        AddChanneld(keys=["t1_image", "t2_image", "label"]), 
        ConcatItemsd(keys=["t1_image", "t2_image"], name="image"),
        DeleteItemsd(keys=["t1_image", "t2_image"]),
        ToTensord(keys=["image", "label"]),
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


val_transform = Compose(
    [
        LoadImaged(keys=["t1_image", "t2_image", "label"]), #[217, 290, 290]
        ConvertToMultiChannelBasedOnDHCPClassesd(keys="label"),
        AddChanneld(keys=["t1_image", "t2_image"]), #(2, 217, 290, 290)
        Spacingd(keys=["t1_image", "t2_image", "label"], pixdim=pixdim, mode=("bilinear", "bilinear", "nearest")),
        CropForegroundd(keys=["t1_image", "t2_image", "label"], source_key="t2_image", select_fn=lambda x: x>1, margin=0),
        ConcatItemsd(keys=["t1_image", "t2_image"], name="image"),
        DeleteItemsd(keys=["t1_image", "t2_image"]),
        RandSpatialCropd(
            keys=["image", "label"], roi_size=roi_size, random_size=False, random_center=True
        ), 
        SpatialPadd(keys=["image", "label"], spatial_size=roi_size),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"]),
     
    ]
)



""" Create Dataloaders """

#Create Train Test Val Split and Dataloader, by default use 20% val data. test data is fixed + held out 
train_indices, val_indices, test_indices, train_dict, val_dict, test_dict = create_indices(data_dict, prop_of_whole=prop_of_whole, val_frac=0.2, test_frac=0)

#train_ds_hide_labels = CacheNTransDataset(train_dict, transform=train_transform, cache_n_trans=0, cache_dir = cache_dir)
train_ds = Dataset(train_dict, transform=train_transform)
val_ds = Dataset(val_dict, transform=train_transform)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

resultlogger.update_hyperparams(size_training_set = len(train_ds), size_val_set = len(val_ds)
print(f"We have {len(train_ds)} training images and {len(val_ds)} val images")

"""Creating the Model"""

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

resultlogger.update_hyperparams(network_architecture = type(model).__name__, 
                                loss_function = type(loss_function).__name__,
                                in_channels = str(model.in_channels),
                                kernel_size = str(model.kernel_size),
                                upsample_kernel_size = str(model.upsample_kernel_size)

""" Training Loop"""

resultlogger.restart_clock()

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
            inputs, labels, slice_matrix, location = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
                batch_data["label_slice_matrix"].to(device), 
                batch_data["label_meta_dict"]["location"].to(device)
            )
            optimizer.zero_grad()
            outputs = model(inputs)

            selected_slices = get_slices_from_matrix(slice_matrix, location)
            selected_slices = selected_slices.to(device)
            
            if location[0] == 0:
                print("sagittal")
                outputs = outputs[:,:,selected_slices,:,:]
                labels = labels[:,:,selected_slices,:,:]

            elif location[0] == 1:
                print("coronal")
                outputs = outputs[:,:,:,selected_slices,:]
                labels = labels[:,:,:,selected_slices,:]

            elif location[0] == 2:
                print("axial")
                outputs = outputs[:,:,selected_slices,:,:]
                labels = labels[:,:,selected_slices,:,:]
        
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

        resultlogger.log_analysis(loss.item(), image_id, subj_age, step, epoch)

    epoch_loss /= step
    resultlogger.results["epoch_loss"].append(epoch_loss)

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
            resultlogger.results["mean_dice"].append(metric)
            metric_batch = dice_metric_batch.aggregate()

            resultlogger.log_tcs(metric_batch)

            resultlogger.results["mean_dice_crucial_structures"].append(np.mean([metric1, metric2, metric3, metric6, metric7, metric8, metric9]))

            dice_metric.reset()
            dice_metric_batch.reset()

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1

                torch.save(
                    model.state_dict(),
                    os.path.join(resultlogger.result_dir, "best_metric_model_epoch_" + str(epoch) + ".pth"),
                )
                print("saved new best metric model")
            
            resultlogger.results["best_mean_dict"].append(best_metric)
            resultlogger.results["best_epoch"].append(best_metric_epoch-1)
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f" background: {metric0:.4f} CSF: {metric1:.4f} cGM: {metric2:.4f}"
                f" WM: {metric3:.4f} BG: {metric4:.4f} Ventricle: {metric5:.4f}"
                f" Cerebellum: {metric6:.4f} dGM: {metric7:.4f} Brainstem: {metric8:.4f}"
                f" Hippocampus: {metric9:.4f}"
                f"\nbest mean dice: {best_metric:.4f}"
                f" at epoch: {best_metric_epoch}"
            )



