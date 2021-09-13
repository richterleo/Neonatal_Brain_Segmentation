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
import csv

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

from torchvision import transforms

import torch

from scipy.stats import chisquare, ttest_ind
import seaborn as sns

os.environ["MONAI_DATA_DIRECTORY"] = 'dHCP_Training' #set the environment variable
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory

print(f"Root dir is {root_dir}.")


''' Load in meta_data from json files into dict list '''
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
meta_data_dir = os.path.join(data_dir, 'meta_data_full')  

# Create list of files
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

#print(f"this is the length of the meta_data_list_dicts: {len(meta_data_list_dicts)}") #709

''' analyse age '''

# list of scan ages
scan_ages = [item["scan_age"] for item in meta_data_list_dicts]
print(len(scan_ages))

median_age = np.median(scan_ages)
avg_age = np.mean(scan_ages)
print(f"Median Scan Age is {median_age} and Average Scan Age is {avg_age}.")
# small dataset: median scan age is40.57 and average scan age is 39.69
# whole dataset: median scan age is 40.57 and average scan age is 39.56

# split age into 10 buckets and see how many subjects are in each bucket
age_bins = np.linspace(np.min(scan_ages), np.max(scan_ages), 10)              
age_df = pd.DataFrame(data=scan_ages, columns=["scan_ages"])
age_df["bucket"] = pd.cut(age_df.scan_ages, age_bins)

distribution, buckets = np.histogram(scan_ages, age_bins)
print(f"Young: under {buckets[5]}, old: after {buckets[6]}") #young: under 36.95, old: over 39
print(f"In the young age group, there are")
print(buckets)
# for the small dataset: [ 1,  0,  4, 10, 16, 14, 40, 35, 22] --> use the 6th bucket as cutoff (14)
# for the whole dataset: [5, 13, 25, 43, 62, 72, 206, 191, 92] 
# buckets are: [26.71, 28.75777778, 30.80555556, 32.85333333, 34.90111111, 36.94888889, 38.99666667, 41.04444444, 43.09222222, 45.14]


# does the distribution of scan ages differ significantly in the small sample from the large sample?
print(chisquare([1,  0,  4, 10, 16, 14, 40, 35, 22], f_exp=[i *  142/709 for i in [5, 13, 25, 43, 62, 72, 206, 191, 92]] ))
# statistic=5.084088612465714, pvalue=0.7485519900666324) --> null hypothesis that thy come from the same distribution cannot be "verworfen"


''' pick new dataset which contains 3 x the data from before. it is not important whether the same data is in this one as in the small dataset '''
random.seed(0)
shuffled_meta_data_dict = random.sample(meta_data_list_dicts, len(meta_data_list_dicts))

#pick 3 x 142 random samples, but random should be fixed
three_fifths_dataset = shuffled_meta_data_dict[:426]

#analyse this dataset again:
scan_ages_three_fifths = [item["scan_age"] for item in three_fifths_dataset]
print(len(scan_ages_three_fifths))


median_age_three_fifths = np.median(scan_ages_three_fifths)
avg_age_three_fifths = np.mean(scan_ages_three_fifths)
print(f"Median Scan Age in this new dataset is {median_age_three_fifths} and Average Scan Age is {avg_age_three_fifths}.")
# for this new dataset: median scan age is 40.57 and average scan age is 39.61

age_bins_three_fifths = np.linspace(np.min(scan_ages_three_fifths), np.max(scan_ages_three_fifths), 10)              
age_df_three_fifths = pd.DataFrame(data=scan_ages_three_fifths, columns=["scan_ages"])
age_df_three_fifths["bucket"] = pd.cut(age_df_three_fifths.scan_ages, age_bins_three_fifths)

distribution_three_fifths, buckets_three_fifths = np.histogram(scan_ages_three_fifths, age_bins_three_fifths)

#again, chi-square test 
print(chisquare(distribution_three_fifths, f_exp=[i *  426/709 for i in [5, 13, 25, 43, 62, 72, 206, 191, 92]] ))
# statistic=3.1668290051229127, pvalue=0.9234558297462837 --> null hypothesis that they come from the same distribution cannot be "verworfen"

''' save down those ids'''

id_list_three_fifths = [item['id'] for item in three_fifths_dataset]

# with open('three_fifths_of_data.csv','w') as result_file:
#     wr = csv.writer(result_file, dialect='excel')
#     wr.writerow(id_list_three_fifths)

# with open('dHCP_Training/three_fifths_of_data.csv', 'r') as f:
#     id_list_recovered = f.read()

# id_list_recovered = id_list_recovered.split(',')
# print(id_list_recovered, len(id_list_recovered))

'''delete t2 images'''

# we have 833 t2 images. we only need those for which we have a t1 image + label as well
id_list = [item['id'] for item in meta_data_list_dicts]

# with open("id_list.csv", "w") as f:
#     wr = csv.writer(f, dialect='excel')
#     wr.writerow(id_list)

t2_dir = os.path.join(root_dir, 'backup_dHCP/T2w_full')
t1_dir = os.path.join(root_dir, 'backup_dHCP/T1w_full')
label_dir = os.path.join(root_dir, 'backup_dHCP/labels_full')

# print(len(os.listdir(t2_dir)))
# print(len(os.listdir(t1_dir)))
# print(len(os.listdir(label_dir)))

# for t2_img in os.listdir(t2_dir):
#     t2_img_name = t2_img.split('/')[-1]
#     if 'drawem9_dseg' in t2_img_name:
#         path = os.path.join(t2_dir, t2_img)
#         os.remove(path)

    
''' construct three more datasets'''

age_dict = {item['id']: item['scan_age'] for item in meta_data_list_dicts}

age_sorted_ids  = {k: v for k, v in sorted(age_dict.items(), key=lambda item: item[1])}
age_sorted_list = list(age_sorted_ids.keys())

old_age_group = age_sorted_list[-100:]
print(f"Oldest of old: {old_age_group[-1], age_dict[old_age_group[-1]]}") # Oldest of old: ('sub-CC00186BN14_ses-61000', 45.14)
print(f"Youngest of old: {old_age_group[0], age_dict[old_age_group[0]]}") # Youngest of old: ('sub-CC00663XX12_ses-195000', 43.0)
young_age_group = age_sorted_list[:120]
print(f"Oldest of young: {young_age_group[-1], age_dict[young_age_group[-1]]}") #Oldest of young: ('sub-CC00395XX17_ses-121300', 35.71)
print(f"Youngest of young: {young_age_group[0], age_dict[young_age_group[0]]}") #Youngest of young: ('sub-CC00718XX17_ses-210400', 26.71)
print(f"mean of young: {np.mean(sorted(scan_ages)[:120])}, std of young: {np.std(sorted(scan_ages)[:120])}")


young_age_group_shuffled = random.sample(young_age_group, len(young_age_group))


# with open('Old_ids.csv','w') as result_file:
#     wr = csv.writer(result_file, dialect='excel')
#     wr.writerow(old_age_group)

# with open('Young_train_ids.csv','w') as result_file:
#     wr = csv.writer(result_file, dialect='excel')
#     wr.writerow(young_age_group_shuffled[:40])

# with open('Young_test_ids.csv','w') as result_file:
#     wr = csv.writer(result_file, dialect='excel')
#     wr.writerow(young_age_group_shuffled[40:])

old_ages = [age_dict[item] for item in old_age_group]
young_ages_train = [age_dict[item] for item in young_age_group_shuffled[:40]]
young_ages_test = [age_dict[item] for item in young_age_group_shuffled[40:]]

print(f"Youngest of young train: {sorted(young_ages_train)[0]}")
print(f"Oldest of young train: {sorted(young_ages_train)[-1]}") 

print(f"Youngest of young test: {sorted(young_ages_test)[0]}") 
print(f"Oldest of young test: {sorted(young_ages_test)[-1]}") 


# print(f"Mean and std of old ds: {np.mean(old_ages), np.std(old_ages)}") # (43.7002, 0.4672557757802465)
# print(f"Mean and std of young train ds: {np.mean(young_ages_train), np.std(young_ages_train)}") #(33.09675, 2.080483582607659)
# print(f"Mean and std of young test ds: {np.mean(young_ages_test), np.std(young_ages_test)}") #(33.24325, 2.2420279742902407)

''' perform Welch test to check whether train and test come from different populations'''
# print(ttest_ind(young_ages_train, young_ages_test, equal_var=False)) #(statistic=-0.3505889440795571, pvalue=0.7267843453939442)

# with open('dHCP_Training/Old_ids.csv', 'r') as f:
#     old_ids = f.read()

# old_ids = old_ids.strip()
# old_id_list = old_ids.split(',')
# print(old_id_list, len(old_id_list))

scan_ages_transfer = young_ages_train + young_ages_test + old_ages

# d = {'Scan Ages': scan_ages_transfer , 'Group': []}

# for item in scan_ages_transfer:
#   if item in old_ages:
#     d['Group'].append('Old')
#   elif item in young_ages_train:
#     d['Group'].append('Young Train')
#   elif item in young_ages_test:
#     d['Group'].append('Young Test')

# # print(len(d['col2']))
# df = pd.DataFrame(data=d)
# # print(df)

# # df_long = pd.wide_to_long(df, stubnames='hey', i='id', j='bla')
# # print(df_long)

# sns.set_palette("pastel")
# hist_plt = sns.histplot(data=df, x='Scan Ages', hue='Group', bins=25)
# hist_plt.figure.savefig("Transfer_learning_training_data.png")

# d = {'Scan Ages': scan_ages}
# df = pd.DataFrame(data=d)

# sns.set_palette("pastel")
# hist_plt = sns.histplot(data=df, x='Scan Ages', bins=25, kde=True)
# hist_plt.figure.savefig("Histogram_of_Scan_Ages.png")

print(len([item for item in scan_ages if item < 37]))

# transform = LoadImage()

# label_dir = os.path.join(data_dir, 'labels_full')

# label_list = sorted([os.path.join(label_dir, file) for file in os.listdir(label_dir)])

# size_list = []
# counter = 0
# anomaly_count = 0
# for item in label_list:
#     loaded_item = transform(item)
#     size_list.append(loaded_item[0].shape)
#     if loaded_item[0].shape != (217, 290, 290):
#         anomaly_count += anomaly_count
#     counter += 1

# print(anomaly_count)
# print(counter)