import numpy as np
import pandas as pd
import torch

from DataHandler import TrainCollector
from Logging import InferenceLogger
from monai.data import DataLoader, Dataset, decollate_batch
from monai.utils import set_determinism
from monai.data import NiftiSaver
from pathlib import Path
from Transforms import create_save_transform, post_trans

# Specify model for creating sample segmentations
model_path = Path('results/Age_Segmentation_Dynunet_results1631399790/best_metric_model_epoch_57.pth')
inferencelogger = InferenceLogger('baseline', model_path=model_path, session_info='Save segmentations') 

set_determinism(seed=inferencelogger.hyperparams['random_seed'])

# Define transform
save_transform = create_save_transform(inferencelogger.hyperparams['pixdim'])
datacollector = TrainCollector(inferencelogger.root_dir, inferencelogger.result_dir, inferencelogger.mode)

# Age analysis
scan_ages = [item["scan_age"] for item in datacollector.meta_data_list_dicts]

# split age into 3 buckets 
age_bins = np.linspace(np.min(scan_ages), np.max(scan_ages), 4)              
age_df = pd.DataFrame(data=scan_ages, columns=["scan_ages"])
age_df["bucket"] = pd.cut(age_df.scan_ages, age_bins)

distribution, buckets = np.histogram(scan_ages, age_bins)
young_neonates = [item for item in datacollector.data_dict if item['meta_data']['scan_age'] < buckets[1]]
medium_neonates = [item for item in datacollector.data_dict if (item['meta_data']['scan_age'] > buckets[1] and item['meta_data']['scan_age'] < buckets[2])]
old_neonates = [item for item in datacollector.data_dict if item['meta_data']['scan_age'] > buckets[2]]

# choose one example in each age group to evaluate on
young_sample = [young_neonates[0]]
print(f"young sample: id: {young_sample[0]['meta_data']['id']}, age: {young_sample[0]['meta_data']['scan_age']}")

medium_sample = [medium_neonates[0]]
print(f"medium sample: id: {medium_sample[0]['meta_data']['id']}, age: {medium_sample[0]['meta_data']['scan_age']}")

old_sample = [old_neonates[0]]
print(f"old sample: id: {old_sample[0]['meta_data']['id']}, age: {old_sample[0]['meta_data']['scan_age']}")

# Build dataset
save_data = young_sample + medium_sample + old_sample
save_ds = Dataset(save_data, transform=save_transform)
save_loader = DataLoader(save_ds, batch_size=1, shuffle=False)

# Load model
device = torch.device("cuda:0")
model = inferencelogger.define_model(inferencelogger.hyperparams['kernels'], 
                            inferencelogger.hyperparams['strides']).to(device)

model.load_state_dict(torch.load(inferencelogger.model_path))
model.eval()

# class for saving down as .nii files
niftisaver = NiftiSaver(output_dir=inferencelogger.result_dir, output_postfix = 'baseline_seg', separate_folder=False)

with torch.no_grad():
    for test_data in save_loader:
        test_inputs, test_labels = ( # list of len batch_size. item has shape [10, 217, 290, 290]
        test_data["image"].to(device),
        test_data["label"].to(device),
        )
        test_outputs = inferencelogger.infer(test_inputs, model)
        test_outputs = [post_trans(i) for i in decollate_batch(test_outputs)] #list with len batch_size,  [10, 217, 290, 290]

        test_output_comb = torch.argmax(test_outputs[0], dim=0)
        test_output_comb = test_output_comb.unsqueeze(0)

        # save nifti file with accompanying meta data for correct visualisatio
        meta_data = {label_key: test_data['label_meta_dict'][label_key][0] for label_key in test_data['label_meta_dict'].keys()}
        meta_data['filename_or_obj']= test_data['meta_data']['id'][0]
        niftisaver.save(test_output_comb, meta_data)







