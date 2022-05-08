import torch

from DataHandler import TestCollector
from Logging import InferenceLogger
from monai.data import DataLoader, Dataset, decollate_batch
from monai.metrics import DiceMetric, MSEMetric, MAEMetric
from monai.utils import set_determinism
from pathlib import Path
from Transforms import create_test_transform, post_trans

model_path = Path('dHCP_Training/results/Age_Segmentation_Dynunet_results1631399790/best_metric_model_epoch_57.pth')
inferencelogger = InferenceLogger('baseline', model_path, session_info='Only for testing')

set_determinism(seed=inferencelogger.hyperparams['random_seed'])

# Create basic training data dict
datacollector = TestCollector(inferencelogger.root_dir, inferencelogger.result_dir)
test_dict = datacollector.create_sets()

# Define test transforms
test_transform = create_test_transform(inferencelogger.hyperparams['pixdim'])
test_ds = Dataset(test_dict, transform=test_transform)

# Create test loader
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
print(f"Number of test images: {len(test_ds)}")

# Load model that we want to evaluate
device = torch.device("cuda:0")
model = inferencelogger.define_model(inferencelogger.hyperparams['kernels'], 
                            inferencelogger.hyperparams['strides']).to(device)

model.load_state_dict(torch.load(inferencelogger.model_path))

# Define metrics for segmentation
dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

# If age prediction, define metrics for age prediction
if inferencelogger.mode == 'agePrediction':
    mse_metric = MSEMetric()
    mae_metric = MAEMetric()

# evaluate model
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

        if inferencelogger.mode == 'agePrediction':
            test_outputs, age_pred = inferencelogger.infer(test_inputs, model)
            age_pred = torch.unsqueeze(torch.unsqueeze(age_pred, 0),0)
            age = torch.unsqueeze(test_meta_data['scan_age'], 0).cpu()
            mae_metric(y_pred=age_pred, y=age)
            mse_metric(y_pred=age_pred, y=age)
        else:
            test_outputs = inferencelogger.infer(test_inputs, model)
        
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




