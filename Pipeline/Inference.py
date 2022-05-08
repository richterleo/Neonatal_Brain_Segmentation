import torch

from DataHandler import TestCollector
from Hyperparams import transfer_strategies_lrs
from Logging import InferenceLogger
from monai.data import decollate_batch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, MSEMetric, MAEMetric
from monai.utils import set_determinism
from pathlib import Path
from Utils import set_parameter_requires_grad
from Transforms import post_trans
from Train import run

# Choose model to evaluate
model_path = Path('results/Age_Segmentation_Dynunet_results1631399790/best_metric_model_epoch_57.pth')
inferencelogger = InferenceLogger('baseline', model_path=model_path, session_info='Only for testing') # if transfer, define strategy

set_determinism(seed=inferencelogger.hyperparams['random_seed'])

# Create (train and) test loaders
testcollector = TestCollector(inferencelogger.mode, inferencelogger.root_dir, inferencelogger.result_dir)
pixdim = inferencelogger.hyperparams['pixdim']
roi_size = inferencelogger.hyperparams['roi_size']
batch_size = inferencelogger.hyperparams['batch_size']

if inferencelogger.mode == 'transfer':
    train_loader, test_loader = inferencelogger.get_loaders(pixdim, roi_size, batch_size)
else:
    test_loader = inferencelogger.get_loaders(pixdim, roi_size, batch_size)


# Load model 
device = torch.device("cuda:0")
model = inferencelogger.define_model(inferencelogger.hyperparams['kernels'], 
                            inferencelogger.hyperparams['strides']).to(device)

if inferencelogger.transfer_strategy == 'new_model':
        pass
else:
    model.load_state_dict(torch.load(inferencelogger.model_path))

# Define metrics for segmentation
dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

# If age prediction, define metrics for age prediction
if inferencelogger.mode == 'agePrediction':
    mse_metric = MSEMetric()
    mae_metric = MAEMetric()


if inferencelogger.mode == 'transfer':

    param_names_to_update, params_to_update = set_parameter_requires_grad(model, 
                                                inferencelogger.transfer_strategy, inferencelogger.model_size)

    loss_function = DiceCELoss(to_onehot_y=False, sigmoid = True, squared_pred=True) 
    optimizer = torch.optim.SGD(
        params_to_update, #only train params for which requires_grad is True
        lr=transfer_strategies_lrs[inferencelogger.transfer_strategy], # smaller learning rate for fine_tuning 
        momentum=0.99,
        weight_decay=3e-5,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, 
                lr_lambda=lambda epoch: (1 - epoch / inferencelogger.hyperparams['max_epochs']) ** 0.9)

    # fine tune model
    if inferencelogger.transfer_strategy not in ['no_finetuning', 'new_model']:
        run(inferencelogger, model, train_loader, device, optimizer, scheduler, dice_metric, dice_metric_batch, loss_function=loss_function)

    # train new model from scratch
    elif inferencelogger.transfer_strategy == 'new_model':
        run(inferencelogger, model, train_loader, device, optimizer, scheduler, dice_metric, dice_metric_batch, loss_function=loss_function)

# evaluate model
model.eval()

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

    inferencelogger.log_tcs(dice_metric.aggregate(), dice_metric_batch.aggregate())
    if inferencelogger.mode == 'agePrediction':
        inferencelogger.log_age_tcs(mse_metric.aggregate(), mae_metric.aggregate())

    dice_metric.reset()
    dice_metric_batch.reset()

inferencelogger.stop_clock()
inferencelogger.save_info()




