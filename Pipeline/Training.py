import torch

from DataHandler import TrainCollector
from Logging import ResultsLogger
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.utils import set_determinism
from Plotting import ResultPlotter
from Train import run


# Creates new directories for saving down results
resultlogger = ResultsLogger('baseline', session_info='Only for testing')
resultlogger.create_result_folder()

# set random seed for reproducability
set_determinism(seed=resultlogger.meta_info["random_seed"])

# load data into dicts
datacollector = TrainCollector(resultlogger.root_dir, resultlogger.result_dir, resultlogger.hyperparams['hide_labels'])

# Create transforms
pixdim = resultlogger.hyperparams['pixdim']
roi_size = resultlogger.hyperparams['roi_size']
batch_size = resultlogger.hyperparams['batch_size']

if resultlogger.mode == 'transfer':
    train_loader = datacollector.get_loaders(pixdim, roi_size, batch_size) # no validation set for pretraining on older population
elif resultlogger.mode == 'labelBudgeting':
    slicing_mode = resultlogger.hyperparams['slicing_mode']
    selection_mode = resultlogger.hyperparams['selection_mode']
    train_loader, val_loader = datacollector.get_loaders(pixdim, roi_size, batch_size, slicing_mode=slicing_mode, selection_mode=selection_mode)
else:
    train_loader, val_loader = datacollector.get_loaders(pixdim, roi_size, batch_size)

# Create nnU-Net model instance with best hyperparams 
device = torch.device("cuda:0")
model = resultlogger.define_model(resultlogger.hyperparams['kernels'], resultlogger.hyperparams['strides']).to(device)

# define second loss function if we have added age prediction
if resultlogger.mode == 'agePrediction':
    seg_loss_function = DiceCELoss(to_onehot_y=False, sigmoid = True, squared_pred=True) 
    age_loss_function = torch.nn.MSELoss()
else:
    loss_function = DiceCELoss(to_onehot_y=False, sigmoid = True, squared_pred=True) 

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=resultlogger.hyperparams['lr'],
    momentum=0.99,
    weight_decay=3e-5,
    nesterov=True,
)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, 
            lr_lambda=lambda epoch: (1 - epoch / resultlogger.hyperparams['max_epochs']) ** 0.9)
dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

# Training loop
if resultlogger.mode == 'agePrediction':
    run(resultlogger, model, train_loader, device, optimizer, scheduler, dice_metric, dice_metric_batch, val_loader=val_loader,
        age_loss_function=age_loss_function, seg_loss_function=seg_loss_function)
else:
    run(resultlogger, model, train_loader, device, optimizer, scheduler, dice_metric, dice_metric_batch, val_loader=val_loader,
        loss_function=loss_function)

# Save down training metrics
resultlogger.stop_clock()
resultlogger.save_info()


