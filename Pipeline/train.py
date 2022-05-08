import os
import torch

from DataHandler import TrainCollector
from Logging import ResultsLogger
from monai.data import DataLoader, Dataset, decollate_batch, CacheNTransDataset
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric
from monai.utils import set_determinism
from Plotting import ResultPlotter
from Transforms import create_train_val_transform, post_trans
from typing import Callable, List, Mapping, Optional, Sequence, Tuple, Union
from Utils import get_slices_from_matrix


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
train_transform, val_transform = create_train_val_transform(pixdim, roi_size)

# Create train and val data dicts
train_dict, val_dict = datacollector.create_sets()

#train_ds_hide_labels = CacheNTransDataset(train_dict, transform=train_transform, cache_n_trans=0, cache_dir = cache_dir)
train_ds = Dataset(train_dict, transform=train_transform)
val_ds = Dataset(val_dict, transform=train_transform)

train_loader = DataLoader(train_ds, batch_size=resultlogger.hyperparams['batch_size'], shuffle=True)
val_loader = DataLoader(val_ds, batch_size=resultlogger.hyperparams['batch_size'], shuffle=True)

resultlogger.update_hyperparams(size_training_set = len(train_ds), size_val_set = len(val_ds))
print(f"We have {len(train_ds)} training images and {len(val_ds)} val images")

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
resultlogger.restart_clock()

val_interval = 1
best_metric = -1
best_metric_epoch = -1

for epoch in range(resultlogger.hyperparams['max_epochs']):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{resultlogger.hyperparams['max_epochs']}")
    model.train()
    if resultlogger.mode == 'agePrediction':
        age_epoch_loss = 0
        seg_epoch_loss = 0
    else:
        epoch_loss = 0
    step = 0

    for batch_data in train_loader:
        step += 1
        
        if resultlogger.hyperparams['hide_labels']:
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
            if resultlogger.mode == 'agePrediction':
                outputs, age_pred = model(inputs)
            else:
                outputs = model(inputs)

        image_id = meta_data["id"]
        subj_age = meta_data["scan_age"]
        subj_age = subj_age.tolist()

        # deep supervision makes output tuple
        outputs = torch.unbind(outputs, dim=1) 

        # if mode == agePrediction, compute age loss
        if resultlogger.mode == 'agePrediction':
            age = torch.Tensor(subj_age).to(device)
            age_pred = age_pred.T[0]
            age_loss = age_loss_function(age_pred, age)

            # compute deep supervision loss 
            seg_loss = sum(0.5 ** i * seg_loss_function(output, labels)
                    for i, output in enumerate(outputs))

            # combine both losses
            loss = seg_loss + resultlogger.hyperparams['age_loss_weight'] * age_loss
        
        else:
            # compute deep supervision loss 
            loss = sum(0.5 ** i * loss_function(output, labels)
                    for i, output in enumerate(outputs))

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        if resultlogger.mode == 'agePrediction':
            age_epoch_loss += age_loss.item()
            seg_epoch_loss += seg_loss.item()
            print(
            f"{step}/{len(train_ds) // train_loader.batch_size}"
            f", train_loss: {loss.item():.4f}" 
            f", seg loss: {seg_loss.item():.4f}"
            f", age loss: {age_loss.item():.4f}"
            f", image_id: {image_id}"
            f", subject age: {subj_age}"
            f", age prediction: {age_pred.tolist()}")

            resultlogger.log_age_analysis(loss, seg_loss, age_loss, image_id, subj_age, step, epoch)

        else:
            print(
            f"{step}/{len(train_ds) // train_loader.batch_size}"
            f", train_loss: {loss.item():.4f}" 
            f", image_ids: {image_id}"
            f", subject ages: {subj_age}")

            resultlogger.log_analysis(loss, image_id, subj_age, step, epoch)

    epoch_loss /= step
    resultlogger.results["epoch_loss"].append(epoch_loss)
    
    if resultlogger.mode == 'agePrediction':
        resultlogger.results["seg_epoch_loss"].append(seg_epoch_loss)
        resultlogger.results["age_epoch_loss"].append(age_epoch_loss)

    scheduler.step()    
    print(f"epoch {epoch + 1} average loss: {epoch_loss}")

    # validation step
    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():

            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )

                if resultlogger.mode == 'agePrediction':
                    val_outputs, _ = model(val_inputs)
                else:
                    val_outputs = model(val_inputs)
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                val_labels = val_labels.byte()
                dice_metric(y_pred=val_outputs, y=val_labels)
                dice_metric_batch(y_pred=val_outputs, y=val_labels)

            metric = dice_metric.aggregate().item()
            metric_batch = dice_metric_batch.aggregate()

            resultlogger.log_results(metric, metric_batch)

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
            print(f"current epoch: {epoch + 1} current mean dice: {metric:.4f}")

resultlogger.stop_clock()
resultlogger.save_info()

