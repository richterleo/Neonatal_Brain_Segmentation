import os
import torch

from Hyperparams import transfer_strategies_epochs
from monai.data import  decollate_batch
from Utils import get_slices_from_matrix
from Transforms import post_trans



def run(logger, model, train_loader, device, optimizer, scheduler, dice_metric, dice_metric_batch, 
            val_loader=None, loss_function = None, age_loss_function = None, seg_loss_function = None):
    
    # Training loop
    logger.restart_clock()

    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1

    num_epochs = logger.hyperparams['max_epochs']
    if logger.mode == 'transfer':
        num_epochs = transfer_strategies_epochs[logger.transfer_strategy]

    for epoch in range(num_epochs):
        model.train()
        if logger.mode == 'agePrediction':
            age_epoch_loss = 0
            seg_epoch_loss = 0
        else:
            epoch_loss = 0
        step = 0

        for batch_data in train_loader:
            step += 1
            
            if logger.mode == 'labelBudgeting':
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
                if logger.mode == 'agePrediction':
                    outputs, age_pred = model(inputs)
                else:
                    outputs = model(inputs)

                image_id = meta_data["id"]
                subj_age = meta_data["scan_age"]
                subj_age = subj_age.tolist()

                # deep supervision makes output tuple
                outputs = torch.unbind(outputs, dim=1) 

                # if mode == agePrediction, compute age loss
                if logger.mode == 'agePrediction':
                    age = torch.Tensor(subj_age).to(device)
                    age_pred = age_pred.T[0]
                    age_loss = age_loss_function(age_pred, age)

                    # compute deep supervision loss 
                    seg_loss = sum(0.5 ** i * seg_loss_function(output, labels)
                            for i, output in enumerate(outputs))

                    # combine both losses
                    loss = seg_loss + logger.hyperparams['age_loss_weight'] * age_loss
            
                else:
                    # compute deep supervision loss 
                    loss = sum(0.5 ** i * loss_function(output, labels)
                            for i, output in enumerate(outputs))

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if logger.mode == 'agePrediction':
                age_epoch_loss += age_loss.item()
                seg_epoch_loss += seg_loss.item()
                try:
                    logger.log_age_analysis(loss, seg_loss, age_loss, image_id, subj_age, step, epoch)
                except:
                    print(f"Inference logger does not have log_age_analysis method implemented")

            else:
                try:
                    logger.log_analysis(loss, image_id, subj_age, step, epoch)
                except:
                    print(f"Inference logger does not have log_analysis method implemented")

        epoch_loss /= step
        try:
            logger.results["epoch_loss"].append(epoch_loss)
        except:
            print(f"Inference logger does not log epoch_loss")
        
        if logger.mode == 'agePrediction':
            try:
                logger.results["seg_epoch_loss"].append(seg_epoch_loss)
                logger.results["age_epoch_loss"].append(age_epoch_loss)
            except:
                print(f"Inference logger does not log epoch_loss")

        scheduler.step()    
        print(f"epoch {epoch + 1} average loss: {epoch_loss}")

        # validation step
        if val_loader:
            if (epoch + 1) % val_interval == 0:
                model.eval()
                with torch.no_grad():

                    for val_data in val_loader:
                        val_inputs, val_labels = (
                            val_data["image"].to(device),
                            val_data["label"].to(device),
                        )

                        if logger.mode == 'agePrediction':
                            val_outputs, _ = model(val_inputs)
                        else:
                            val_outputs = model(val_inputs)
                        val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                        val_labels = val_labels.byte()
                        dice_metric(y_pred=val_outputs, y=val_labels)
                        dice_metric_batch(y_pred=val_outputs, y=val_labels)

                    metric = dice_metric.aggregate().item()
                    metric_batch = dice_metric_batch.aggregate()

                    logger.log_results(metric, metric_batch)

                    dice_metric.reset()
                    dice_metric_batch.reset()

                    if metric > best_metric:
                        best_metric = metric
                        best_metric_epoch = epoch + 1

                        torch.save(
                            model.state_dict(),
                            os.path.join(logger.result_dir, "best_metric_model_epoch_" + str(epoch) + ".pth"),
                            )

                        logger.results['best_mean_dice'].append(best_metric)
                        logger.results["best_epoch"].append(best_metric_epoch-1)

    # if we don't have a validation set, just save the model after training
    if not val_loader:
        torch.save(model.state_dict(), os.path.join(logger.result_dir, "best_metric_model_epoch_" + str(epoch) + ".pth"),)