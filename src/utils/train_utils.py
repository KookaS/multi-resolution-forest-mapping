from abc import abstractmethod, ABC
import numpy as np
from numpy.core.numeric import NaN
from tqdm import tqdm
import torch
import torch.nn.functional as f
import torch.nn as nn
import math
from dataset.ExpUtils import F_NODATA_VAL, I_NODATA_VAL
from data.generate_data import generate_simulated_image



def fit(model, device, dataloader, optimizer, n_batches, seg_criterion, seg_criterion_2, lambda_bin = 1.0, lambda_feature = 1.0):
    """
    Runs 1 training epoch
    n_batches is used to setup the progress bar only. The actual number of batches is defined by the dataloader.
    Returns average losses over the batches. Warning: the average assumes the same number of pixels contributed to the
    loss for each batch (unweighted average over batches)
    """
    model.train()
    losses = []
    binary_losses = []
    losses_sim = []
    binary_losses_sim = []
    losses_feature = []
    running_loss = 0.0
    dump_period = 100 # period (in number of batches) for printing running loss

    # train batch by batch
    progress_bar = tqdm(enumerate(dataloader), total=n_batches)
    for batch_idx, data in progress_bar:    
        total_loss = torch.tensor([0.], requires_grad=True, device=device)
        inputs, target = data
        inputs_sim = []
        for input in inputs:
            if (input.shape[1] == 3):
                temp = generate_simulated_image(img=input.clone())
                inputs_sim.append(temp)

        inputs = [d.to(device) for d in inputs]
        inputs_sim = [d.to(device) for d in inputs_sim]
        target = target.to(device)

        # collect outputs of forward pass
        optimizer.zero_grad()
        feature_space = model.encode(*inputs, sim=False)
        feature_space_sim = model.encode(*inputs_sim, sim=True)

        final_actv = model.decode(*feature_space)
        final_actv_sim = model.decode(*feature_space_sim)

        # RMSE of feature spaces
        feature_criterion = nn.MSELoss()
        feature_loss=0
        # first output in feature space does not match dimensions torch.Size([8, 64, 256, 256]) vs torch.Size([8, 64, 128, 128])
        # TODO [1:] or [-1]
        for (f, fs) in zip(feature_space[-1], feature_space_sim[-1]):
            feature_loss += torch.sqrt(feature_criterion(f, fs))
        total_loss = total_loss + lambda_feature * feature_loss
        losses_feature.append(feature_loss.item())
                     
        # segmentation loss(es)
        if seg_criterion_2 is not None: # 2 sub-tasks
            seg_actv, bin_seg_actv = final_actv[:, :-1], final_actv[:, -1]
            seg_actv_sim, bin_seg_actv_sim = final_actv_sim[:, :-1], final_actv_sim[:, -1]
            seg_target, bin_seg_target = target[:, 0], target[:, 1].float()
            # backpropagate for binary subtask (last channels)
            bin_seg_loss = seg_criterion_2(bin_seg_actv, bin_seg_target)
            bin_seg_loss_sim = seg_criterion_2(bin_seg_actv_sim, bin_seg_target)
            total_loss = total_loss + lambda_bin * bin_seg_loss + lambda_bin * bin_seg_loss_sim 
        else: # only 1 task
            seg_actv = final_actv
            seg_actv_sim = final_actv_sim
            seg_target = target.squeeze(1)
        # main supervision
        seg_loss = seg_criterion(seg_actv, seg_target)
        seg_loss_sim = seg_criterion(seg_actv_sim, seg_target)
        total_loss = total_loss + seg_loss + seg_loss_sim

        # backward pass
        total_loss.backward()
        optimizer.step()

        # store current losses
        losses.append(seg_loss.item())
        losses_sim.append(seg_loss_sim.item())
        if seg_criterion_2 is not None:
            binary_losses.append(bin_seg_loss.item())
            binary_losses_sim.append(bin_seg_loss_sim.item())

        
        running_loss += total_loss.item() 
        # print running loss
        if batch_idx % dump_period == dump_period - 1: 
            # this is an approximation because each patch has a different number of valid pixels
            progress_bar.set_postfix(loss=running_loss/dump_period)
            running_loss = 0.0

    # average losses over the epoch
    avg_loss = np.mean(losses, axis = 0)
    avg_binary_loss = NaN if seg_criterion_2 is None else np.mean(binary_losses)
    avg_loss_sim = np.mean(losses_sim, axis = 0)
    avg_binary_loss_sim = NaN if seg_criterion_2 is None else np.mean(binary_losses_sim)
    avg_loss_feature = np.mean(losses_feature, axis = 0)

    return (avg_loss, avg_binary_loss), (avg_loss_sim, avg_binary_loss_sim), avg_loss_feature
    
        


