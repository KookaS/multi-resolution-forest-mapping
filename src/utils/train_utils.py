from abc import abstractmethod, ABC
import numpy as np
from numpy.core.numeric import NaN
from tqdm import tqdm
import torch
import torch.nn.functional as f
import torch.nn as nn
import math
from dataset.ExpUtils import F_NODATA_VAL, I_NODATA_VAL


def fit(model, device, dataloader, optimizer, n_batches, seg_criterion, seg_criterion_2, lambda_bin = 1.0):
    """
    Runs 1 training epoch
    n_batches is used to setup the progress bar only. The actual number of batches is defined by the dataloader.
    Returns average losses over the batches. Warning: the average assumes the same number of pixels contributed to the
    loss for each batch (unweighted average over batches)
    """
    model.train()
    losses = []
    binary_losses = []
    running_loss = 0.0
    dump_period = 100 # period (in number of batches) for printing running loss

    # train batch by batch
    
    progress_bar = tqdm(enumerate(dataloader), total=n_batches)
    for batch_idx, data in progress_bar:    
        total_loss = torch.tensor([0.], requires_grad=True, device=device)
        inputs, target = data
        inputs = [d.to(device) for d in inputs]
        target = target.to(device) 

        # collect outputs of forward pass
        optimizer.zero_grad()
        final_actv = model(*inputs)
                     
        # segmentation loss(es)
        if seg_criterion_2 is not None: # 2 sub-tasks
            seg_actv, bin_seg_actv = final_actv[:, :-1], final_actv[:, -1]
            seg_target, bin_seg_target = target[:, 0], target[:, 1].float()
            # backpropagate for binary subtask (last channels)
            bin_seg_loss = seg_criterion_2(bin_seg_actv, bin_seg_target)
            total_loss = total_loss + lambda_bin * bin_seg_loss 
        else: # only 1 task
            seg_actv = final_actv
            seg_target = target.squeeze(1)
        # main supervision
        seg_loss = seg_criterion(seg_actv, seg_target)
        total_loss = total_loss + seg_loss

        # backward pass
        total_loss.backward()
        optimizer.step()

        # store current losses
        losses.append(seg_loss.item())
        if seg_criterion_2 is not None:
            binary_losses.append(bin_seg_loss.item())

        
        running_loss += total_loss.item() 
        # print running loss
        if batch_idx % dump_period == dump_period - 1: 
            # this is an approximation because each patch has a different number of valid pixels
            progress_bar.set_postfix(loss=running_loss/dump_period)
            running_loss = 0.0

    # average losses over the epoch
    avg_loss = np.mean(losses, axis = 0)
    avg_binary_loss = NaN if seg_criterion_2 is None else np.mean(binary_losses)

    return avg_loss, avg_binary_loss


    
        


