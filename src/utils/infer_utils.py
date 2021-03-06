import os
import sys
from numpy.core.numerictypes import english_lower
import rasterio
from rasterio.windows import Window
import time
import shutil
import random
from osgeo import gdal
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from .eval_utils import cm2rates, rates2metrics, my_confusion_matrix, get_seg_error_map
from dataset import ExpUtils, InferenceDataset
from dataset.ExpUtils import I_NODATA_VAL, F_NODATA_VAL
from .write_utils import Writer
import random
from tqdm import tqdm
import gc
from data.generate_data import generate_simulated_image


class Inference():
    """
    Class to perform inference and evaluate predictions on a set of samples. If used for validation during training, 
    the class must be instantiated once before training and Inference.infer() can be called at each epoch.
    Virtual mosaics with all the tiles for each source are created so that the Dataset can sample patches that overlap 
    several neighboring tiles. If they exist, the nodata values of the inputs rasters are used to fill the gaps, 
    otherwise a new nodata value is introduced depending on the raster data type. When calling infer(), the criteria 
    ignore_index attributes are modified accordingly.
    """
    def __init__(self, model, file_list, exp_utils, output_dir = None, 
                        evaluate = True, save_hard = True, save_soft = True, save_error_map = False,
                        batch_size = 32, num_workers = 0, device = 0, undersample = 1, decision = 'f', compare_dates=False):

        """
        Args:
            - model (nn.Module): model to perform inference with
            - file_list (str): csv file containing the files to perform inference on (1 sample per row)
            - exp_utils (ExpUtils): object containing information of the experiment/dataset
            - output_dir (str): directory where to write output files
            - evaluate (bool): whether to evaluate the predictions
            - save_hard (bool): whether to write hard predictions into image files
            - save_soft (bool): whether to write soft predictions into image files
            - batch_size (int): batch size
            - num_workers (int): number of workers to use for the DataLoader. Recommended value is 0 because the tiles
                are processed one by one
            - device (torch.device): device to use to perform inference 
            - undersample (int): undersampling factor to reduction the size of the dataset. Example: if undersample = 100, 
                1/100th of the dataset's samples will be randomly picked to perform inference on.
        """

        self.evaluate = evaluate  
        self.save_hard = save_hard
        self.save_soft = save_soft
        self.model = model
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.undersample = undersample
        self.exp_utils = exp_utils
        self.binary_map = self.exp_utils.decision_func_2 is not None # generate binary map
        self.patch_size = self.exp_utils.patch_size
        self.decision = decision
        self.input_vrt_fn = None # used to indicate that virtual raster mosaic(s) has not been created yet
        self.save_error_map = save_error_map
        self.compare_dates = compare_dates

        self.n_inputs = self.exp_utils.n_input_sources

        # create a temporary directory to save virtual raster mosaics
        self.tmp_dir = 'tmp'
        i = 0
        while os.path.isdir(self.tmp_dir):
            i += 1
            self.tmp_dir = 'tmp_{}'.format(i)
        os.mkdir(self.tmp_dir)           
        # create the column strings to read the dataframe
        self._get_col_names()

        file_list_ext = os.path.splitext(file_list)[-1]
        if file_list_ext == '.csv':
            df = pd.read_csv(file_list)
        else:
            raise ValueError('file_list should be a csv file ("*.csv")')
        self.n_samples = len(df)
        self._fns_df = df # self._fns_df should not be modified

        # Define predictions averaging kernel
        self.kernel = torch.from_numpy(self.exp_utils.get_inference_kernel()) #.to(device)

        # Define output normalization (logits -> probabilities) functions
        if self.decision == 'f':
            self.seg_normalization = nn.Softmax(dim = 1)
        else:
            self.seg_normalization = self._normalize_hierarchical_output
        # Initialize cumulative confusion matrix
        self.cum_cms = {}
        if self.evaluate:
            self.cum_cms['seg'] = np.empty((self.exp_utils.n_classes,) * 2)
            self.cum_cms['seg_sim'] = np.empty((self.exp_utils.n_classes,) * 2)
            if self.binary_map:
                self.cum_cms['seg_2'] = np.empty((self.exp_utils.n_classes_2,) * 2)
                self.cum_cms['seg_2_sim'] = np.empty((self.exp_utils.n_classes_2,) * 2)
                if self.decision == 'h':
                    self.cum_cms['seg_1'] = np.empty((self.exp_utils.n_classes_1,) * 2)
                    self.cum_cms['seg_1_sim'] = np.empty((self.exp_utils.n_classes_1,) * 2)
                      

    def _normalize_hierarchical_output(self, output):
        output_1 = output[:, :self.exp_utils.n_classes_1]
        output_2 = output[:, self.exp_utils.n_classes_1:]
        norm_output_1 = torch.softmax(output_1, dim = 1)
        norm_output_2 = torch.sigmoid(output_2)
        return torch.cat((norm_output_1, norm_output_2), dim = 1)
        
    def _get_col_names(self):
        """Get the column names used to read the dataset dataframe"""
        if self.n_inputs < 2:
            self.input_col_names = ['input']
        else:
            self.input_col_names = ['input_' + str(i) for i in range(self.n_inputs)]
    
    def _get_vrt_from_df(self, df):
        """Build virtual mosaic rasters from files listed in dataframe df"""
        #### inputs ###########################################################

        keys = {key:1 for key in self.exp_utils.input_channels.keys()}
        if 'ALTI' in self.exp_utils.input_channels.keys():
            keys['SI2017'] = 2
            del keys['ALTI']

        self.input_vrt_fns = {input:[None]*keys[input] for input in self.exp_utils.input_channels.keys()}
        self.input_vrt_nodata_val = {input:[None]*keys[input] for input in self.exp_utils.input_channels.keys()}
        for i, col_name in enumerate(self.input_col_names):
            fns = df[col_name]
            vrt_fn = os.path.join(self.tmp_dir, '{}.vrt'.format(col_name))
            key = [input for input in self.exp_utils.input_channels.keys() if input[2:] in fns[0]][0]
            if self.exp_utils.input_nodata_val[i] is None:
                # read the first tile just to know the data type:
                with rasterio.open(fns[0], 'r') as f_tile:
                    dtype = f_tile.profile['dtype']
                if dtype == 'uint8':
                    self.input_vrt_nodata_val[key] = [I_NODATA_VAL]
                elif dtype.startswith('uint'):
                    self.input_vrt_nodata_val[key] = [I_NODATA_VAL]
                    print('WARNING: nodata value for {} set to {}'.format(col_name, I_NODATA_VAL))
                else:
                    # the min and max float32 values are not handled by GDAL, using value -1 instead
                    self.input_vrt_nodata_val[key] = [F_NODATA_VAL]
                    print('WARNING: nodata value for {} set to {}'.format(col_name, F_NODATA_VAL)) 
            else:
                self.input_vrt_nodata_val[key] = self.exp_utils.input_nodata_val[i]

            gdal.BuildVRT(  vrt_fn, 
                            list(fns),
                            VRTNodata=self.input_vrt_nodata_val[key][0],
                            options = ['overwrite']) 
            self.input_vrt_fns[key] = [vrt_fn]
        
        self.target_vrt_fn = None
        self.target_vrt_nodata_val = None
        if self.evaluate or self.save_error_map:
            #### main target ##################################################
            fns = df['target']  
            self.target_vrt_fn = os.path.join(self.tmp_dir, 'target.vrt') 
            if self.exp_utils.target_nodata_val is None:
                # read the tile just to know the data type:
                with rasterio.open(fns[0], 'r') as f_tile:
                    dtype = f_tile.profile['dtype']
                if dtype == 'uint8':
                    self.target_vrt_nodata_val = I_NODATA_VAL
                else:
                    raise ValueError('The main target should be of type uint8, found {} instead'.format(dtype))
            else:
                self.target_vrt_nodata_val = self.exp_utils.target_nodata_val
            gdal.BuildVRT(  self.target_vrt_fn, 
                            list(fns),
                            VRTNodata=self.target_vrt_nodata_val,
                            options = ['overwrite'])
            

    def _select_samples(self):
        """Select samples to perform inference on"""
        # use a random subset of the data
        idx = random.sample(range(self.n_samples), self.n_samples//self.undersample)
        df = self._fns_df.iloc[idx]
        return df.reset_index(drop = True)

    def _reset_cm(self):
        """Reset the confusion matrix/matrices with zeros"""
        if self.evaluate:
            for key in self.cum_cms:
                self.cum_cms[key].fill(0)
                

    def _get_decisions(self, actv, actv_sim, target_data):
        """Obtain decisions from soft outputs (argmax) and update confusion matrix/matrices"""
        # define main and binary outputs/targets and compute hard predictions
        if self.decision == 'f':
            # define the outputs 
            output = actv
            output_2 = actv
            output_sim = actv_sim
            output_2_sim = actv_sim
            # compute hard predictions
            output_hard = self.exp_utils.decision_func(output)
            output_hard_1 = None
            output_hard_2 = None
            output_hard_sim = self.exp_utils.decision_func(output_sim)
            output_hard_1_sim = None
            output_hard_2_sim = None
            # define the targets 
            if self.evaluate: 
                target = target_data
                if self.binary_map:
                    # get naive binary output
                    target_2 = self.exp_utils.target_recombination(target_data)
                    #if self.exp_utils.decision_func_2 is not None:
                    output_hard_2 = self.exp_utils.decision_func_2(output_2)
                    output_hard_2_sim = self.exp_utils.decision_func_2(output_2_sim)
         
        else:
            # define the outputs 
            output_1 = actv[:-1]
            output_2 = actv[-1]
            output_1_sim = actv_sim[:-1]
            output_2_sim = actv_sim[-1]
            # compute hard predictions
            output_hard_1 = self.exp_utils.decision_func(output_1) # ForestType
            output_hard_2 = self.exp_utils.decision_func_2(output_2) # PresenceOfForest
            output_hard = (output_hard_1 + 1) * output_hard_2 # apply decision tree -> TLM4c
            output_hard_1_sim = self.exp_utils.decision_func(output_1_sim) # ForestType
            output_hard_2_sim = self.exp_utils.decision_func_2(output_2_sim) # PresenceOfForest
            output_hard_sim = (output_hard_1_sim + 1) * output_hard_2_sim # apply decision tree -> TLM4c
            # define the targets
            if self.evaluate or self.save_error_map:
                target = target_data[-1] # TLM4c
                target_1 = target_data[0] # ForestType
                if output_hard_2 is not None:
                    target_2 = target_data[1] # PresenceOfForest

                    
                    
                
        ########## update confusion matrices #########
        # main task
        if self.evaluate:
            self.cum_cms['seg']+= my_confusion_matrix(target, 
                                                     output_hard,
                                                     self.exp_utils.n_classes)
            self.cum_cms['seg_sim']+= my_confusion_matrix(target, 
                                                     output_hard_sim,
                                                     self.exp_utils.n_classes)
            
            # other tasks / output
            if self.binary_map:
                self.cum_cms['seg_2'] += my_confusion_matrix(
                                            target_2, 
                                            output_hard_2, self.exp_utils.n_classes_2)
                self.cum_cms['seg_2_sim'] += my_confusion_matrix(
                                            target_2, 
                                            output_hard_2_sim, self.exp_utils.n_classes_2)
                if self.decision == 'h':
                    if self.evaluate:
                        self.cum_cms['seg_1'] += my_confusion_matrix(
                                                target_1, 
                                                output_hard_1, self.exp_utils.n_classes_1)
                        self.cum_cms['seg_1_sim'] += my_confusion_matrix(
                                                target_1, 
                                                output_hard_1_sim, self.exp_utils.n_classes_1)


        return (output_hard, output_hard_2, output_hard_1), (output_hard_sim, output_hard_2_sim, output_hard_1_sim)

    def _compute_metrics(self):
        """Compute classification metrics from confusion matrices"""
        reports = {}
        for key in self.cum_cms:
            reports[key] = rates2metrics(cm2rates(self.cum_cms[key]), self.exp_utils.class_names[key])
        return reports

    def _infer_sample(self, data, coords, dims, margins, 
                      seg_criterion = None, seg_criterion_2 = None, data_low_res = None):
        """Performs inference on one (multi-source) input accessed through dataset ds, with multiple outputs."""

        # compute dimensions of the output
        s = self.exp_utils.target_scale
        height, width = (d * s for d in dims)
        top_margin, left_margin, bottom_margin, right_margin = (m*s for m in margins)

        # initialize accumulators
        output = torch.zeros((self.exp_utils.output_channels, height, width), dtype=torch.float32)
        output_sim = torch.zeros((self.exp_utils.output_channels, height, width), dtype=torch.float32)
        counts = torch.zeros((height, width), dtype=torch.float32)

        inputs, targets = data
        num_batches = len(inputs[0])
        if self.evaluate:
            feature_losses = [0] * num_batches
            if seg_criterion is not None:
                seg_losses = [0] * num_batches
                seg_losses_sim = [0] * num_batches
                valid_px_list = [0] * num_batches
            if seg_criterion_2 is not None:
                seg_bin_losses = [0] * num_batches
                seg_bin_losses_sim = [0] * num_batches
                valid_px_bin_list = [0] * num_batches
        # iterate over batches of small patches
        for batch_idx in range(num_batches):
            # get the prediction for the batch
            if data_low_res is not None:
                # lower resolution data given, like 1946 iamges
                input_data = [data[batch_idx].to(self.device) for data in inputs]
                inputs_low_res, _ = data_low_res
                input_data_sim = [data[batch_idx].to(self.device) for data in inputs_low_res]
            else:
                # lower resolution data generated from 2017
                input_data = [data[batch_idx].to(self.device) for data in inputs]
                input_data_sim = []
                for data in inputs:
                    if (data[batch_idx].shape[1] == 3):
                        input_data_sim.append(generate_simulated_image(data[batch_idx]).to(self.device))
            
            if targets is not None:
                target_data = targets[batch_idx].to(self.device) 

            with torch.no_grad():
                # forward pass
                feature_space = self.model.encode(*input_data, sim=False)
                feature_space_sim = self.model.encode(*input_data_sim, sim=True)
                t_main_actv = self.model.decode(*feature_space)
                t_main_actv_sim = self.model.decode(*feature_space_sim)

                # compute validation losses
                if self.evaluate:
                    # RMSE of feature spaces
                    feature_criterion = nn.MSELoss()
                    feature_loss=0
                    # match the last element of the feature space
                    # first output in feature space does not match dimensions torch.Size([8, 64, 256, 256]) vs torch.Size([8, 64, 128, 128])
                    for (f, fs) in zip(feature_space[-1], feature_space_sim[-1]):
                        feature_loss += torch.sqrt(feature_criterion(f, fs))
                    feature_losses[batch_idx] = feature_loss.item()

                    if seg_criterion is not None:
                        if seg_criterion_2 is not None:
                            seg_actv, bin_seg_actv = t_main_actv[:, :-1], t_main_actv[:, -1]
                            seg_actv_sim, bin_seg_actv_sim = t_main_actv_sim[:, :-1], t_main_actv_sim[:, -1]
                            seg_target, bin_seg_target = target_data[:, 0], target_data[:, 1].float() # BCE loss needs float
                            # compute validation loss for binary subtask (last two channels)
                            bin_seg_mask = bin_seg_target != self.target_vrt_nodata_val # custom ignore_index
                            seg_bin_losses[batch_idx] = seg_criterion_2(bin_seg_actv[bin_seg_mask], bin_seg_target[bin_seg_mask]).item()
                            seg_bin_losses_sim[batch_idx] = seg_criterion_2(bin_seg_actv_sim[bin_seg_mask], bin_seg_target[bin_seg_mask]).item()
                            valid_px_bin_list[batch_idx] = torch.sum(bin_seg_mask).item()
                        else:
                            seg_actv = t_main_actv
                            seg_actv_sim = t_main_actv_sim
                            seg_target = target_data #.squeeze(1)

                        # main loss
                        seg_mask = seg_target != seg_criterion.ignore_index
                        loss = seg_criterion(seg_actv, seg_target)
                        if torch.isnan(loss): 
                            seg_losses[batch_idx] = 0
                        else:
                            seg_losses[batch_idx] = loss.item()

                        loss = seg_criterion(seg_actv_sim, seg_target)
                        if torch.isnan(loss): 
                            seg_losses_sim[batch_idx] = 0
                        else:
                            seg_losses_sim[batch_idx] = loss.item()
                        valid_px_list[batch_idx] = torch.sum(seg_mask).item()
                        
                # move predictions to cpu
                main_pred = self.seg_normalization(t_main_actv).cpu()
                main_pred_sim = self.seg_normalization(t_main_actv_sim).cpu()
            # accumulate the batch predictions
            for j in range(main_pred.shape[0]):
                x, y =  coords[batch_idx][j]
                x_start, x_stop = x*s, (x+self.patch_size)*s
                y_start, y_stop = y*s, (y+self.patch_size)*s
                counts[x_start:x_stop, y_start:y_stop] += self.kernel
                output[:, x_start:x_stop, y_start:y_stop] += main_pred[j] * self.kernel
                output_sim[:, x_start:x_stop, y_start:y_stop] += main_pred_sim[j] * self.kernel
                
        # normalize the accumulated predictions
        counts = torch.unsqueeze(counts, dim = 0)
        mask = counts != 0

        rep_mask = mask.expand(output.shape[0], -1, -1)
        rep_counts = counts.expand(output.shape[0], -1, -1)
        output[rep_mask] = output[rep_mask] / rep_counts[rep_mask]
        output_sim[rep_mask] = output_sim[rep_mask] / rep_counts[rep_mask]
        
        # aggregate losses
        if self.evaluate:
            feature_loss = np.average(feature_losses, axis = 0)
            if seg_criterion is None:
                seg_loss, total_valid_px = None, None
                seg_loss_sim, total_valid_px_sim = None, None
            else:
                seg_loss, total_valid_px = self._aggregate_batch_losses(seg_losses, 
                                                                        valid_px_list)
                seg_loss_sim, total_valid_px_sim = self._aggregate_batch_losses(seg_losses_sim, 
                                                                        valid_px_list)
            if seg_criterion_2 is None:
                seg_bin_loss, total_valid_bin_px = None, None
                seg_bin_loss_sim, total_valid_bin_px_sim = None, None
            else:
                seg_bin_loss, total_valid_bin_px = self._aggregate_batch_losses(seg_bin_losses, 
                                                                                valid_px_bin_list)
                seg_bin_loss_sim, total_valid_bin_px_sim = self._aggregate_batch_losses(seg_bin_losses_sim, 
                                                                                valid_px_bin_list)
        else:
            seg_loss, total_valid_px = None, None 
            seg_bin_loss, total_valid_bin_px = None, None
            seg_loss_sim, total_valid_px_sim = None, None
            seg_bin_loss_sim, total_valid_bin_px_sim = None, None
            feature_loss = None
        # remove margins
        output = output[:, top_margin:height-bottom_margin, left_margin:width-right_margin]
        output_sim = output_sim[:, top_margin:height-bottom_margin, left_margin:width-right_margin]
        
        return (output, output_sim), ((seg_loss, total_valid_px), (seg_bin_loss, total_valid_bin_px), 
                (seg_loss_sim, total_valid_px_sim), (seg_bin_loss_sim, total_valid_bin_px_sim), feature_loss)
             
    def _aggregate_batch_losses(self, loss_list, valid_px_list):
        total_valid_px = sum(valid_px_list)
        if total_valid_px > 0:
            seg_loss = np.average(loss_list, axis = 0, weights = valid_px_list)
        else:
            seg_loss = 0
        return seg_loss, total_valid_px

    def infer(self, seg_criterion = None, seg_criterion_2 = None):
        """
        Perform tile by tile inference on a dataset, evaluate and save outputs if needed

        Args:
            - criterion (nn.Module): criterion used for training, to be evaluated at validation as well to track 
                    overfitting
        """
        self.model.eval()
        
        if self.undersample > 1 or self.input_vrt_fn is None:
            # select sample to perform inference on
            df = self._select_samples()
            # create virtual mosaics (and set nodata values)
            self._get_vrt_from_df(df)
        # set the cumulative confusion matrix to 0
        if self.evaluate:
            self._reset_cm()
            feature_losses = [0] * len(df)
            if seg_criterion is not None:
                seg_losses = [0] * len(df)
                valid_px_list = [0] * len(df)
                seg_losses_sim = [0] * len(df)
                valid_px_list_sim = [0] * len(df)
            if seg_criterion_2 is not None:
                seg_bin_losses = [0] * len(df)
                valid_px_bin_list = [0] * len(df)
                seg_bin_losses_sim = [0] * len(df)
                valid_px_bin_list_sim = [0] * len(df)
                
        #create dataset
        target_keys = self.exp_utils.target_scale
        iteration_keys = [key for key in self.exp_utils.input_channels.keys() if key not in ['ALTI']]
        if len(iteration_keys) < 1:
            raise ValueError('Not enough inputs valid for dataloards iterations')

        ds = {input: InferenceDataset(self.input_vrt_fns[input], 
                              exp_utils=self.exp_utils, 
                              batch_size = self.batch_size,
                              target_vrt_fn = self.target_vrt_fn,
                              input_nodata_val = self.input_vrt_nodata_val[input],
                              target_nodata_val = self.target_vrt_nodata_val,
                              input_keys = [input],
                              target_keys=[target_keys]) for input in iteration_keys}

        dataloaders = [torch.utils.data.DataLoader(
            ds[input],
            batch_size=None, # manual batching to obtain batches with patches from the same image
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn = lambda x : x
        ) for input in iteration_keys]
        # iterate over dataset (tile by tile)

        progress_bar = tqdm(zip(df.iterrows(), *dataloaders), total=len(df))
        for progress_elements in progress_bar:
            (tile_idx, fns), *dataloaders_data = progress_elements
            template_fn = fns.iloc[0]
            tile_num = self.exp_utils.tilenum_extractor[0](template_fn)
            progress_bar.set_postfix_str('Tiles(s): {}'.format(tile_num))

            # compute forward pass and aggregate outputs for each input type
            batch_data, batch_data_sim, target_data, coords, dims, margins, input_nodata_mask = None, None, None, None, None, None, None
            for i, dataloader_data in enumerate(dataloaders_data):
                if iteration_keys[i] == 'SI2017':
                    batch_data, target_data, coords, dims, margins, input_nodata_mask = dataloader_data
                if iteration_keys[i] == 'SI1946':
                    batch_data_sim, target_data, coords, dims, margins, input_nodata_mask = dataloader_data
            outputs, losses  = self._infer_sample(batch_data, coords, dims, margins, 
                                                seg_criterion=seg_criterion, 
                                                seg_criterion_2=seg_criterion_2,
                                                data_low_res = batch_data_sim)
            output, output_sim = outputs
            (seg_loss, valid_px), (seg_bin_loss, valid_bin_px), (seg_loss_sim, valid_px_sim), (seg_bin_loss_sim, valid_bin_px_sim), feature_loss = losses
            # store validation losses
            if self.evaluate:
                feature_losses[tile_idx] = feature_loss
                if seg_criterion is not None:
                    seg_losses[tile_idx] = seg_loss
                    valid_px_list[tile_idx] = valid_px
                    seg_losses_sim[tile_idx] = seg_loss_sim
                    valid_px_list_sim[tile_idx] = valid_px_sim
                if seg_criterion_2 is not None:
                    seg_bin_losses[tile_idx] = seg_bin_loss
                    valid_px_bin_list[tile_idx] = valid_bin_px
                    seg_bin_losses_sim[tile_idx] = seg_bin_loss_sim
                    valid_px_bin_list_sim[tile_idx] = valid_bin_px_sim

            # compute hard predictions and update confusion matrix
            output = output.numpy()
            output_sim = output_sim.numpy()
            hard, hard_sim = self._get_decisions(actv=output, 
                                                    actv_sim=output_sim,
                                                    target_data=target_data)
            output_hard, output_hard_2, output_hard_1 = hard
            output_hard_sim, output_hard_2_sim, output_hard_1_sim = hard_sim

            
            # restore nodata values found in the inputs
            if np.any(input_nodata_mask):
                rep_mask = np.repeat(input_nodata_mask[np.newaxis, :, :], output.shape[0], axis = 0)
                output[rep_mask] = self.exp_utils.f_out_nodata_val
                output_hard[input_nodata_mask] = self.exp_utils.i_out_nodata_val
                output_sim[rep_mask] = self.exp_utils.f_out_nodata_val
                output_hard_sim[input_nodata_mask] = self.exp_utils.i_out_nodata_val
                if output_hard_1 is not None:
                    output_hard_1[input_nodata_mask] = self.exp_utils.i_out_nodata_val
                if output_hard_2 is not None:
                    output_hard_2[input_nodata_mask] = self.exp_utils.i_out_nodata_val
                if output_hard_1_sim is not None:
                    output_hard_1_sim[input_nodata_mask] = self.exp_utils.i_out_nodata_val
                if output_hard_2_sim is not None:
                    output_hard_2_sim[input_nodata_mask] = self.exp_utils.i_out_nodata_val
            if self.save_error_map: 
                valid_mask = ~input_nodata_mask
                if self.decision == 'f':
                    main_target = target_data
                    valid_mask *= (main_target != self.target_vrt_nodata_val)# * ~input_nodata_mask
                    seg_error_map = get_seg_error_map(pred=output_hard, 
                                                    target=main_target, 
                                                    valid_mask=valid_mask, 
                                                    n_classes=self.exp_utils.n_classes)
                    seg_error_map_sim = get_seg_error_map(pred=output_hard_sim, 
                                                    target=main_target, 
                                                    valid_mask=valid_mask, 
                                                    n_classes=self.exp_utils.n_classes)
                else:
                    seg_error_map_1 = get_seg_error_map(pred=output_hard_1, 
                                                    target=target_data[0], 
                                                    valid_mask=valid_mask*(target_data[0]!=self.target_vrt_nodata_val), 
                                                    n_classes=self.exp_utils.n_classes_1)
                    seg_error_map_2 = get_seg_error_map(pred=output_hard_2, 
                                                    target=target_data[1], 
                                                    valid_mask=valid_mask*(target_data[1]!=self.target_vrt_nodata_val), 
                                                    n_classes=self.exp_utils.n_classes_2)
                    seg_error_map_1_sim = get_seg_error_map(pred=output_hard_1_sim, 
                                                    target=target_data[0], 
                                                    valid_mask=valid_mask*(target_data[0]!=self.target_vrt_nodata_val), 
                                                    n_classes=self.exp_utils.n_classes_1)
                    seg_error_map_2_sim = get_seg_error_map(pred=output_hard_2_sim, 
                                                    target=target_data[1], 
                                                    valid_mask=valid_mask*(target_data[1]!=self.target_vrt_nodata_val), 
                                                    n_classes=self.exp_utils.n_classes_2)
                    # 0: no error, 1: forest type error, 2: presence of forest error, 3: both errors
                    seg_error_map = (seg_error_map_1>0).astype(np.uint8)
                    seg_error_map[seg_error_map_2>0] += 2
                    seg_error_map_sim = (seg_error_map_1_sim>0).astype(np.uint8)
                    seg_error_map_sim[seg_error_map_2_sim>0] += 2

            # write outputs 
            if self.save_hard or self.save_soft:
                writer = Writer(self.exp_utils, tile_num, template_fn, 
                                template_scale = self.exp_utils.input_scales['SI2017'], 
                                dest_scale=self.exp_utils.target_scale)
                # main segmentation output
                writer.save_seg_result(self.output_dir, 
                                        save_hard = self.save_hard, output_hard = output_hard, 
                                        save_soft = self.save_soft, output_soft = output, 
                                        colormap = self.exp_utils.colormap)
                writer.save_seg_result(self.output_dir, 
                                        save_hard = self.save_hard, output_hard = output_hard_sim, 
                                        save_soft = self.save_soft, output_soft = output_sim, 
                                        colormap = self.exp_utils.colormap,
                                        suffix = '_sim')
                if self.binary_map:
                    # binary forest/non-forest
                    writer.save_seg_result(self.output_dir, 
                                            save_hard = self.save_hard, output_hard = output_hard_2, 
                                            save_soft = False, output_soft = None, 
                                            suffix = self.exp_utils.suffix_2, 
                                            colormap = self.exp_utils.colormap_2)
                    writer.save_seg_result(self.output_dir, 
                                            save_hard = self.save_hard, output_hard = output_hard_2_sim, 
                                            save_soft = False, output_soft = None, 
                                            suffix = self.exp_utils.suffix_2 + "_sim", 
                                            colormap = self.exp_utils.colormap_2)
                    
                    if self.decision == 'h':
                        # forest type
                        writer.save_seg_result(self.output_dir, 
                                                save_hard = self.save_hard, output_hard = output_hard_1, 
                                                save_soft = False, output_soft = None, 
                                                suffix = self.exp_utils.suffix_1, 
                                                colormap = self.exp_utils.colormap_1)
                        writer.save_seg_result(self.output_dir, 
                                                save_hard = self.save_hard, output_hard = output_hard_1_sim, 
                                                save_soft = False, output_soft = None, 
                                                suffix = self.exp_utils.suffix_1 + "_sim", 
                                                colormap = self.exp_utils.colormap_1)
                        if self.save_error_map:
                            writer.save_seg_result(self.output_dir, 
                                                save_hard = self.save_hard, output_hard = seg_error_map_1, 
                                                save_soft = False, output_soft = None, 
                                                suffix = '_error_1', 
                                                colormap = None)
                            writer.save_seg_result(self.output_dir, 
                                                save_hard = self.save_hard, output_hard = seg_error_map_2, 
                                                save_soft = False, output_soft = None, 
                                                suffix = '_error_2', 
                                                colormap = None)
                            writer.save_seg_result(self.output_dir, 
                                                save_hard = self.save_hard, output_hard = seg_error_map_1_sim, 
                                                save_soft = False, output_soft = None, 
                                                suffix = '_error_1_sim', 
                                                colormap = None)
                            writer.save_seg_result(self.output_dir, 
                                                save_hard = self.save_hard, output_hard = seg_error_map_2_sim, 
                                                save_soft = False, output_soft = None, 
                                                suffix = '_error_2_sim', 
                                                colormap = None)
                if self.save_error_map:
                    writer.save_seg_result(self.output_dir, 
                                        save_hard = self.save_hard, output_hard = seg_error_map, 
                                        save_soft = False, output_soft = None, 
                                        suffix = '_error',
                                        colormap = None)
                    writer.save_seg_result(self.output_dir, 
                                        save_hard = self.save_hard, output_hard = seg_error_map_sim, 
                                        save_soft = False, output_soft = None, 
                                        suffix = '_error_sim',
                                        colormap = None)
            
            del output
            del output_hard
            del output_hard_2
            del output_hard_1
            del output_sim
            del output_hard_sim
            del output_hard_2_sim
            del output_hard_1_sim
            gc.collect()

        ###### compute metrics ######
        
        if self.evaluate:
            # compute confusion matrix and report
            reports = self._compute_metrics()

            feature_loss = np.average(feature_losses, axis = 0)
            # aggregate losses/errors/samples the validation set
            seg_loss = None if seg_criterion is None else np.average(seg_losses, axis = 0, 
                                                                                weights = valid_px_list)
            seg_bin_loss = None if seg_criterion_2 is None else np.average(seg_bin_losses, axis = 0, 
                                                                                weights = valid_px_bin_list)
            seg_loss_sim = None if seg_criterion is None else np.average(seg_losses_sim, axis = 0, 
                                                                                weights = valid_px_list_sim)
            seg_bin_loss_sim = None if seg_criterion_2 is None else np.average(seg_bin_losses_sim, axis = 0, 
                                                                                weights = valid_px_bin_list_sim)
            return self.cum_cms, reports, ((seg_loss, seg_bin_loss), (seg_loss_sim, seg_bin_loss_sim), feature_loss)
        else:
            return None
        
    def __del__(self):
        shutil.rmtree(self.tmp_dir)

    

