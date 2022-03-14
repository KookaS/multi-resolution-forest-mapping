import sys
import os
import argparse
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataset import ExpUtils, StreamSingleOutputTrainingDataset, StreamMultiOutputTrainingDataset
from models import Unet, RuleUnet
import utils
import psutil
from copy import copy, deepcopy
########### Parameters ###############

def get_parser():
    parser = argparse.ArgumentParser(description='Launch model training')
    parser.add_argument('--input_sources', type=str, nargs='*', default=['SI2017', 'ALTI'],
            choices = ['SI2017', 'ALTI'], \
            help='Source of inputs. Order matters.'\
                'Example: --input_sources SI2017 ALTI')
    parser.add_argument('--target_source', type=str, default='TLM5c',
            choices = ['TLM3c','TLM4c', 'TLM5c'], \
            help='Source of targets. TLMxc: SwissTLM3D forest targets with x classes')
    parser.add_argument('--aux_target_sources', type=str, nargs='*', default=[],
            choices = ['VHM', 'TH', 'TCDCopHRL', 'TCD1', 'TCD2'], \
            help='Sources of supervision for intermediate regression tasks. TCDCopHRL: Copernicus HRL Tree Canopy Density. '
                'VHM: Vegetation Height Model (National Forest Inventory).')
    parser.add_argument('--output_dir', type = str, help='Directory where to store models and metrics. '
                        'The name of the directory will be used to name the model and metrics files. '
                        'A "training/" subdirectory will be automatically created)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size used for training')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, nargs='*', default=[1e-5], help='Learning rate')
    parser.add_argument('--learning_schedule', type=int, nargs='*', default = [], help='Number of epochs for'
                        'which the learning rate will be set to the corresponding value in --lr. The remaining epochs ' 
                        'are trained with the last value in --lr. --learning_schedule should have the same number of elements '
                        'as --lr if --lr has more than 1 value.')
    parser.add_argument('--n_negative_samples', type=int, nargs='*', default = [], help='Number of negative examples '
                        'to be used for training for each --negative_sampling_schedule period')
    parser.add_argument('--negative_sampling_schedule', type=int, nargs='*', default = [], help='Number of epochs for'
                        'which the number of negative samples will be set to the corresponding value in '
                        '--n_negative_samples. The remaining epochs are trained with all samples.'
                        '--negative_sampling_schedule should have the same number of elements as --n_negative_samples.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers to be used by cuda for data '
                        'loading')
    parser.add_argument('--skip_validation', action = 'store_true', help='Whether to skip validation at each epoch or '
                        'not')
    parser.add_argument('--undersample_validation', type=int, default=1, help='If n is the specified value, '
                        'a 1/n random subset of the validation set will be used for validation '
                        'at each epoch (speeds up validation). The subset is drawn at random at each epoch, '
                        'it will thus be different for each epoch')
    parser.add_argument('--resume_training', action='store_true', help='Flag to indicate that we want to resume '
                        'training from pretrained model stored in output_dir/training')
    parser.add_argument('--no_user_input', action='store_true', help='Flag to disable asking user confirmation for '
                        'overwriting files')
    parser.add_argument('--debug', action='store_true', help='Uses a small subset of the training and validation sets'
                        'to accelerate debugging')

    #TEMPORARY
    parser.add_argument('--adapt_loss_weights', action='store_true')
    parser.add_argument('--weight_bin_segmentation', action='store_true', help='NOT IMPLEMENTED')
    parser.add_argument('--use_subset', action='store_true')
    parser.add_argument('--weight_regression', action='store_true')
    parser.add_argument('--regression_loss', type=str, nargs='*', default=['MSElog', 'MSE'], choices=['MSE', 'MAE', 'MSElog', 'RMSE', 'RMSElog'])
    parser.add_argument('--penalize_residual', action='store_true')
    parser.add_argument('--decision', type=str, default='f', choices=['f', 'h']) # f for flat, h for hierarchical
    parser.add_argument('--lambda_bin', type=float, default=1.0)
    parser.add_argument('--lambda_regr', type=float, nargs='*', default=[1.])
    parser.add_argument('--lambda_corr', type=float, nargs='*', default=[1.])
    parser.add_argument('--regression_weight_slope', type=float, nargs='*', default=[1.0, 1.0])
    return parser


class DebugArgs():
    """Class for setting arguments directly in this python script instead of through a command line"""
    def __init__(self):
        self.debug = True
        self.input_sources = ['SI2017', 'ALTI']
        self.target_source = 'TLM5c'
        self.aux_target_sources = ['TH', 'TCD1']
        self.use_subset = False
        self.batch_size = 16
        self.num_epochs = 10
        self.lr = [1e-5, 1e-6, 1e-6, 1e-7] 
        self.learning_schedule = [5, 5, 5, 5] 
        self.n_negative_samples = [0, 5, 10, 20, 40, 80, 160, 320, 640, 1280] #[5, 10, 25, 100] 
        self.negative_sampling_schedule = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2] #[5, 5, 5, 5] 
        # supervision parameters
        self.weight_bin_segmentation = False
        self.adapt_loss_weights = False
        self.weight_regression = False # no arg checking and parser yet
        self.regression_loss = ['MSElog', 'MSE']
        self.regression_weight_slope = [0.0, 1.0]
        self.penalize_residual = True # no arg checking and parser yet
        self.decision = 'h' # no arg checking and parser yet
        self.lambda_bin = 1.0 # no arg checking and parser yet
        self.lambda_regr = [1., 0.75, 0.5, 0.25] #[1.0, 1.0] # no arg checking and parser yet
        self.lambda_corr = [0., 0.25, 0.5, 0.75] # no arg checking and parser yet

        self.num_workers = 2
        self.skip_validation = False
        self.undersample_validation = 1
        self.resume_training = True
        #self.output_dir = '/home/tanguyen/Documents/Projects/2020/ForestMapping/Code/ForestMapping/output/' + \
        #    'SI2017_ALTI_TLM_OF_F_SF_nns_0_10_10_10_sched_10_5_5_5'
        self.output_dir = '/home/tanguyen/Documents/Projects/2020/ForestMapping/Code/ForestMapping/output/sb_hierarchical_MSElog1em1_MSE_doubling_negatives_eps_1em1'
        self.no_user_input = True


########################################################################################################################

def train(args):

    exp_name = os.path.basename(args.output_dir)
    log_fn = os.path.join(args.output_dir, 'training','{}_metrics.pt'.format(exp_name))
    model_fn = os.path.join(args.output_dir, 'training', '{}_model.pt'.format(exp_name))

    ############ Check the output paths ###########
    if os.path.isfile(model_fn):
        if os.path.isfile(log_fn):
            if args.resume_training:
                print('Resuming the training process, {} and {} will be updated.'.format(log_fn, model_fn))
            else:
                print('WARNING: Training from scratch, {} and {} will be overwritten'.format(log_fn, model_fn))
                if not args.no_user_input:
                    print('Continue? (yes/no)')
                    while True:
                        proceed = input()
                        if proceed == 'yes':
                            break
                        elif proceed == 'no':
                            return
                        else:
                            print('Please answer by yes or no')
                            continue
        else:
            if args.resume_training:
                raise FileNotFoundError('Cannot resume training, {} does not exist'.format(log_fn))
            elif not os.path.isdir(os.path.dirname(log_fn)):
                print('Directory {} does not exist, it will be created'.format(os.path.dirname(log_fn)))
                os.makedirs(os.path.dirname(log_fn))
    else:
        if args.resume_training:
            raise FileNotFoundError('Cannot resume training, {} does not exist'.format(model_fn))
        elif not os.path.isdir(os.path.dirname(model_fn)):
            print('Directory {} does not exist, it will be created'.format(os.path.dirname(model_fn)))
            os.makedirs(os.path.dirname(model_fn))

    ############ Check other parameters ############

    n_input_sources = len(args.input_sources)
    n_aux_targets = len(args.aux_target_sources)
    
    use_schedule = len(args.learning_schedule) > 1
    if use_schedule:
        if len(args.lr) != len(args.learning_schedule):
            raise ValueError('lr and learning_schedule should have the same number of elements')
        
        
    use_aux_targets = n_aux_targets > 0        
    if use_aux_targets:
        # if len(args.lambda_regr) != n_aux_targets:
        #     if len(args.lambda_regr) == 1:
        #         args.lambda_regr = args.lambda_regr * n_aux_targets # replicate the singleton
        #     else:
        #         args.lambda_regr = args.lambda_regr[:n_aux_targets]
        #         print('WARNING: Values {} will be used as lambda_regr to match the number of auxiliary targets'
        #                                                                                     .format(args.lambda_regr))
        if use_schedule:
            if len(args.lambda_regr) != len(args.learning_schedule):
                raise ValueError('lambda_regr and learning_schedule should have the same number of elements')
            if len(args.lambda_corr) != len(args.learning_schedule):
                raise ValueError('lambda_corr and learning_schedule should have the same number of elements')
        
    else:
        args.lambda_regr = [None] # not used, just for clarity when reading the saved arguments 
        if use_schedule:
            args.lambda_corr = [1.0] * len(args.learning_schedule)  
        else:  
            args.lambda_corr = [1.0]
    if not use_aux_targets:
        if args.penalize_residual:
            print('WARNING: penalize_residual will be set to False because no auxiliary targets are specified')
            args.penalize_residual = False

    if len(args.n_negative_samples) != len(args.negative_sampling_schedule):
        raise ValueError('n_negative_samples and negative_sampling_schedule should have the same number of elements')
    control_training_set = len(args.n_negative_samples) > 0

    
        
    if args.undersample_validation < 1:
        raise ValueError('undersample_validation factor should be greater than 1')
    if args.debug:
        args.undersample_validation = 50
        print('Debug mode: only 1/{}th of the validation set will be used'.format(args.undersample_validation))
        
    if not args.weight_regression:
        args.regression_weight_slope = [0.0] * n_aux_targets
        print('Argument regression_weight_slope is ignored since weight_regression is set to false')
        
    exp_utils = ExpUtils(args.input_sources, args.aux_target_sources, args.target_source, decision = args.decision)
    # create dictionary used to save args        
    if isinstance(args, DebugArgs):
        args_dict = args.__dict__.copy()
    else:
        args_dict = vars(args).copy()
    # add the auxiliary targets in the args (useful for plotting the loss weights)
    try:
        args_dict['aux_variables'] = exp_utils.aux_variables
        try:
            args_dict['aux_target_stds'] = exp_utils.aux_target_stds
            try:
                args_dict['thresholds'] = exp_utils.unprocessed_thresholds
            except AttributeError:
                pass
        except AttributeError:
            pass
    except AttributeError:
            pass
    if args.resume_training:
        # check that the previous args match the new ones
        save_dict = torch.load(log_fn)
        previous_args_dict = save_dict['args']
        if args_dict != previous_args_dict:
            print('WARNING: The args saved in {} do not match the current args. '
            'The current args will be appended to the existing ones:')
            for key in args_dict:
                current_val =  args_dict[key]
                try:
                    # using tuples because some of the args are already lists (makes the code simpler)
                    if isinstance(previous_args_dict[key], tuple): #avoid nested tuples
                        args_dict[key] = (*previous_args_dict[key], current_val)
                        previous_val = previous_args_dict[key][-1]
                    else:
                        args_dict[key] = (previous_args_dict[key],current_val)
                        previous_val = previous_args_dict[key]
                    try:
                        val_change = previous_val != current_val   
                    except ValueError:
                        val_change = any([x != y for x, y in zip(np.ravel(previous_val), np.ravel(current_val))])
                    if val_change:
                        print('\t{}: previous value {}, current value {}'.format(key, previous_val, current_val))        
                except KeyError:
                    pass
        # check the keys
        keys = ['train_losses', 'train_total_losses', 'proportion_negative_samples', 'model_checkpoints', 'optimizer_checkpoints']
        if use_aux_targets:
            keys.extend(('train_regression_losses', 'train_residual_penalties'))
        if args.decision == 'h':
            keys.append('train_binary_losses')
        if not args.skip_validation:
            keys.extend(('val_reports', 'val_cms', 'val_epochs', 'val_losses', 'val_total_losses'))
            if use_aux_targets:
                keys.extend(('val_regression_error', 'val_pos_regression_error', 'val_neg_regression_error', 'val_regression_losses', 'val_regression_prediction_points', 'val_regression_target_points'))
                if args.penalize_residual:
                    keys.append('val_residual_penalties')
            if args.decision == 'h':
                keys.append('val_binary_losses')
        keys_not_found = list(k not in save_dict.keys() for k in keys)
        if any(keys_not_found):
            raise KeyError('Did not find ({}) entry(ies) in {}'.format(
                            ', '.join([k for k, not_found in zip(keys, keys_not_found) if not_found]), 
                            log_fn))


    if torch.cuda.is_available():
            device = torch.device("cuda")
    else:
        raise RuntimeError("CUDA is not available")

    print(args.__dict__)

    ############ Setup data ###################################################
    prefix = "_".join(args.input_sources + args.aux_target_sources + [args.target_source])
    suffix = "_subset" if args.use_subset else ""
    train_csv_fn = 'data/csv/{}_train{}_with_counts.csv'.format(prefix, suffix)
    val_csv_fn = 'data/csv/{}_val{}.csv'.format(prefix, suffix)

    if n_input_sources == 1:
        input_col_names = ['input']
    else:
        input_col_names = ['input_' + str(i) for i in range(n_input_sources)]
    if n_aux_targets == 1:
        aux_target_col_names = ['aux_target']
    else:
        aux_target_col_names = ['aux_target_' + str(i) for i in range(n_aux_targets)]

    all_fns = utils.get_fns(train_csv_fn, *input_col_names, *aux_target_col_names, 'target')
    input_fns = np.stack(all_fns[:n_input_sources], axis = 0)
    aux_target_fns = np.stack(all_fns[n_input_sources:-1], axis = 0) if use_aux_targets else None
    target_fns = all_fns[-1]
    
    print('Creating dataset...')
    tic = time.time()
    

    # check class counts if needed
    if control_training_set:
        positive_counts = np.array(utils.get_fns(
                                        train_csv_fn, 
                                        *['count_' + str(i) for i in range(1, exp_utils.n_classes)])
                                    )
        if None in positive_counts:
            raise RuntimeError('Could not read target counts in {}, which are necessary '
                                'for undersampling the training set'.format(train_csv_fn))
        negatives_mask = ~np.any(positive_counts, axis = 0)
    else: 
        negatives_mask = None

    # for debugging: use subset of training set
    if args.debug:
        n_samples = 100
        input_fns = input_fns[:n_samples]
        
        target_fns = target_fns[:n_samples]
        if use_aux_targets:
            aux_target_fns = aux_target_fns[:n_samples]
        if negatives_mask is not None:
            negatives_mask = negatives_mask[:n_samples]
    
    # create dataset
    if use_aux_targets:
        dataset = StreamMultiOutputTrainingDataset(
            input_fns=input_fns, 
            aux_target_fns=aux_target_fns,
            target_fns=target_fns, 
            exp_utils = exp_utils,
            n_neg_samples = None,
            negatives_mask= negatives_mask,
            verbose=False
            ) 
    else:
        dataset = StreamSingleOutputTrainingDataset(
        input_fns=input_fns, 
        target_fns=target_fns, 
        exp_utils = exp_utils,
        n_neg_samples = None,
        negatives_mask= negatives_mask,
        verbose=False
        )

    # create array containing the number of negatives samples to be selected for each epoch
    n_neg_samples = np.full(args.num_epochs, dataset.n_negatives)
    if control_training_set:
        n_controlled_epochs = min(args.num_epochs, np.sum(args.negative_sampling_schedule))
        n_neg_samples[:n_controlled_epochs] = np.repeat(
                                                args.n_negative_samples, 
                                                args.negative_sampling_schedule
                                                        )[:n_controlled_epochs]
        # clip the array to the total number of negative samples in the dataset
        n_neg_samples[:n_controlled_epochs] = np.minimum(n_neg_samples[:n_controlled_epochs], dataset.n_negatives)

    # create array containing the learning rate for each epoch
    lambda_corr_list = np.full(args.num_epochs, args.lambda_corr[-1])
    lambda_regr_list = np.full(args.num_epochs, args.lambda_regr[-1])
    if use_schedule:
        n_controlled_epochs = min(args.num_epochs, np.sum(args.learning_schedule))
        lr_list = np.full(args.num_epochs, args.lr[-1])
        lr_list[:n_controlled_epochs] = np.repeat(args.lr, args.learning_schedule)[:n_controlled_epochs]
        if use_aux_targets:
            lambda_corr_list[:n_controlled_epochs] = np.repeat(args.lambda_corr, args.learning_schedule)[:n_controlled_epochs]
            lambda_regr_list[:n_controlled_epochs] = np.repeat(args.lambda_regr, args.learning_schedule)[:n_controlled_epochs]
        
    print("finished in %0.4f seconds" % (time.time() - tic))

    # create dataloader
    print('Creating dataloader...')
    tic = time.time()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print("finished in %0.4f seconds" % (time.time() - tic))

    ############ Setup model ##################################################

    # Set model architecture
    decoder_channels = (256, 128, 64, 32)
    upsample = (True, True, True, False)
    if n_input_sources > 1:
        # 2 input modalities
        aux_in_channels = exp_utils.input_channels[1]
        aux_in_position = 1
    else:
        # 1 input modality
        aux_in_channels = None
        aux_in_position = None

    # Create model and criterion, and forward + backward pass function
    
    # Compute the constant loss weights if adapt_loss_weights is set to False
    if not args.adapt_loss_weights or args.decision == 'h': # use class frequencies corresponding to "positive" images only
        if args.decision == 'h':
            weights = torch.FloatTensor(exp_utils.get_weights(exp_utils.class_freq['seg_1'], 1, 0))
        else:
            weights = torch.FloatTensor(exp_utils.get_weights(exp_utils.class_freq['seg'], 1, 0))
        print('Loss weights: {}'.format(weights))
        if not args.adapt_loss_weights:
            if args.weight_bin_segmentation:
                raise NotImplementedError('Loss weighting for BCE loss not implemented')
                # Implementation for cross-entropy loss
                # weights_2 = torch.FloatTensor(exp_utils.get_weights(exp_utils.class_freq_2, 1, 0))
                # print('Loss weights: {}'.format(weights_2))
                # seg_criterion_2.weight = weights_2.to(device)
    else:
        weights = None
    
    # ignore_index to ignore the forest patch / gehoelzflaeche pixels and/or non-forest pixels
    seg_criterion = nn.CrossEntropyLoss(reduction = 'mean', ignore_index=exp_utils.i_nodata_val, weight=weights.to(device))

    if args.decision == 'h':
        seg_criterion_2 = nn.BCEWithLogitsLoss() #nn.CrossEntropyLoss()
        for i in range(n_input_sources):
            if exp_utils.input_nodata_val[i] is not None:
                print('WARNING: {}th input sources has nodata value {}, '
                      'but torch.nn.BCEWithLogitsLoss used for the binary task does '
                      'not handle nodata values'.format(i, exp_utils.input_nodata_val[i]))
    else:
        seg_criterion_2 = None
        
      
    if use_aux_targets:
        model = RuleUnet(encoder_depth=4, 
                decoder_channels=decoder_channels,
                in_channels=exp_utils.input_channels[0], 
                aux_channels=exp_utils.aux_channels,
                corr_channels=exp_utils.corr_channels,
                thresholds=exp_utils.thresholds,
                rules=exp_utils.rules,
                act_encoding=exp_utils.act_encoding,
                classes=exp_utils.output_channels,
                upsample=upsample,
                aux_in_channels=aux_in_channels,
                aux_in_position=aux_in_position,
                decision=args.decision)
        buffers = deepcopy(list(model.segmentation_head.buffers()))

        # regression criteria of auxiliary variables
        regr_criteria = [None] * n_aux_targets
        if not args.skip_validation:
            val_regr_criteria = [None] * n_aux_targets # different signature to collect reduction weights at inference
        for i, l in enumerate(args.regression_loss):
            # define the slope for target value-dependent regression weights
            if args.weight_regression:
                slope = args.regression_weight_slope[i]
            else:
                slope = 0.0
            # define the module used to compute the loss
            if l == 'MAE':
                loss_module = utils.WeightedMAE
            elif l == 'MSE':  
                loss_module = utils.WeightedMSE
            elif l == 'MSElog':
                loss_module = utils.WeightedMSElog
            elif l == 'RMSE':
                loss_module = utils.WeightedRMSE
            elif l == 'RMSElog':
                loss_module = utils.WeightedRMSElog
            else:
                raise ValueError('Regression loss "{}" not recognized'.format(args.regression_loss))
            # instantiate the loss
            regr_criteria[i] = loss_module(exp_utils.aux_target_means[i], 
                                                exp_utils.aux_target_stds[i],
                                                slope=slope,
                                                ignore_val=exp_utils.f_nodata_val,
                                                return_weights=False)
            val_regr_criteria[i] = loss_module(exp_utils.aux_target_means[i], 
                                                        exp_utils.aux_target_stds[i],
                                                        slope=slope,
                                                        ignore_val=exp_utils.f_nodata_val,
                                                        return_weights=True) 

        # debug
        res_penalizer = lambda x : torch.linalg.norm(x.view(-1), ord = 1) / x.nelement() #L-1 penalty

    else:
        model = Unet(encoder_depth=4, 
                    decoder_channels=decoder_channels,
                    in_channels=exp_utils.input_channels[0], 
                    classes=exp_utils.output_channels,
                    upsample=upsample,
                    aux_in_channels=aux_in_channels,
                    aux_in_position=aux_in_position)

        regr_criteria = None
        val_regr_criteria = None
        res_penalizer = None
        #fit = utils.fit

    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr[0], amsgrad=True)
    print('Initial learning rate: {}'.format(optimizer.param_groups[0]['lr']))

    ############ Setup training ###############################################

    if not args.skip_validation:

        inference = utils.Inference(model, 
                val_csv_fn, exp_utils, output_dir=None, 
                evaluate=True, save_hard=False, save_soft=False, save_error_map=False,
                batch_size=args.batch_size, num_workers=args.num_workers, device=device,
                undersample=args.undersample_validation, decision=args.decision)
                
    # load checkpoints if resuming training from existing model
    if args.resume_training:
        # load the state dicts (model and optimizer)
        starting_point = torch.load(model_fn)
        model.load_state_dict(starting_point['model'])
        if exp_utils.sem_bot:
            # restore buffer values from before load_state_dict
            model.segmentation_head.load_buffers(buffers, device = device)
        optimizer.load_state_dict(starting_point['optimizer'])
        for g in optimizer.param_groups:
            g['lr'] = args.lr[0]
        # set the starting epoch
        starting_epoch = starting_point['epoch'] + 1
    else:
        save_dict = {
                'args': args_dict,
                'train_losses': [],
                'train_total_losses': [],
                'model_checkpoints': [],
                'optimizer_checkpoints' : [],
                'proportion_negative_samples' : []
            }
        if use_aux_targets:
            save_dict['train_regression_losses'] = []
            save_dict['train_residual_penalties'] = []
        if args.decision == 'h':
            save_dict['train_binary_losses'] = []
        if not args.skip_validation:
            save_dict['val_reports'] = []
            save_dict['val_cms'] = []
            save_dict['val_epochs'] = []
            save_dict['val_losses'] = []
            save_dict['val_total_losses'] = []
            if use_aux_targets:
                save_dict['val_regression_error'] = []
                save_dict['val_pos_regression_error'] = []
                save_dict['val_neg_regression_error'] = []
                save_dict['val_regression_losses'] = []
                save_dict['val_regression_prediction_points'] = []
                save_dict['val_regression_target_points'] = []
                if args.penalize_residual:
                    save_dict['val_residual_penalties'] = []
            if args.decision == 'h':
                save_dict['val_binary_losses'] = []
        

        starting_epoch = 0

    # model = model.to(device)
    ############ Training #####################################################
    print('Starting training') 
    n_batches_per_epoch = int(len(dataset.fns) * exp_utils.num_patches_per_tile / args.batch_size)

    # debug
    mem = psutil.virtual_memory().used/1e6
    print('Initial system memory: \t{:.3f} MB'.format(mem))

    for i, epoch in enumerate(range(starting_epoch, starting_epoch + args.num_epochs)):
        print('\nTraining epoch: {}'.format(epoch))
        if control_training_set:
            # update the dataset to select the right number of random negative samples
            dataset.select_negatives(n_neg_samples[i])     
            if n_neg_samples[i] != n_neg_samples[i-1] or i==0:
                # recalculate the number of batches per epoch (for the progress bar)
                n_batches_per_epoch = int(len(dataset.fns) * exp_utils.num_patches_per_tile / args.batch_size) 
                if args.adapt_loss_weights:
                    # adapt the loss weights to the new negatives/positives ratio
                    if args.decision == 'f':
                        weights = torch.FloatTensor(
                                    exp_utils.get_weights(exp_utils.class_freq['seg'], dataset.n_positives, n_neg_samples[i])
                                    )
                        print('Updated loss weights: {}'.format(weights))
                        seg_criterion.weight = weights.to(device) 
                    else:
                        if args.weight_bin_segmentation:
                            raise NotImplementedError('Loss weighting for BCE loss not implemented')
                    
        # set learning rate            
        if use_schedule:
            if lr_list[i] != lr_list[i-1] and i > 0:
                print('Updated learning rate: {}'.format(lr_list[i]))
                for g in optimizer.param_groups:
                    g['lr'] = lr_list[i]
                    
        print('Lambda_corr: {}, lambda_regr: {}'.format(lambda_corr_list[i], lambda_regr_list[i]))
        
        # shuffle data at every epoch (placed here so that all the workers use the same permutation)
        np.random.shuffle(dataset.fns)

        # forward and backward pass
        regr_only = lambda_corr_list[i]==0
        training_loss = utils.fit(
                model = model,
                device = device,
                dataloader = dataloader,
                optimizer = optimizer,
                n_batches = n_batches_per_epoch,
                seg_criterion = seg_criterion,
                seg_criterion_2 = seg_criterion_2,
                regr_criteria = regr_criteria,
                res_penalizer = res_penalizer,
                lambda_bin = args.lambda_bin,
                lambda_regr = lambda_regr_list[i], 
                lambda_corr = lambda_corr_list[i],
                regr_only=regr_only
            )

        # debug
        new_mem = psutil.virtual_memory().used/1e6
        print('System memory after training epoch: \t\t{:.3f} MB'.format(new_mem))
        print('Training memory increment: \t\t{:.3f} MB'.format(new_mem-mem))
        mem = new_mem

        # evaluation (validation) 
        if not args.skip_validation: 
            print('Validation')
            results = inference.infer(seg_criterion=None if regr_only else seg_criterion, 
                                        seg_criterion_2=None if regr_only else seg_criterion_2, 
                                        regr_criteria= val_regr_criteria, 
                                        res_penalizer=None if regr_only else res_penalizer)
            if use_aux_targets:
                cm, report, val_losses, (val_regr_error, val_pos_regr_error, val_neg_regr_error), (regr_pred_pts, regr_target_pts) = results
            else:
                cm, report, val_losses = results
            # collect individual validation losses and compute total validation loss
            val_loss, val_loss_2, *other_losses = val_losses
            val_total_loss = 0 if regr_only else lambda_corr_list[i] * val_loss
            if args.decision == 'h' and not regr_only:
                val_total_loss += lambda_corr_list[i] * args.lambda_bin * val_loss_2
            if use_aux_targets:
                val_regr_losses, val_res_penalty = other_losses
                val_total_loss += sum(val_regr_losses) * lambda_regr_list[i]
                if args.penalize_residual and not regr_only:
                    val_total_loss += lambda_corr_list[i] * val_res_penalty
    
        # debug
        new_mem = psutil.virtual_memory().used/1e6
        print('System memory after validation: \t{:.3f} MB'.format(new_mem))
        print('Validation memory increment: \t\t{:.3f} MB'.format(new_mem-mem))
        mem = new_mem


        # update and save dictionary containing metrics and checkpoints
        
        save_dict['proportion_negative_samples'].append(n_neg_samples[i]/dataset.n_fns_all)
        save_dict['model_checkpoints'].append(deepcopy(model.state_dict()))
        save_dict['optimizer_checkpoints'].append(deepcopy(optimizer.state_dict()))
        save_dict['args']['num_epochs'] = epoch + 1 # number of epochs already computed

        # store training losses
        training_loss, training_binary_loss, training_regr_loss, train_res_penalty = training_loss
        training_total_loss = 0 if regr_only else lambda_corr_list[i] * training_loss
        if args.decision == 'h' and not regr_only:
                training_total_loss += lambda_corr_list[i] * args.lambda_bin * training_binary_loss
        if use_aux_targets:
            training_total_loss += sum(training_regr_loss) * lambda_regr_list[i]
            if args.penalize_residual and not regr_only:
                training_total_loss += lambda_corr_list[i] * train_res_penalty
        save_dict['train_total_losses'].append(training_total_loss)        
        save_dict['train_losses'].append(training_loss)
        if use_aux_targets:
            save_dict['train_regression_losses'].append(training_regr_loss)
            save_dict['train_residual_penalties'].append(train_res_penalty)
        if args.decision == 'h':
            save_dict['train_binary_losses'].append(training_binary_loss)
        
        # store validation losses/metrics
        if not args.skip_validation: 
            save_dict['val_reports'].append(report)
            save_dict['val_cms'].append(deepcopy(cm)) # deepcopy is necessary
            save_dict['val_epochs'].append(epoch)
            save_dict['val_total_losses'].append(val_total_loss)
            save_dict['val_losses'].append(val_loss)
            if use_aux_targets:
                save_dict['val_regression_error'].append(val_regr_error)
                save_dict['val_pos_regression_error'].append(val_pos_regr_error)
                save_dict['val_neg_regression_error'].append(val_neg_regr_error)
                save_dict['val_regression_losses'].append(val_regr_losses)
                save_dict['val_regression_prediction_points'].append(regr_pred_pts)
                save_dict['val_regression_target_points'].append(regr_target_pts)
                if args.penalize_residual:
                    save_dict['val_residual_penalties'].append(val_res_penalty)
            if args.decision == 'h':
                save_dict['val_binary_losses'].append(val_loss_2)
                
        with open(log_fn, 'wb') as f:
            torch.save(save_dict, f)

        # save last checkpoint in a separate file
        last_checkpoint = {'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'epoch': epoch,
                            'model_params': {'decision': args.decision}} 
        torch.save(last_checkpoint, model_fn)

########################################################################################################################

if __name__ == "__main__":
    if len(sys.argv) < 2:
        args = DebugArgs() # enables to run the script through IDE debugger without arguments
    else:
        parser = get_parser()
        args = parser.parse_args()

    train(args)