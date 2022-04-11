import sys
import os
import argparse
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataset import ExpUtils, StreamSingleOutputTrainingDataset
from models import Unet
from models.encoders import ResNetEncoder
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
            choices = ['TLM4c', 'TLM5c'], \
            help='Source of targets. TLMxc: SwissTLM3D forest targets with x classes')
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
    parser.add_argument('--adapt_loss_weights', action='store_true')
    parser.add_argument('--weight_bin_segmentation', action='store_true', help='NOT IMPLEMENTED')
    parser.add_argument('--use_subset', action='store_true')
    parser.add_argument('--decision', type=str, default='f', choices=['f', 'h']) # f for flat, h for hierarchical
    parser.add_argument('--lambda_bin', type=float, default=1.0)
    parser.add_argument('--lambda_feature', type=float, default=0.1)
    return parser


class DebugArgs():
    """Class for setting arguments directly in this python script instead of through a command line"""
    def __init__(self):
        self.debug = True
        self.input_sources = ['SI2017', 'ALTI']
        self.target_source = 'TLM5c'
        self.use_subset = False
        self.batch_size = 16
        self.num_epochs = 10
        self.lr = [1e-5, 1e-6, 1e-6, 1e-7] 
        self.learning_schedule = [5, 5, 5, 5] 
        self.n_negative_samples = [0, 5, 10, 20, 40, 80, 160, 320, 640, 1280] #[5, 10, 25, 100] 
        self.negative_sampling_schedule = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2] #[5, 5, 5, 5] 
        # supervision parameters
        self.weight_bin_segmentation = False
        self.decision = 'h' # no arg checking and parser yet
        self.lambda_bin = 1.0 # no arg checking and parser yet
        self.lambda_feature = 1.0 # no arg checking and parser yet
        self.adapt_loss_weights = False

        self.num_workers = 2
        self.skip_validation = False
        self.undersample_validation = 1
        self.resume_training = False
        self.output_dir = '/media/data/charrez/multi-resolution-forest-mapping/results'
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
    
    use_schedule = len(args.learning_schedule) > 1
    if use_schedule:
        if len(args.lr) != len(args.learning_schedule):
            raise ValueError('lr and learning_schedule should have the same number of elements')

    if len(args.n_negative_samples) != len(args.negative_sampling_schedule):
        raise ValueError('n_negative_samples and negative_sampling_schedule should have the same number of elements')
    control_training_set = len(args.n_negative_samples) > 0

    
        
    if args.undersample_validation < 1:
        raise ValueError('undersample_validation factor should be greater than 1')
    if args.debug:
        args.undersample_validation = 20
        print('Debug mode: only 1/{}th of the validation set will be used'.format(args.undersample_validation))
        
    exp_utils = ExpUtils(args.input_sources, args.target_source, decision = args.decision)
    # create dictionary used to save args        
    if isinstance(args, DebugArgs):
        args_dict = args.__dict__.copy()
    else:
        args_dict = vars(args).copy()
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
        keys = ['train_losses', 'train_losses_sim', 'train_losses_feature', 'train_total_losses', 'proportion_negative_samples', 'model_checkpoints', 'optimizer_checkpoints']
        if args.decision == 'h':
            keys.append('train_binary_losses')
            keys.append('train_binary_losses_sim')
        if not args.skip_validation:
            keys.extend(('val_reports', 'val_cms', 'val_epochs', 'val_losses', 'val_losses_sim', 'val_losses_feature', 'val_total_losses'))
            if args.decision == 'h':
                keys.append('val_binary_losses')
                keys.append('val_binary_losses_sim')
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
    prefix = "_".join(args.input_sources + [args.target_source])
    suffix = "_subset" if args.use_subset else ""
    train_csv_fn = 'src/data/csv/{}_train{}_with_counts.csv'.format(prefix, suffix)
    val_csv_fn = 'src/data/csv/{}_val{}.csv'.format(prefix, suffix)

    if n_input_sources == 1:
        input_col_names = ['input']
    else:
        input_col_names = ['input_' + str(i) for i in range(n_input_sources)]

    all_fns = utils.get_fns(train_csv_fn, *input_col_names, 'target')
    input_fns = np.stack(all_fns[:n_input_sources], axis = 0)
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
        n_samples = 200
        input_fns = input_fns[:n_samples]
        
        target_fns = target_fns[:n_samples]
        if negatives_mask is not None:
            negatives_mask = negatives_mask[:n_samples]
    
    # TODO [:50] for input_fns, target_fns, negatives_mask
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
    if use_schedule:
        n_controlled_epochs = min(args.num_epochs, np.sum(args.learning_schedule))
        lr_list = np.full(args.num_epochs, args.lr[-1])
        lr_list[:n_controlled_epochs] = np.repeat(args.lr, args.learning_schedule)[:n_controlled_epochs]
        
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

    model = Unet(encoder_depth=4, 
                decoder_channels=decoder_channels,
                in_channels=exp_utils.input_channels[0],
                classes=exp_utils.output_channels,
                upsample=upsample,
                aux_in_channels=aux_in_channels,
                aux_in_position=aux_in_position)

    model = model.to(device)
    # encoder = encoder.to(device)
    # encoder_sim = encoder_sim.to(device)

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
        optimizer.load_state_dict(starting_point['optimizer'])
        for g in optimizer.param_groups:
            g['lr'] = args.lr[0]
        # set the starting epoch
        starting_epoch = starting_point['epoch'] + 1
    else:
        save_dict = {
                'args': args_dict,
                'train_losses': [],
                'train_losses_sim': [],
                'train_losses_feature': [],
                'train_total_losses': [],
                'model_checkpoints': [],
                'optimizer_checkpoints' : [],
                'proportion_negative_samples' : []
            }
        if args.decision == 'h':
            save_dict['train_binary_losses'] = []
            save_dict['train_binary_losses_sim'] = []
        if not args.skip_validation:
            save_dict['val_reports'] = []
            save_dict['val_cms'] = []
            save_dict['val_epochs'] = []
            save_dict['val_losses'] = []
            save_dict['val_losses_sim'] = []
            save_dict['val_losses_feature'] = []
            save_dict['val_total_losses'] = []
            if args.decision == 'h':
                save_dict['val_binary_losses'] = []
                save_dict['val_binary_losses_sim'] = []
        

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
                            
        # shuffle data at every epoch (placed here so that all the workers use the same permutation)
        np.random.shuffle(dataset.fns)

        # forward and backward pass
        training_loss = utils.fit(
                model = model,
                device = device,
                dataloader = dataloader,
                optimizer = optimizer,
                n_batches = n_batches_per_epoch,
                seg_criterion = seg_criterion,
                seg_criterion_2 = seg_criterion_2,
                lambda_bin = args.lambda_bin,
                lambda_feature = args.lambda_feature
            )

        # debug
        new_mem = psutil.virtual_memory().used/1e6
        print('System memory after training epoch: \t\t{:.3f} MB'.format(new_mem))
        print('Training memory increment: \t\t{:.3f} MB'.format(new_mem-mem))
        mem = new_mem

        # evaluation (validation) 
        if not args.skip_validation: 
            print('Validation')
            results = inference.infer(seg_criterion, 
                                        seg_criterion_2)
            cm, report, losses = results
            val_losses, val_losses_sim, val_loss_feature = losses
            # collect individual validation losses and compute total validation loss
            val_loss, val_loss_2, *other_losses = val_losses
            val_loss_sim, val_loss_2_sim, *other_losses_sim = val_losses_sim
            val_total_loss = val_loss + val_loss_sim + args.lambda_feature * val_loss_feature
            if args.decision == 'h':
                val_total_loss += args.lambda_bin * val_loss_2 + args.lambda_bin * val_loss_2_sim
    
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
        loss, loss_sim, training_loss_feature = training_loss
        training_loss, training_binary_loss = loss
        training_loss_sim, training_binary_loss_sim = loss_sim
        training_total_loss = training_loss + training_loss_sim + args.lambda_feature*training_loss_feature
        if args.decision == 'h':
                training_total_loss += args.lambda_bin * training_binary_loss + args.lambda_bin * training_binary_loss_sim
        save_dict['train_total_losses'].append(training_total_loss)        
        save_dict['train_losses'].append(training_loss)
        save_dict['train_losses_sim'].append(training_loss_sim)
        save_dict['train_losses_feature'].append(training_loss_feature)
        if args.decision == 'h':
            save_dict['train_binary_losses'].append(training_binary_loss)
            save_dict['train_binary_losses_sim'].append(training_binary_loss_sim)
        
        # store validation losses/metrics
        if not args.skip_validation: 
            save_dict['val_reports'].append(report)
            save_dict['val_cms'].append(deepcopy(cm)) # deepcopy is necessary
            save_dict['val_epochs'].append(epoch)
            save_dict['val_total_losses'].append(val_total_loss)
            save_dict['val_losses'].append(val_loss)
            save_dict['val_losses_sim'].append(val_loss_sim)
            save_dict['val_losses_feature'].append(val_loss_feature)
            if args.decision == 'h':
                save_dict['val_binary_losses'].append(val_loss_2)
                save_dict['val_binary_losses_sim'].append(val_loss_2_sim)
                
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