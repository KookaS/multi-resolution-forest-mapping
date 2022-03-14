import sys
import os
import argparse
import torch
from models import Unet, RuleUnet
import utils
from dataset import ExpUtils
from copy import deepcopy


def get_parser():

    parser = argparse.ArgumentParser(description='Launch model inference')
    parser.add_argument('--csv_fn', type=str, nargs='+', \
        help='Path to a CSV file containing at least two columns -- "input" or "input_x" '
            '(x an integer, for multimodal model), "target", and optionally "aux_target" or "aux_target_x" and '
            '"neighbour_group", that point to files of input imagery and targets and optionally auxiliary targets and '
            'which neighbour_group (i.e. mosaic) each tile is in.')
    parser.add_argument('--input_sources', type=str, nargs='+', default=['SI2017', 'ALTI'],
            choices = ['SI2017', 'ALTI'],
            help='Source of inputs. Order matters. Example: --input_sources SI2017 ALTI')
    parser.add_argument('--target_source', type=str, nargs='?', default=['TLM4c'],
            choices = ['TLM3c','TLM4c'],
            help='Source of targets. TLMxc: SwissTLM3D forest targets with x classes')
    parser.add_argument('--aux_target_sources', type=str, nargs='*', default=[], choices = ['TH','VHM', 'TCD1', 'TCD2', 'TCDCopHRL'],
            help='Sources of supervision for intermediate regression tasks. TCD: Copernicus HRL Tree Canopy Density. '
                'VHM: Vegetation Height Model (National Forest Inventory).')
    parser.add_argument('--model_fn', type=str, required=True,
        help='Path to the model file.')
    parser.add_argument('--output_dir', type=str, required = True,
        help='Directory where the output predictions will be stored.')
    parser.add_argument('--overwrite', action="store_true",
        help='Flag for overwriting "--output_dir" if that directory already exists.')
    parser.add_argument('--batch_size', type=int, default=32,
        help='Batch size to use during inference.')
    parser.add_argument('num_workers', type=int, default=0,
        help='Number of workers to use for data loading.')
    parser.add_argument('--save_hard', action="store_true",
        help='Flag that enables saving the "hard" class predictions.')
    parser.add_argument('--save_soft', action="store_true",
        help='Flag that enables saving the "soft" class predictions.')
    parser.add_argument('--save_error_map', action="store_true",
        help='Flag that enables saving the error maps for auxiliary regression predictions. Requires auxiliary targets '
                'to be specified in --csv_fn file')
    
    return parser


class DebugArgs():
    """Class for setting arguments directly in this python script instead of through a command line"""
    def __init__(self):
        self.input_sources = ['SI2017', 'ALTI']
        self.target_source = 'TLM5c' 
        self.aux_target_sources = [] #['TH', 'TCD1'] 
        self.batch_size = 16
        self.num_workers = 2 #0
        self.save_hard = True
        self.save_soft = False
        self.save_error_map = False
        self.overwrite = True
        set = 'test_with_context'
        #self.csv_fn = '/home/tanguyen/Documents/Projects/2020/ForestMapping/Code/ForestMapping/data/csv/SI2017_ALTI_TH_TCD1_TLM5c_{}.csv'.format(set) #_with_counts.csv'.format(set) #val_viz_subset.csv'
        self.csv_fn = '/home/tanguyen/Documents/Projects/2020/ForestMapping/Code/ForestMapping/data/csv/SI2017_ALTI_{}.csv'.format(set) 
        exp_name =  'bb_hierarchical' # 'sb_hierarchical_MSElog1em1_MSE_doubling_negatives' 
        self.model_fn = '/home/tanguyen/Documents/Projects/2020/ForestMapping/Code/ForestMapping/'\
                        'output/{}/training/{}_model.pt'.format(exp_name, exp_name)
        # self.model_fn = '/home/tanguyen/Documents/Projects/2020/ForestMapping/Code/ForestMapping/output/bb_hierarchical_slow_no_bin_weights_BCE/training/bb_hierarchical_slow_no_bin_weights_BCE_model_epoch_14.pt'
        self.output_dir = '/home/tanguyen/Documents/Projects/2020/ForestMapping/Code/ForestMapping/output/'\
           '{}/inference/epoch_19/{}'.format(exp_name, set)
        # self.output_dir = '/home/tanguyen/Documents/Projects/2020/ForestMapping/Code/ForestMapping/output/' + \
        #     'bb_hierarchical_slow_no_bin_weights_BCE/inference/epoch_14/'
        self.evaluate = False # option False does not work


#################################################################################################################

def infer(args):

    ############ Argument checking ###############

    # check paths of model and input
    if not os.path.exists(args.csv_fn):
        raise FileNotFoundError("{} does not exist".format(args.csv_fn))
    if os.path.exists(args.model_fn):
        model_fn = args.model_fn
        exp_name = os.path.basename(os.path.dirname(os.path.dirname(args.model_fn)))
        # os.makedirs(inference_dir, exist_ok = True)
    else:
        raise FileNotFoundError('{} does not exist.'.format(args.model_fn))

    # check output path
    output_dir = args.output_dir
    #if args.save_hard or args.save_soft:
    if output_dir is None: # defaut directory for output images
        inference_dir = os.path.join(os.path.dirname(os.path.dirname(args.model_fn)), 'inference')
        model_name = os.path.splitext(os.path.basename(args.model_fn))[0]
        output_dir = os.path.join(inference_dir, model_name)
        os.makedirs(output_dir, exist_ok = True)
    else: # custom directory for output images /metrics
        if os.path.exists(output_dir):
            if os.path.isfile(output_dir):
                raise NotADirectoryError("A file was passed as `--output_dir`, please pass a directory!")
            elif len(os.listdir(output_dir)) > 0:
                if args.overwrite:
                    print("WARNING: Output directory {} already exists, we might overwrite data in it!"
                            .format(output_dir))
                else:
                    raise FileExistsError("Output directory {} already exists and isn't empty."
                                            .format(output_dir))
        else:
            print("{} doesn't exist, creating it.".format(output_dir))
            os.makedirs(output_dir)
    if args.evaluate:
            metrics_fn = os.path.join(output_dir, '{}_metrics.pt'.format(exp_name))
             
    # check gpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise RuntimeError("CUDA is not available")

    # Set up data and training parameters
    model_obj = torch.load(model_fn)
    decision = model_obj['model_params']['decision']
    n_input_sources = len(args.input_sources)
    exp_utils = ExpUtils(args.input_sources, 
                               args.aux_target_sources, 
                               args.target_source, 
                               decision=decision)

    ############ Setup model ###############
    
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
    # Create model
    if exp_utils.sem_bot:
        model = RuleUnet(encoder_depth=4, 
                decoder_channels=decoder_channels,
                in_channels = exp_utils.input_channels[0], 
                aux_channels = exp_utils.aux_channels,
                corr_channels = exp_utils.corr_channels,
                thresholds = exp_utils.thresholds,
                rules = exp_utils.rules,
                act_encoding = exp_utils.act_encoding,
                classes = exp_utils.output_channels,
                upsample = upsample,
                aux_in_channels = aux_in_channels,
                aux_in_position = aux_in_position,
                decision=decision)
        buffers = deepcopy(list(model.segmentation_head.buffers()))
    else:
        model = Unet(encoder_depth=4, 
                    decoder_channels=decoder_channels,
                    in_channels = exp_utils.input_channels[0], 
                    classes = exp_utils.output_channels,
                    upsample = upsample,
                    aux_in_channels = aux_in_channels,
                    aux_in_position = aux_in_position)


    model.load_state_dict(model_obj['model'])
    if exp_utils.sem_bot:
        # restore buffer values from before load_state_dict
        model.segmentation_head.load_buffers(buffers, device=device)
    model = model.to(device)

    ############ Inference ###############

    # inference = utils.get_inference_instance(model, args.csv_fn, exp_utils, output_dir = args.output_dir, 
    #             evaluate = args.evaluate,
    #             save_hard = args.save_hard, save_soft = args.save_soft, save_error_map = args.save_error_map,
    #             batch_size = args.batch_size, 
    #             num_workers = args.num_workers, 
    #             device = device, decision = decision)
    inference = utils.Inference(model, args.csv_fn, exp_utils, output_dir = output_dir, 
                                        evaluate = args.evaluate, save_hard = args.save_hard, save_soft = args.save_soft, 
                                        save_error_map = args.save_error_map, batch_size = args.batch_size, 
                                        num_workers = args.num_workers, device = device, decision = decision)

    result = inference.infer()

    ############ Evaluation ###############
    
    if args.evaluate:
        if result is not None:
            #cumulative_cm, report, _, val_regr_error, regr_pts = result
            cumulative_cm, report, *other_outputs = result
            if isinstance(args, DebugArgs):
                args_dict = args.__dict__
            else:
                args_dict = vars(args).copy()
            if exp_utils.sem_bot:
                args_dict['aux_variables'] = exp_utils.aux_variables
            # Save metrics
            d = {
                'args': args_dict,
                'val_reports': report,
                'val_cms': cumulative_cm
            }    
            #if len(val_regr_error) > 0:
            if exp_utils.sem_bot:
                _, val_regr_error, regr_pts = other_outputs
                val_regr_error, val_pos_regr_error, val_neg_regr_error = val_regr_error
                d['val_regression_error'] = val_regr_error
                d['val_pos_regression_error'] = val_pos_regr_error
                d['val_neg_regression_error'] = val_neg_regr_error
                d['val_regression_prediction_points'], d['val_regression_target_points'] = regr_pts
            with open(metrics_fn, 'wb') as f:
                torch.save(d, f)

#################################################################################################################

if __name__ == "__main__":
    if len(sys.argv) < 2:
        args = DebugArgs() # enables to run the script through IDE debugger without arguments
    else:
        parser = get_parser()
        args = parser.parse_args()

    infer(args)