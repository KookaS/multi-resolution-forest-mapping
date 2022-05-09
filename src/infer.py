import sys
import os
import argparse
import torch
from models import Unet
import utils
from dataset import ExpUtils


def get_parser():

    parser = argparse.ArgumentParser(description='Launch model inference')
    parser.add_argument('--csv_fn', type=str, \
        help='Path to a CSV file containing at least two columns -- "input" or "input_x" '
            '(x an integer, for multimodal model), "target", and optionally "neighbour_group", that point to files of '
            'input imagery and targets and optionally which neighbour_group (i.e. mosaic) each '
            'tile is in.')
    parser.add_argument('--input_sources', type=str, nargs='+', default=['SI2017', 'ALTI'],
            choices = ['SI2017', 'ALTI', 'SI1946'],
            help='Source of inputs. Order matters. Example: --input_sources SI2017 ALTI')
    parser.add_argument('--target_source', type=str, nargs='?', default=['TLM4c'],
            choices = ['TLM4c', 'TLM5c'],
            help='Source of targets. TLMxc: SwissTLM3D forest targets with x classes')
    parser.add_argument('--model_fn', type=str, required=True,
        help='Path to the model file.')
    parser.add_argument('--output_dir', type=str, required = True,
        help='Directory where the output predictions will be stored.')
    parser.add_argument('--overwrite', action="store_true",
        help='Flag for overwriting "--output_dir" if that directory already exists.')
    parser.add_argument('--batch_size', type=int, default=32,
        help='Batch size to use during inference.')
    parser.add_argument('--num_workers', type=int, default=0,
        help='Number of workers to use for data loading.')
    parser.add_argument('--save_hard', action="store_true",
        help='Flag that enables saving the "hard" class predictions.')
    parser.add_argument('--save_soft', action="store_true",
        help='Flag that enables saving the "soft" class predictions.')
    parser.add_argument('--save_error_map', action="store_true",
        help='Flag that enables saving prediction error maps.')
    parser.add_argument('--evaluate', action="store_true", help='Flag that enables computing metrics')
    parser.add_argument('--compare_dates', action="store_true",
        help='Compare images from 1946 and 2017')
    
    return parser


class DebugArgs():
    """Class for setting arguments directly in this python script instead of through a command line"""
    def __init__(self):
        dir = '/home/tanguyen/Documents/Projects/2020/ForestMapping/Code/ForestMapping'
        self.input_sources = ['SI2017', 'ALTI']
        self.target_source = 'TLM5c' 
        self.batch_size = 16
        self.num_workers = 2 #0
        self.save_hard = False
        self.save_soft = False
        self.save_error_map = False
        self.overwrite = True
        set = 'test_subset'
        suffix = '_with_counts' if set == 'train' else ''
        self.csv_fn = os.path.join(dir, 'data/csv/{}_{}_{}{}.csv'.format('_'.join(self.input_sources), 
                                                                         self.target_source, 
                                                                         set, 
                                                                         suffix))
        exp_name =  'baseline_hierarchical'
        self.model_fn = os.path.join(dir,'output/{}/training/{}_model.pt'.format(exp_name, exp_name))
        self.output_dir = os.path.join(dir,'output/{}/inference/epoch_19/{}'.format(exp_name, set))
        self.evaluate = True  


#################################################################################################################

def infer(args):

    ############ Argument checking ###############

    # check paths of model and input
    if not os.path.exists(args.csv_fn):
        raise FileNotFoundError("{} does not exist".format(args.csv_fn))
    if os.path.exists(args.model_fn):
        model_fn = args.model_fn
        exp_name = os.path.basename(os.path.dirname(os.path.dirname(args.model_fn)))
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
                         args.target_source, 
                         decision=decision)

    ############ Setup model ###############
    
    # Set model architecture
    decoder_channels = (256, 128, 64, 32)
    upsample = (True, True, True, False)
    if 'ALTI' in args.input_sources:
        # 2 input modalities
        aux_in_channels = exp_utils.input_channels['ALTI']
        aux_in_position = 1
    else:
        # 1 input modality
        aux_in_channels = None
        aux_in_position = None
    # Create model
    model = Unet(encoder_depth=4,
                decoder_channels=decoder_channels,
                in_channels=exp_utils.input_channels,
                classes=exp_utils.output_channels,
                upsample=upsample,
                aux_in_channels=aux_in_channels,
                aux_in_position=aux_in_position,
                input_sources=args.input_sources)

    model.load_state_dict(model_obj['model'])
    model = model.to(device)

    ############ Inference ###############

    inference = utils.Inference(model, args.csv_fn, exp_utils, output_dir = output_dir, 
                                        evaluate = args.evaluate, save_hard = args.save_hard, save_soft = args.save_soft, 
                                        save_error_map = args.save_error_map, batch_size = args.batch_size, 
                                        num_workers = args.num_workers, device = device, decision = decision, compare_dates = args.compare_dates)

    result = inference.infer()

    ############ Evaluation ###############
    
    if args.evaluate:
        if result is not None:
            cumulative_cm, report, _ = result
            if isinstance(args, DebugArgs):
                args_dict = args.__dict__
            else:
                args_dict = vars(args).copy()
            # Save metrics
            d = {
                'args': args_dict,
                'val_reports': report,
                'val_cms': cumulative_cm
            }    
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