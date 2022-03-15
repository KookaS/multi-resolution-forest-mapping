import os
from PIL import Image
from osgeo import gdal
import numpy as np

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

if __name__ == "__main__":
    # data_path = '/media/data/charrez/'
    data_path = r'C:\Users\Olivier\Documents\EPFL 2021-2022\Space project'
    dataset_path = 'SwissIMAGE'
    path = os.path.join(data_path,dataset_path)

    entries = os.listdir(path)
    files_path = os.path.join(path,entries[0])
    tif_files = os.listdir(files_path)

    file_path = os.path.join(files_path,tif_files[0])
    
    """tif_file = Image.open(file_path)
    tif_file.show()"""

    """tif_file = gdal.Open(file_path)
    metadata=tif_file.GetMetadata()
    print(metadata)
    tif_file.show()"""

    """step2 = tif_file.GetRasterBand(1)
    img_as_array = step2.ReadAsArray()
    size1,size2=img_as_array.shape"""

    tif_file = Image.open(file_path)
    print(np.array(tif_file).shape)
    image = np.array(tif_file)

    input = image.reshape((1, image.shape[0]*image.shape[1]))
    print(input.shape)


    