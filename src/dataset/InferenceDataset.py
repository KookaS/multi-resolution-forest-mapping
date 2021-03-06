import numpy as np
import torch
import rasterio
import rasterio.merge
import rasterio.transform
import rasterio.warp
from rasterio.windows import Window
import torch.nn.functional as f

import time

from torch.utils.data.dataset import Dataset

class InferenceDataset(Dataset):

    """
    Dataset for inference. 
    - Generates regularly spaced, and possibly overlapping patches over a dataset. 
    - Reads each tile with a margin around it
    - Supports multi-source input and target rasters with different depths and resolutions (pixel size for every source should be 
    a multiple of the smallest pixel size)
    """

    def __init__(self, input_vrt_fns, exp_utils, batch_size = 16, target_vrt_fn = None, 
                 input_nodata_val = None, target_nodata_val = None, input_keys = [], target_keys = []):
        """
        Args:
            - image_fn (1-D ndarray of str): path to the files to sample inputs from (files from each input source 
                corresponding to the same location)
            - target_fn (str): path to the file containing the ground truth
            - exp_utils (ExpUtils): object containing the information about the data sources and the patch 
                parameters (patch size, patch stride)
        """
        
        if isinstance(target_vrt_fn, list) or isinstance(target_vrt_fn, np.ndarray):
            target_vrt_fn = target_vrt_fn[0]
        
        # set parameters
        self.n_inputs = len(input_vrt_fns)
        self.input_vrt = [rasterio.open(fn, 'r') for fn in input_vrt_fns]
        self.input_fns = [vrt.files[1:] for vrt in self.input_vrt] # the first filename is the vrt itself
        self.n_tiles = len(self.input_fns[0])
        if target_vrt_fn is None:
            self.sample_target = False
            self.target_vrt = None
            self.target_fns = None
        else:
            self.sample_target = True
            self.target_vrt = rasterio.open(target_vrt_fn)
            self.target_fns = self.target_vrt.files[1:]

        self.exp_utils = exp_utils
        self.input_scales = exp_utils.input_scales
        self.target_scale = exp_utils.target_scale
        self.patch_size = exp_utils.patch_size 
        self.patch_stride = exp_utils.patch_stride 
        self.tile_margin = exp_utils.tile_margin

        self.batch_size = batch_size
        
        self.input_nodata_val = input_nodata_val
        self.target_nodata_val = target_nodata_val
        
        self.input_keys = input_keys
        self.target_keys = target_keys

    
    def _get_patch_coordinates(self, height, width): 
        """
        Fills self.patch_coordinates with an array of dimension (n_patches, 2) containing upper left pixels of patches, 
        at the resolution of the coarsest input/targets
        """
        xs = list(range(0, height - self.patch_size, self.patch_stride)) + [height - self.patch_size]
        ys = list(range(0, width - self.patch_size, self.patch_stride)) + [width - self.patch_size]
        xgrid, ygrid = np.meshgrid(xs, ys)
        #patch_coordinates = np.vstack([xgrid.ravel(), ygrid.ravel()]).T
        #num_patches = patch_coordinates.shape[0]
        return np.vstack((xgrid.ravel(), ygrid.ravel())).T

    def _get_input_nodata_mask(self, data, height, width, margins): 
        """
        Create nodata mask. A nodata pixel in the mask corresponds to an overlapping nodata pixel in any of the inputs.
        """
        # get the dimensions of the output
        top_margin, left_margin, bottom_margin, right_margin = [int(m * self.target_scale) for m in margins]
        output_height = height * self.target_scale - top_margin - bottom_margin
        output_width = width * self.target_scale - left_margin - right_margin
        check = np.full((output_height, output_width), False) 
        
        # check each input
        for i, image_data in enumerate(data):
            
            op1, _ = self.exp_utils.input_nodata_check_operator[i] 
            if self.input_nodata_val[i] is not None:
                check_orig = op1(image_data[top_margin:image_data.shape[0]-bottom_margin, left_margin:image_data.shape[1]-right_margin] == self.input_nodata_val[i], axis = -1)
                # downscale and combine with current mask
                index = 0   # for one element only
                s = self.input_scales[self.input_keys[index]] // self.target_scale
                if s == 0:
                    raise RuntimeError('At least one of the inputs is coarser that the target, this is curently not '
                                        'supported.')
                for j in range(s):
                    check = np.logical_or(check, check_orig[j::s, j::s][:output_height, :output_width])

        return check

    def _read_tile(self, vrt, fn, max_margin = None, squeeze = True):
        with rasterio.open(fn, 'r') as f_tile:
            left, top = f_tile.bounds.left, f_tile.bounds.top
            h, w, = f_tile.height, f_tile.width
        i_min, j_min = vrt.index(left, top)
        if max_margin is not None:
            # compute available margins around the tile
            top_margin = min(max(0, i_min-max_margin), max_margin)
            left_margin = min(max(0, j_min-max_margin), max_margin)
            bottom_margin = min(max(0, vrt.height - (i_min + h+max_margin)), max_margin)
            right_margin = min(max(0, vrt.width - (j_min + w+max_margin)), max_margin)
        else:
            top_margin, left_margin, bottom_margin, right_margin = 0, 0, 0, 0
        # read the tile + margins
        win = Window(   j_min - left_margin, 
                        i_min - top_margin, 
                        w + left_margin + right_margin, 
                        h + top_margin + bottom_margin)

        data = vrt.read(window = win)
        if data.shape[0] == 1 and squeeze:
            data = data.squeeze(0)
        else:
            data = np.moveaxis(data, (1, 2, 0), (0, 1, 2))
        return data, (h + top_margin + bottom_margin, w + left_margin + right_margin), \
                (top_margin, left_margin, bottom_margin, right_margin)
                
    def _build_batch(self, data, coords, s, num_batches, remainder):
        """first dimension of data should be the depth (channels)"""
        batches = [None] * num_batches
        mult_channels = data.dim() > 2
        if mult_channels:
            shape = (self.patch_size*s, self.patch_size*s, data.shape[0])
            data_copy = torch.clone(data).movedim((1, 2, 0),(0, 1, 2)) #to make the indexing easier
        else:
            shape = (self.patch_size*s, self.patch_size*s)
            data_copy = torch.clone(data)
        for batch_num in range(num_batches):
            this_batch_size = remainder if (batch_num == num_batches - 1 and remainder > 0) else self.batch_size
            batch = torch.empty((this_batch_size, *shape), dtype = data.dtype)
            offset = self.batch_size * batch_num
            for patch_num in range(this_batch_size):
                xp, yp = coords[offset + patch_num]
                batch[patch_num] = data_copy[xp * s:(xp+self.patch_size) * s, yp * s:(yp+self.patch_size) * s]
            if mult_channels:
                batch = batch.movedim((0, 3, 1, 2), (0, 1, 2, 3))
            batches[batch_num] = batch
        return batches

    def __getitem__(self, idx):
        '''
        Output:
            - data (list of tensors): contains patches corresponding to the same location in each input source 
            - np.array containing upper left coordinates of the patch, expressed at the coarsest resolution
        '''

        #### read tiles
        
        image_data = [None] * self.n_inputs
        for i, input_key in enumerate(self.input_keys):
            
            s = self.input_scales[input_key]
            data, (height, width), margins = self._read_tile(self.input_vrt[i], 
                                                            self.input_fns[i][idx], 
                                                            max_margin=self.tile_margin*s,
                                                            squeeze = False)
            # data = np.moveaxis(data, 0, -1) # the preprocessing function will swap axis back
            height_i = height // s
            width_i = width // s
            margins_i = [m/s for m in margins]
            if i == 0:
                tile_height, tile_width = height_i, width_i
                tile_margins = margins_i
            else:
                if height_i != tile_height or width_i != tile_width or margins_i != tile_margins:
                    raise RuntimeError('The dimensions of the input sources do not match: '
                                        '(height={}, width={}, margins={}) for the first source '
                                        'v.s (height={}, width={}, margins={}) for the {}th source'
                                        .format(tile_height, tile_width, ','.join(tile_margins),
                                                height_i, width_i, ','.join(margins_i), i))
            image_data[i] = data
        
            
        if self.sample_target:
            s = self.target_scale
            target_data, (target_height, target_width), margins = self._read_tile(self.target_vrt, 
                                                self.target_fns[idx], 
                                                max_margin=self.tile_margin*s)
            # target_data = np.squeeze(target_data, axis = 0)
            height_i = target_height // s
            width_i = target_width // s
            margins_i = [m/s for m in margins]
            if height_i != tile_height or width_i != tile_width or margins_i != tile_margins:
                    raise RuntimeError('The dimensions of the inputs and targets do not match: '
                                        '(height={}, width={}, margins={}) for the inputs '
                                        'v.s (height={}, width={}, margins={}) for the target'
                                        .format(tile_height, tile_width, ','.join(str(tile_margins)),
                                                height_i, width_i, ','.join(str(margins_i))))
        else:
            target_data = None
        
              
        #### build batches
        input_nodata_mask = self._get_input_nodata_mask(image_data, tile_height, tile_width, tile_margins)
        coords = self._get_patch_coordinates(tile_height, tile_width)
        num_patches = coords.shape[0]
        num_batches, remainder = divmod(num_patches, self.batch_size)
        if remainder > 0:
            num_batches += 1
            
        # build input batches
        input_batches = [None] * self.n_inputs
        # image_data = self.exp_utils.preprocess_inputs(image_data)
        for i in range(self.n_inputs):
            input_batches[i] = self._build_batch(self.exp_utils.preprocess_input(image_data[i], i), coords, 
                                                 self.input_scales[self.input_keys[i]], num_batches, remainder)

        # build target batches
        if self.sample_target:
            target_data_for_batches = self.exp_utils.preprocess_training_target(target_data)
            target_batches = self._build_batch(target_data_for_batches, coords, self.target_scale, num_batches, remainder)
            # target data as a full tile
            top_margin, left_margin, bottom_margin, right_margin = [int(m * self.target_scale) for m in tile_margins]
            target_tile = self.exp_utils.preprocess_inference_target(
                            target_data[top_margin:target_height-bottom_margin, left_margin:target_width-right_margin]
                                                                        ) #.squeeze(0)
        else:
            target_batches = None
            target_tile = None
            

        #split the coordinates into chunks corresponding to the batches
        coords = [coords[i:min(i+self.batch_size, num_patches)] for i in range(0, num_patches, self.batch_size)]

        return (input_batches, target_batches), \
                target_tile, \
                coords, (tile_height, tile_width), margins, input_nodata_mask   

    def __len__(self):
        return self.n_tiles
    
    def __del__(self):
        for vrt in self.input_vrt:
            vrt.close()
        if self.target_vrt is not None:
            self.target_vrt.close()

    





    

    

