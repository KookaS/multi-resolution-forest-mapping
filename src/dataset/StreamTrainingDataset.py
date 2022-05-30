import numpy as np
import torch
from torch.utils.data.dataset import IterableDataset
import rasterio
from rasterio.errors import RasterioError, RasterioIOError
from math import ceil
import cv2

class StreamSingleOutputTrainingDataset(IterableDataset):
    """
    Dataset for training. Generates random small patches over the whole training set.
    """

    def __init__(self, input_fns, target_fns, exp_utils, n_neg_samples = None,
                negatives_mask = None, verbose=False, input_keys=[], **kwargs):
        """
        Args:
            - input_fns (ndarray of str): array of dimension (n_input_sources, n_samples) or (n_samples,), containing 
                paths to input files
            - targets_fns (ndarray of str): array of dimension (n_samples,) containing paths to ground truth files
            - exp_utils (ExpUtils)
            - n_neg_samples (int): number of negative samples (i.e. containing class 0 only) to use
            - negatives_mask (1-D array of bool): indicates for each sample if it is negative or not
            - verbose (bool)
        """

        self._create_file_list(input_fns, target_fns, **kwargs)

        self.patch_size = exp_utils.patch_size
        self.num_patches_per_tile = exp_utils.num_patches_per_tile
        self.exp_utils = exp_utils
        self.verbose = verbose
        self.negatives_mask = negatives_mask
        self.input_keys = input_keys
        
        self.n_fns_all = len(self.fns)

        # store filenames of positive and negative examples separately
        if self.negatives_mask is None:
            self.fns_positives = None
            self.fns_negatives = None
        else:
            self._split_fns()
            self.select_negatives(n_neg_samples)

    def _create_file_list(self, input_fns, target_fns, **kwargs):
        """Create an file array with one sample per row"""
        # check dimensions of input_fns
        if not input_fns.all():
            raise ValueError('input_fns is not defined.'\
                            'It should in the csv be `input`' )
        if not target_fns.all():
            raise ValueError('target_fns is not defined.'\
                            'It should in the csv be `target`' )
                            
        if input_fns.ndim == 1:
            self.n_inputs = 1
            self.input_fns = np.expand_dims(input_fns, 0)
        elif input_fns.ndim == 2:
            self.n_inputs = input_fns.shape[0]
            self.input_fns = input_fns
        else:
            raise ValueError('input_fns should have one dimension (n_samples,) \
                                or two dimensions(n_input_sources, n_samples) ')

        self.fns = list(zip(zip(*input_fns), target_fns))

    def _split_fns(self):
        """
        Creates two lists self.fns_positives and self.fns_negatives which store positive and negative filenames 
        separately
        """
        negative_idx = np.where(self.negatives_mask)[0]
        self.fns_positives = [self.fns[i] for i in range(len(self.fns)) if i not in negative_idx]
        self.fns_negatives = [self.fns[i] for i in negative_idx]
        self.n_positives = len(self.fns_positives)
        self.n_negatives = len(self.fns_negatives)

    def select_negatives(self, n_neg_samples, negatives_mask= None):
        """
        Fills self.fn with the right number of negative samples
        Also updates self.negatives_mask with the specified negatives_mask array if needed
        """
        if n_neg_samples is not None:           
            if negatives_mask is not None:
                # update self.negatives_mask, self.fns_positives and self.fns_negatives if necessary
                if np.all(negatives_mask != self.negatives_mask):
                    self.negatives_mask = negatives_mask
                    self._split_fns()

            if self.negatives_mask is None: # negatives_mask argument has never been provided
                raise ValueError('Argument "negatives" must be specified at initialization or in select_negatives() '
                                'in order to control the number of negative samples')
            else: # select negative samples if necessary
                if n_neg_samples == 0: # do not use negative samples
                    self.fns = self.fns_positives
                elif n_neg_samples < self.n_negatives: # pick negative samples randomly
                    draw_idx = np.random.choice(self.n_negatives, size=(n_neg_samples,), replace = False)
                    self.fns = self.fns_positives + [self.fns_negatives[i] for i in draw_idx]
                elif n_neg_samples >= self.n_negatives: # use all negative samples
                    self.fns = self.fns_positives + self.fns_negatives
                print('Using {} training samples out of {}'.format(len(self.fns), self.n_fns_all))

    def _get_worker_range(self, fns):
        """Get the range of tiles to be assigned to the current worker"""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        # WARNING: when several workers are created they all have the same numpy random seed but different torch random 
        # seeds. 
        seed = torch.randint(low=0,high=2**32-1,size=(1,)).item()
        np.random.seed(seed) # set a different seed for each worker

        # define the range of files that will be processed by the current worker: each worker receives 
        # ceil(num_filenames / num_workers) filenames
        num_files_per_worker = ceil(len(fns) / num_workers)
        lower_idx = worker_id * num_files_per_worker
        upper_idx = min(len(fns), (worker_id+1) * num_files_per_worker)

        return lower_idx, upper_idx

    def _stream_tile_fns(self, lower_idx, upper_idx):
        """Generator providing input and target paths tile by tile from lower_idx to upper_idx"""
        for idx in range(lower_idx, upper_idx):
            yield self.fns[idx]

    def _extract_multi_patch(self, data, scales, x, xstop, y, ystop, input_keys):
        """
        Extract a patch from multisource data given the relative scales of the sources and boundary coordinates
        """
        return [self._extract_patch(data[i],scales[key], x, xstop, y, ystop) for i, key in enumerate(input_keys)]

    def _extract_patch(self, data, scale, x, xstop, y, ystop):
        return data[y*scale:ystop*scale, x*scale:xstop*scale]

    def _generate_patch(self, data, num_skipped_patches, coord = None):
        """
        Generates a patch from the input(s) and the targets, randomly or using top left coordinates "coord"

        Args:
            - data (list of (list of) tensors): input and target data
            - num_skipped_patches (int): current number of skipped patches (will be updated)
            - coord: top left coordinates of the patch to extract, in the coarsest modality

        Output:
            - patches (list of (list of) tensors): input and target patches
            - num_skipped_patches (int)
            - exit code (int): 0 if success, 1 if IndexError or invalid patch (due to nodata)
            - (x,y) (tuple of ints): top left coordinates of the extracted patch
        """

        input_data, target_data = data
        # find the coarsest data source
        height, width = target_data.shape[:2] 
        height = height // self.exp_utils.target_scale
        width = width // self.exp_utils.target_scale

        if coord is None: # pick the top left pixel of the patch randomly
            x = np.random.randint(0, width-self.patch_size)
            y = np.random.randint(0, height-self.patch_size)
        else: # use the provided coordinates
            x, y = coord
            
        # extract the patch
        try:
            xstop = x + self.patch_size
            ystop = y + self.patch_size
            # extract input patch
            input_patches = self._extract_multi_patch(input_data, self.exp_utils.input_scales, x, xstop, y, ystop, self.input_keys)
            # extract target patch
            target_patch = self._extract_patch(target_data, self.exp_utils.target_scale, x, xstop, y, ystop)
        except IndexError:
            if self.verbose:
                print("Couldn't extract patch (IndexError)")
            return (None, num_skipped_patches, 1, (x, y))

        # check for no data
        skip_patch = self.exp_utils.target_nodata_check(target_patch) or self.exp_utils.inputs_nodata_check(*input_patches) 
        if skip_patch: # the current patch is invalid
            num_skipped_patches += 1
            return (None, num_skipped_patches, 1, (x, y))

        # preprocessing (needs to be done after checking nodata)
        input_patches = self.exp_utils.preprocess_inputs(input_patches)
        target_patch = self.exp_utils.preprocess_training_target(target_patch)
        patches = [input_patches, target_patch]

        return (patches, num_skipped_patches, 0, (x, y))

    def _read_tile(self, img_fn, target_fn):
        """
        Reads the files in files img_fn and target_fn
        Args:
            - img_fn (tuple of str): paths to the inputs
            - target_fn (str): path to the target file
        Output:
            - img_data (list of tensors)
            - target_data (tensor)
        """
        try: # open files
            img_fp = [rasterio.open(fn, "r") for fn in img_fn]
            target_fp = rasterio.open(target_fn, "r")
        except (RasterioIOError, rasterio.errors.CRSError):
            print("WARNING: couldn't open {} or {}".format(img_fn, target_fn))
            return None

        # read data for each input source and for the targets
        try:
            img_data = [None] * self.n_inputs
            for i, fp in enumerate(img_fp):  
                img_data[i] = np.moveaxis(fp.read(), (1, 2, 0), (0, 1, 2))
            target_data = target_fp.read(1)

        except RasterioError as e:
            print("WARNING: Error reading file, skipping to the next file")
            return None

        # close file pointers and return data
        for fp in img_fp:
            fp.close()
        target_fp.close()

        return img_data, target_data

    def _get_patches_from_tile(self, *fns):
        """Generator returning patches from one tile"""
        num_skipped_patches = 0
        # read data
        data = self._read_tile(*fns)
        if data is None:
            return #skip tile if couldn't read it

        # yield patches one by one
        for _ in range(self.num_patches_per_tile):
            data_patch, num_skipped_patches, code, _ = self._generate_patch(data, num_skipped_patches, None)
            if code == 1: #IndexError or invalid patch
                continue #continue to next patch
            yield data_patch

        if num_skipped_patches>0 and self.verbose:
            print("We skipped %d patches on %s" % (num_skipped_patches, fns[0]))

    def _stream_patches(self):
        """Generator returning patches from the samples the worker calling this function is assigned to"""
        lower_idx, upper_idx = self._get_worker_range(self.fns)
        for fns in self._stream_tile_fns(lower_idx, upper_idx): #iterate over tiles assigned to the worker
            yield from self._get_patches_from_tile(*fns) #generator

    def __iter__(self):
        if self.verbose:
            print("Creating a new StreamTrainingDataset iterator")
        return iter(self._stream_patches())
   
