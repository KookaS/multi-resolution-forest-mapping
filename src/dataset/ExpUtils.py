import numpy as np
import torch
import os
from math import gcd, nan
import pandas as pd
import psutil
import gc
from copy import copy

############## Constants and datasource-specific parameters ###################

# means and standard deviations
MEANS = {'SI2017': [ 98.01336916, 106.46617234, 93.43728537], 
        'ALTI' : 1878.01851825,
        'VHM': 3.90032556,
        'TH': 4.95295663,
        'TCDCopHRL': 21.18478328,
        'TCD1': 29.91515737,
        'TCD2': 26.84415381}
MEANS['TCD'] =  MEANS['TCDCopHRL'] # for backward compatibility

STDS = {'SI2017': [54.22041366, 52.69225063, 46.55903685], 
        'ALTI' : 1434.79671951,
        'VHM': 7.52012624,
        'TH': 8.5075463,
        'TCDCopHRL': 33.00766675,
        'TCD1': 37.9657575,
        'TCD2': 37.10486429}
STDS['TCD'] =  STDS['TCDCopHRL'] # for backward compatibility

# nodata value
I_NODATA_VAL = 255 #nodata value for integer arrays/rasters
F_NODATA_VAL = -1 #nodata value for float arrays/rasters

NODATA_VAL = {'SI2017': None,
                'TLM3c' : None,
                'TLM4c' : None,
                'TLM5c' : None,
                'ALTI' : None,
                'VHM' : -3.4028234663852886e+38,
                'TH' : -3.4028234663852886e+38,
                'TCDCopHRL': 240.0,
                'TCD1' : -1,
                'TCD2' : -1,
                'hard_predictions': I_NODATA_VAL,
                'soft_predictions': np.finfo(np.float32).max}

# maximum values used for clipping regression predictions at inference (minimum value is assumed to be 0)
CLIP_PRED_VAL = {'TH': 100, 'TCD': 100}

# value above which regression targets are ignored
IGNORE_TARGET_VAL = {   'TH': None, #40, 
                        'TCD': None}

# operators to use to check nodata
NODATA_CHECK_OPERATOR = {'SI2017': ['all', 'all'], # operators used to skip a training patch
                        'ALTI': ['all', 'all'],
                        'TLM3c': 'any',
                        'TLM4c': 'any',
                        'TLM5c': 'any',
                        'TH' : 'all',
                        'TCD' : 'all'
                        }
GET_OPERATOR = {'any': np.any, 'all': np.all}

# relative resolution of the datasources
RELATIVE_RESOLUTION = {'SI2017': 4, 'ALTI': 2, 'TLM3c': 1, 'TLM4c': 1, 'TLM5c': 1, 'VHM': 1, 'TH': 1,'TCD': 1}

# number of channels
CHANNELS = {'SI2017': 3, 'ALTI' : 1}

# class names
CLASS_NAMES = {'PresenceOfForest' : ['NF', 'F'], 'TLM3c': ['NF', 'OF', 'CF'], 'TLM4c': ['NF', 'OF', 'CF', 'SF'], 
                'ForestType': ['OF', 'CF', 'SF'], 'TH': None, 'TCD': None}

# number of classes
N_CLASSES = {'PresenceOfForest' : 2, 'TLM3c': 3, 'TLM4c': 4, 'ForestType': 3, 'TH': None, 'TCD': None}

# thresholds for intermediate variables
THRESHOLDS = {'TH': [1.0, 3.0], 'TCD': [20.0, 60.0], 'VHM': [20.0, 60.0]}

# classification tables
# CLASS_TABLES = {'VHM_TCD': torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
#                                             [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]]])}

# CLASS_TABLES = {'VHM_TCD': torch.log(torch.tensor([[[1.0 - 3*eps, eps, eps, eps], [1.0 - 3*eps, eps, eps, eps], [1.0 - 3*eps, eps, eps, eps]],
#                                             [[1.0 - 3*eps, eps, eps, eps], [eps, 1.0 - 3*eps, eps, eps], [eps, eps, 0.5 - eps, 0.5 - eps]]])) + C }
# RULES = {'VHM_TCD': torch.tensor([0, 0, 0,
#                                   0, 1, 2])}

#                             TCD   <20       [20, 60)    >= 60]    TH
RULES = {'TH_TCD': torch.tensor([  0,        0,          0,        # < 1
                                    0,        0,          2,        # [1, 3)
                                    0,        1,          3])}      # >= 3
eps, C = 1e-3, 3 #1e-3, 3
PROB_ENCODING = {
    'f':              # NF              OF              CF          SF
    torch.tensor([  [   1.0 - 3*eps,    eps,            eps,        eps         ],      # code 0: non-forest
                    [   eps,            1.0 - 3*eps,    eps,        eps         ],      # code 1: open forest
                    [   0.5 - eps,      eps,            eps,        0.5 - eps   ],      # code 2: non-forest or shrub forest
                    [   eps,            eps,            0.5 - eps,  0.5 - eps   ]]),    # code 3: closed forest or shrub forest

    'h' :             # OF              CF              SF           F (all types)           
    torch.tensor([  [   1.0/3.0,        1.0/3.0,        1.0/3.0,     eps       ],   # code 0: non-forest
                    [   1.0 - 2*eps,    eps,            eps,         1.0 - eps ],   # code 1: open forest
                    [   eps,            eps,            1.0 - 2*eps, 0.5       ],   # code 2: non-forest or shrub forest
                    [   eps,            0.5 - eps/2,    0.5 - eps/2, 1.0 - eps ]])} # code 3: closed forest or shrub forest
#ACT_ENCODING = torch.log(PROB_ENCODING) + C

# TLM translation for sub-tasks
nodata_mapping = np.full(251, fill_value = I_NODATA_VAL)
#                                                                       NF            OF  CF  SF  Gehoelzflaeche
TARGET_CONVERSION_TABLE = { 'PresenceOfForest':        np.concatenate((np.array([  0,            1,  1,  1,  1]), 
                                                            nodata_mapping)),
                            'TLM4c':        np.concatenate((np.array([  0,            1,  2,  3,  I_NODATA_VAL]),
                                                            nodata_mapping)),
                            'ForestType':   np.concatenate((np.array([  I_NODATA_VAL, 0,  1,  2,  I_NODATA_VAL]), 
                                                            nodata_mapping))}

# target colormap
COLORMAP = {'PresenceOfForest': {  
                        0: (0, 0, 0, 0),
                        1: (255, 255, 255, 255),
                        },
            'TLM3c': { 
                        0: (0, 0, 0, 0),
                        1: (21, 180, 0, 255),
                        2: (25, 90, 0, 255)
                        },
            'TLM4c': { 
                        0: (0, 0, 0, 0),
                        1: (21, 180, 0, 255),
                        2: (25, 90, 0, 255),
                        3: (151, 169, 93, 255)
                        },
            'ForestType': 
                        { 
                        0: (21, 180, 0, 255),
                        1: (25, 90, 0, 255),
                        2: (151, 169, 93, 255)
                        },
            'TH': None, 'TCD': None        
            }

# class frequencies (used to weight the loss)
CLASS_FREQUENCIES = {
    'PresenceOfForest' : { # non-forest, forest (the latter including open forest, closed forest, shrub forest and forest patches)
        'all': {'train': np.array([0.7332189312030073, 0.2667810687969927])},
        'positives': {'train': np.array([0.6044306120401336, 0.3955693879598663])}
                },
    'TLM3c': { # non-forest, open forest, closed forest (WRONG, forest patches considererd as non-forest)
        'all': {    'train': np.array([0.7534807958646614, 0.016324748120300762, 0.23019445601503774]),
                    'val': np.array([0.7035771146711639, 0.0186117234401349, 0.2778111618887016]),
                    'test': np.array([0.7641926017830614, 0.015225793462109956, 0.22058160475482902])
                },
        'positives' : {'train': np.array([0.6042690374170179, 0.02621146771273391, 0.3695194948702477]),
                    'val': np.array([0.5733500048543688, 0.02679510194174758, 0.3998548932038842]),
                    'test': np.array([0.6230919406175778, 0.024336940617577196, 0.3525711187648456])
                    }
                },    
    'TLM4c': { # non-forest, open forest, closed forest, shrub forest 
        'all': {
            'train': np.array([0.7345804185679481, 0.016352612423765258, 0.23061658770687501, 0.018450381301411623]),
                },
        'positives' : {
            'train': np.array([0.6060958648207276, 0.024271733586653965, 0.34225351010837374, 0.027378891484244623]),
                    }
            },
    'TLM5c': {  # non-forest, open forest, closed forest, shrub forest, forest patches/nodata 
                # (frequencies for the last class are not used)
            'all': {
            'train': np.array([0.7332189312030076, 0.016322304135338337, 0.23018915789473723, 0.01841618496240605, 0.0018534218045112767]),
            'val' : np.array([0.6846235885328839, 0.018615264755480594, 0.27776981450252985, 0.017019370994940983, 0.001971961214165261]),
            'test' : np.array([0.7481924457652306, 0.015229600297176828, 0.22057757800891528, 0.014307291233283791, 0.0016930846953937583])
                },/media/data/charrez/SwissIMAGE
/media/data/charrez/SwissALTI3D
/media/data/charrez/Labels
        'positives' : {
            'train': np.array([0.6044306120401334, 0.024205046822742493, 0.34131316610925294, 0.027303667781493884, 0.0027475072463768136]),
                    }
            },
    'ForestType' : { # open forest, closed forest, shrub forest
                'all': {
                    'train': np.array([0.06161042201760675, 0.8688755609612528, 0.06951401702114042]),
                },
                'positives': { 
                    'train': np.array([0.061618377211519984, 0.8688751387507229, 0.06950648403775714])}
            }       
        }

# methods to extract the tile number from the filename
default_tilenum_extractor = lambda x: os.path.splitext('_'.join(os.path.basename(x).split('_')[-2:]))[0]
TILENUM_EXTRACTOR = {'SI2017': lambda x: '_'.join(os.path.basename(x).split('_')[2:4]),
                    'ALTI': default_tilenum_extractor,
                    'TLM3c': default_tilenum_extractor,
                    'TLM4c': default_tilenum_extractor,
                    'TLM5c': default_tilenum_extractor,
                    'VHM': default_tilenum_extractor,
                    'TH': default_tilenum_extractor,
                    'TCDCopHRL': default_tilenum_extractor,
                    'TCD1': default_tilenum_extractor,
                    'TCD2': default_tilenum_extractor}

IGNORE_INDEX = I_NODATA_VAL
IGNORE_FLOAT = F_NODATA_VAL #np.finfo(np.float32).max

############## ExpUtils class ##############################################

class ExpUtils:
    """
    Class used to define all parameters and methods specific to the data sources (inputs and targets) and the current
    experiment
    """

    def __init__(self, input_sources, aux_target_sources = [], target_source = None, decision = 'f'):
        """
        Args:
            - inputs_sources (list of str): input sources
            - aux_target_sources (list of str): sources to use as target for auxiliary regression task
            - target_source (str): main classification target source
            - decision (str): decision type, 'f' for all decisions at the same level (e.g. non-forest, open forest, 
                closed forest, shrub forest), 'h' for a 2-step decision (forest/non-forest then forest type)
        """

        # Get methods and parameters corresponding to input and target sources
        self.n_input_sources = len(input_sources)
        self.n_aux_targets = len(aux_target_sources)
        self.sem_bot = self.n_aux_targets > 0 # sem_bot = semantic bottleneck

        self.input_channels = [CHANNELS[source] for source in input_sources]

        self.tilenum_extractor = [TILENUM_EXTRACTOR[source] for source in input_sources + aux_target_sources \
                                    + [target_source]]

        self.input_means = [MEANS[source] for source in input_sources]
        self.input_stds = [STDS[source] for source in input_sources]

        self.input_nodata_val = [NODATA_VAL[source] for source in input_sources]
        self.target_nodata_val = NODATA_VAL[target_source]

        self.input_nodata_check_operator = [[GET_OPERATOR[op] for op in NODATA_CHECK_OPERATOR[source]] \
                                                for source in input_sources]
        self.target_nodata_check_operator = GET_OPERATOR[NODATA_CHECK_OPERATOR[target_source]]

        

        if self.sem_bot:
            self.aux_target_means = [MEANS[source] for source in aux_target_sources]
            self.aux_target_stds = [STDS[source] for source in aux_target_sources]
            self.aux_target_nodata_val = [NODATA_VAL[source] for source in aux_target_sources]
            self.aux_target_sources = aux_target_sources

            # aux_target_sources specifies the data used as target (there can be different options for a same variable)
            # aux_variables specifies the targeted variables
            aux_variables = copy(aux_target_sources)
            for i in range(len(aux_target_sources)):
                if aux_target_sources[i].startswith('TCD'):
                    aux_variables[i] = 'TCD'
            self.aux_variables = aux_variables
            

            self.aux_target_nodata_check_operator = [GET_OPERATOR[NODATA_CHECK_OPERATOR[source]] for source in \
                                                    aux_variables]
            self.aux_target_ignore_val = [IGNORE_TARGET_VAL[source] for source in aux_variables]

        # compute relative scale (resolution) of the data sources
        sources = input_sources + aux_variables + [target_source] if self.sem_bot else input_sources + [target_source]
        scales = [RELATIVE_RESOLUTION[source] for source in sources]
        scale_min = min(scales)
        rem = [scale%scale_min for scale in scales]
        if any(rem):
            raise RuntimeError('All the pixel sizes should be multiples of the smallest pixel size. Other cases are '
                                'not supported')
        scales = [scale//scale_min for scale in scales]
        self.input_scales = scales[:len(input_sources)]
        self.target_scale = scales[-1]
        if self.sem_bot:
            self.aux_target_scales = scales[len(input_sources):-1]

        # setup output(s)
        self.class_names = {}
        self.class_freq = {}
        if target_source == 'TLM5c': 
            # assume we don't want to predict the class forest patch / gehoelzflaeche
            # (only use it for nodata information for example)
            target = 'TLM4c' 
            # specify paraneters for the second decision tree layer as well (forest type)
            if decision == 'h':
                target_1 = 'ForestType'
                self.n_classes_1 =  N_CLASSES[target_1]
                self.class_names['seg_1'] = CLASS_NAMES[target_1]
                if self.sem_bot:
                    self.class_names['seg_rule_1'] = CLASS_NAMES[target_1]
                self.class_freq['seg_1'] = CLASS_FREQUENCIES[target_1]
                self.colormap_1 = COLORMAP[target_1]
                self.suffix_1 = '_ForestType'
        else:
            target = target_source
        self.n_classes = N_CLASSES[target] 
        self.class_freq['seg'] = CLASS_FREQUENCIES[target]
        self.colormap = COLORMAP[target]
        self.class_names['seg'] = CLASS_NAMES[target]
        self.decision_func = self.argmax_decision
                    
        # setup binary output
        if self.n_classes > 2: 
            # add an additional output for binary forest/non-forest prediction (might be supervised or just for metrics)
            target_2 = 'PresenceOfForest'
            self.suffix_2 = '_2c'
            self.class_names['seg_2'] = CLASS_NAMES[target_2]
            if self.sem_bot:
                self.class_names['seg_rule_2'] = CLASS_NAMES[target_2]
            self.colormap_2 = COLORMAP[target_2]
            self.n_classes_2 = 2
            if decision == 'f':
                # the binary prediction is not supervised, but computed only at inference for metrics
                self.target_recombination = self.target_binary_recombination
                self.decision_func_2 = self.argmax_binary_decision
            else:
                # the binary prediction is supervised (first level of the decision tree)
                self.class_freq['seg_2'] = CLASS_FREQUENCIES['PresenceOfForest']
                self.decision_func_2 = self.binary_decision
        else: # the model has only one output corresponding to binary forest/non-forest prediction
            self.decision_func_2 = None
            self.target_recombination = None

        if target_source == 'TLM5c' and decision == 'h':
            # +1 for the binary task
            self.output_channels = self.n_classes_1 + 1 #self.n_classes_2
        else:
            self.output_channels = self.n_classes 

        

        # target preprocessing for training
        if target_source == 'TLM5c':
            if decision == 'h':
                self.preprocess_training_target = self.preprocess_training_hierarchical_target
            else: 
                self.preprocess_training_target = lambda x : torch.from_numpy(TARGET_CONVERSION_TABLE['TLM4c'][x]) #.unsqueeze(0)
        else:
            self.preprocess_training_target = lambda x : torch.from_numpy(x).long() #.unsqueeze(0)

        # target preprocessing for inference
        if target_source == 'TLM5c' and decision == 'h':
            self.preprocess_inference_target = self.preprocess_inference_hierarchical_target
        else: # same processing function for training and inference
            self.preprocess_inference_target = lambda x : x

        
            
        # set patch parameters for inference
        self.patch_size = 128 #patch size in the coarsest input/target
        self.num_patches_per_tile = 32 #64 #(1000/128)^2 = 61
        self.padding = self.patch_size // 4 #self.patch_size // 4 #must be even
        self.patch_stride = self.patch_size - self.padding
        self.kernel_std = 16
        self.tile_margin = 64

        # parameters for auxiliary targets
        if self.sem_bot:
            self.unprocessed_thresholds = [np.array(THRESHOLDS[source]) for source in aux_variables]
            self.thresholds = self.preprocess_thresholds(self.unprocessed_thresholds)
            # self.aux_target_max_val = [CLIP_PRED_VAL[source] for source in aux_variables]
            # set number of channels, class names and frequencies
            self.class_names['seg_rule'] = self.class_names['seg']
            self.aux_channels = [0] * len(aux_variables)
            for i in range(self.n_aux_targets):
                key = 'regr_{}'.format(i)
                thresholds = self.unprocessed_thresholds[i].astype(np.uint8)
                class_names = ['<{}'.format(thresholds[0])]
                for j in range(len(thresholds)-1):
                    class_names.append('[{},{})'.format(thresholds[j], thresholds[j+1]))
                class_names.append('>={}'.format(thresholds[-1]))
                self.class_names[key] = class_names
                self.aux_channels[i] = 1 
                    
            self.corr_channels = self.output_channels - 1
            self.rules = RULES['_'.join(aux_variables)]
            self.prob_encoding = PROB_ENCODING[decision]
            self.rule_decision_func = self.argmax_randtie_decision
            if decision == 'f':
                self.act_encoding = torch.log(self.prob_encoding) + C
            else:
                self.act_encoding = torch.cat((torch.log(self.prob_encoding[:, :-1]) + C, 
                                            torch.logit(self.prob_encoding[:, -1:], eps=None)), 
                                            dim = 1)
                #self.rule_decision_func_2 = self.binary_decision
            self.rule_decision_func_2 = self.decision_func_2
            self.preprocess_training_aux_targets = [self.preprocess_training_aux_regr_target] * self.n_aux_targets # TODO
            self.preprocess_inference_aux_targets = [self.preprocess_inference_aux_regr_target] * self.n_aux_targets
            # self.preprocess_aux_targets = lambda aux_target_list: [f(aux_target, i) 
            #                                                             for i, (f, aux_target) in enumerate(zip(self.preprocess_aux_target, aux_target_list))]
        
        # nodata values for writing output rasters
        self.i_out_nodata_val = NODATA_VAL['hard_predictions']
        self.f_out_nodata_val = NODATA_VAL['soft_predictions']
        
        # nodata values for internal arrays/tensors (i.e. after pre-processing and before post-processing)
        self.i_nodata_val = I_NODATA_VAL
        self.f_nodata_val = F_NODATA_VAL

    ################# Methods for pre/post-processing #########################
    
    def preprocess_input(self, img, idx):
        """
        Number of channels should be the last dimension for the broadcasting to work.
        A nodata mask must be computed before this function, and use to avoid backpropagating loss on nodata pixels.
        """
        return torch.from_numpy(((img - self.input_means[idx]) / self.input_stds[idx] )).movedim((2, 0, 1), (0, 1, 2)).float()

    def preprocess_inputs(self, img_list):
        return [self.preprocess_input(img, i) for i, img in enumerate(img_list)]
        # return  [torch.from_numpy(((img_list[i] - self.input_means[i]) / self.input_stds[i] ).astype(np.float32))
        #             .movedim(-1, 0) for i in range(len(img_list))]

    def preprocess_training_hierarchical_target(self, target):
        # nodata values are automatically restored/added using the conversion tables
        h, w = target.shape
        target_out = np.empty((2, h, w), dtype = np.int64)
        target_out[0] = TARGET_CONVERSION_TABLE['ForestType'][target]
        target_out[1] = TARGET_CONVERSION_TABLE['PresenceOfForest'][target]
        return torch.from_numpy(target_out)

    def preprocess_inference_hierarchical_target(self, target):
        # nodata values are automatically restored/added using the conversion tables
        h, w = target.shape
        target_out = np.empty((3, h, w), dtype = np.int64)
        target_out[0] = TARGET_CONVERSION_TABLE['ForestType'][target]
        target_out[1] = TARGET_CONVERSION_TABLE['PresenceOfForest'][target]
        target_out[2] = TARGET_CONVERSION_TABLE['TLM4c'][target]
        return target_out # torch.from_numpy(target_out)

    def preprocess_training_aux_regr_target(self, aux_target, nodata_in=F_NODATA_VAL, idx=0):
        """For regression of continuous values"""
        #result_i = (aux_target - self.aux_target_means[idx]) / self.aux_target_stds[idx] 
        result_i = aux_target / self.aux_target_stds[idx]
        # set pixels to be ignored by the loss 
        if nodata_in is not None:
            if nodata_in != self.f_nodata_val:
                result_i[aux_target == nodata_in] = self.f_nodata_val
            # if self.aux_target_ignore_val[idx] is not None:
            #     result_i[aux_target > self.aux_target_ignore_val[idx]] = self.f_nodata_val
        return torch.from_numpy(result_i).float()
    
    def preprocess_inference_aux_regr_target(self, aux_target, *args, **kwargs): #nodata_in=F_NODATA_VAL, *args, **kwargs):
        """For regression of continuous values"""
        # set pixels to be ignored by the loss 
        # if nodata_in is not None:
        #     if nodata_in != self.f_nodata_val:
        #         aux_target[aux_target == nodata_in] = self.f_nodata_val
            # if self.aux_target_ignore_val[idx] is not None:
            #     result_i[aux_target > self.aux_target_ignore_val[idx]] = self.f_nodata_val
        return aux_target #torch.from_numpy(aux_target).float()

    def preprocess_training_aux_ord_regr_target(self, aux_target, nodata_in=F_NODATA_VAL, idx=0):
        # for ordinal regression
        comp = (torch.from_numpy(
                aux_target) >= self.thresholds[idx].unsqueeze(-1).unsqueeze(-1).repeat(1, *aux_target.shape
                                )).long()
        #mask = torch.tensor(aux_target == self.aux_target_nodata_val[idx]).unsqueeze(0).repeat(comp.shape[0], 1, 1)
        if nodata_in is not None:
            if nodata_in != self.i_nodata_val:
                mask = torch.tensor(aux_target == nodata_in).unsqueeze(0).repeat(comp.shape[0], 1, 1)
                comp[mask] = self.i_nodata_val
        return comp
    
    def preprocess_inference_aux_ord_regr_target(self, aux_target, idx=0):#nodata_in=I_NODATA_VAL, idx=0):
        # for ordinal regression
        comp = aux_target >= self.thresholds[idx].unsqueeze(-1).unsqueeze(-1).repeat(1, *aux_target.shape).numpy()
        cat = np.sum(comp, axis = 0)
        #mask = torch.tensor(aux_target == self.aux_target_nodata_val[idx]).unsqueeze(0).repeat(comp.shape[0], 1, 1)
        # if nodata_in is not None:
        #     if nodata_in != self.i_nodata_val:
        #         mask = aux_target == nodata_in
        #         cat[mask] = self.i_nodata_val
        return cat
    
    def preprocess_thresholds(self, thresholds):
        new_thresholds = [None] * len(thresholds)
        for i in range(len(thresholds)):
            t = thresholds[i] / self.aux_target_stds[i] 
            new_thresholds[i] = torch.from_numpy(t).float()
        return new_thresholds
        # return [torch.from_numpy(((thresholds[i] - self.aux_target_means[i]) / self.aux_target_stds[i] )
        #         .astype(np.float32)) for i in range(len(thresholds))]

    def postprocess_target(self, targets):
        return targets

    def postprocess_regr_predictions(self, pred, idx):
        """postprocesses regression predictions for idx-th variable (1 channel)"""
        # return np.clip(pred * self.aux_target_stds[idx] + self.aux_target_means[idx], 0, self.aux_target_max_val[idx])
        return pred * self.aux_target_stds[idx] #+ self.aux_target_means[idx]
    
    def postprocess_ord_regr_predictions(self, pred):
        return pred

    ######################## Methods to check nodata ##########################

    def inputs_nodata_check(self, *inputs):
        """
        each input should have 3 dimensions (height, width, bands)
        """
        check = False
        for i in range(len(inputs)):
            op1, op2 = self.input_nodata_check_operator[i]
            if self.input_nodata_val[i] is None:
                check = check or False
            else:
                data = np.reshape(inputs[i], (-1, inputs[i].shape[-1])) # flatten along height and width
                check = check or op2(op1(data == self.input_nodata_val[i], axis = -1), axis = 0)
        return check

    @staticmethod
    def single_band_nodata_check(data, nodata_val, nodata_check_operator):
        """
        data should have 2 dimensions (height, width)
        """
        if nodata_val is None:
            return False
        else:
            check = nodata_check_operator(np.ravel(data) == nodata_val)
        return check

    def target_nodata_check(self, data):
        return self.single_band_nodata_check(data, self.target_nodata_val, self.target_nodata_check_operator)

    def aux_targets_nodata_check(self, data):
        return any(list(map(self.single_band_nodata_check, 
                            data, 
                            self.aux_target_nodata_val, 
                            self.aux_target_nodata_check_operator)))

    ######## Methods for converting soft predictions to hard predictions ######

    def argmax_decision(self, output):
        output_hard = output.argmax(axis=0).astype(np.uint8)
        return output_hard
    
    @staticmethod
    def random_num_per_grp_cumsumed(L):
        # For each element in L pick a random number within range specified by it
        # The final output would be a cumsumed one for use with indexing, etc.
        r1 = np.random.rand(np.sum(L)) + np.repeat(np.arange(len(L)),L)
        offset = np.r_[0,np.cumsum(L[:-1])]
        return r1.argsort()[offset]

    def argmax_randtie_decision(self, output): 
        max_mask = output==output.max(axis=0,keepdims=True)
        n_max = max_mask.sum(axis=0)
        
        if np.all(n_max == 1): # no ties
            return self.argmax_decision(output)
        else:
            # set_mask = np.zeros(n_max.sum(), dtype=bool)
            # select_idx = self.random_num_per_grp_cumsumed(n_max)
            # set_mask[select_idx] = True
            # max_mask.T[max_mask.T] = set_mask
            # return max_mask.argmax(axis=0) 
            noise = np.random.rand(*output.shape)
            arr = output + noise * max_mask
            return arr.argmax(axis=0).astype(np.uint8)

    def binary_decision(self, output):
        output_hard = (output > 0.5).astype(np.uint8)
        return output_hard
    
    def binary_randtie_decision(self, output):
        eps = np.random.normal(0, loc=0.0, scale=1e-3, size=output.shape) 
        noisy_output = output 
        noisy_output[output == 0.5] += eps[output == 0.5] # TODO check that output is not modified
        output_hard = (noisy_output > 0.5).astype(np.uint8)
        return output_hard

    def argmax_binary_decision(self, output):
        """
        Takes a binary argmax decision on soft predictions with a number of classes greater than 2:
            - new class 0 corresponds to original class 0
            - new class 1 corresponds to all classes other than 0 (their logits are summed up)
        """
        #binary_prob = torch.cat((final_prob[:,0:1], torch.sum(final_prob[:,1:], axis = 1, keepdim=True)), axis = 1)
        bin_output = np.concatenate((output[0:1], np.sum(output[1:], axis = 0, keepdims=True)), axis = 0)
        #pred = output[:2].copy()
        #pred[1] += np.sum(output[2:], axis = 0)
        output_hard = bin_output.argmax(axis=0).astype(np.uint8)
        return output_hard
    
    @staticmethod
    def ord_regr_decision(prob):
        #cat = np.sum((prob > 0.5).astype(np.uint8), axis = 0)
        cat = np.sum((prob > 0.5).astype(np.int64), axis = 0)
        return cat

    def target_binary_recombination(self, targets):
        """Convert ndarray targets to a binary case"""
        new_targets = (targets > 0).astype(np.int64)
        return new_targets

    ######## Other methods ####################################################

    @staticmethod
    def get_weights(class_freq, n_pos = None, n_neg = None):
        """
        Compute class weights (for weighting the loss) with weight = 1/frequency

        Args:
            - n_pos (int): number of positive examples (i.e. examples with a least one non-zero pixel)
            - n_neg (int): number of negative examples (i.e. examples containing class 0 only)
        """
        if n_pos is None and n_neg is None:
            # default weights (correspond to using the full dataset)
            prob = class_freq['all']['train']
        elif n_pos is not None and n_neg is not None: 
            # correct the class probabilities, assuming class 0 is the "negative" class
            pos_class_freq = class_freq['positives']['train']
            neg_class_freq = np.zeros_like(pos_class_freq)
            neg_class_freq[0] = 1.0
            prob = (pos_class_freq * n_pos + neg_class_freq * n_neg) / (n_pos + n_neg)
        else:
            raise ValueError('Both n_pos and n_neg should be specified')
        return np.max(prob) / prob

    # def get_binary_weights(self, n_pos = None, n_neg = None):
    #     """
    #     Compute class weights (for weighting the loss) with weight = 1/frequency, for a binary task

    #     Args:
    #         - n_pos (int): number of positive examples (i.e. examples with a least one non-zero pixel)
    #         - n_neg (int): number of negative examples (i.e. examples containing class 0 only)
    #     """
    #     if n_pos is None and n_neg is None:
    #         # default weights (correspond to using the full dataset)
    #         prob = self.class_freq['all']['train']
    #         bin_prob = prob[:2]
    #         bin_prob[1] += sum(prob[2:])
    #     elif n_pos is not None and n_neg is not None:
    #         # compute class probabilities
    #         prob = self.class_freq['all']['train']
    #         bin_prob_pos = prob[:2]
    #         bin_prob_pos[1] += sum(prob[2:])
    #         bin_prob = (bin_prob_pos * n_pos + np.array([1.0, 0.0]) * n_neg) / (n_pos + n_neg)
    #     else:
    #         raise ValueError('Both n_pos and n_neg should be specified')
    #     return np.max(bin_prob) / bin_prob


    def create_kernel(self,scale):
        # create a 2D gaussian kernel
        size = self.patch_size*scale
        ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
        gauss = np.exp(-0.5 * np.square(ax) / np.square(self.kernel_std))
        kernel = np.outer(gauss, gauss)
        return kernel / np.min(kernel)

    def get_inference_kernel(self):
        kernel = self.create_kernel(self.target_scale)
        # if self.sem_bot:
        #     aux_kernels = list(map(self.create_kernel, self.aux_target_scales))
        #     return kernel, aux_kernels
        # else:
        #     return kernel
        return kernel

    




  
        