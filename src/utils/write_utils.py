import os
import rasterio
import numpy as np

class Writer():
    def __init__(self, exp_utils, tilenum, template_fn, template_scale, dest_scale):
        # get rasterio profile
        with rasterio.open(template_fn, 'r') as f:
            profile = f.profile.copy()
            profile["driver"] = "GTiff"
            del profile["nodata"]
        profile = self._rescale_profile(profile, template_scale, dest_scale) 
        self.profile = profile
        self.tilenum = tilenum
        self.exp_utils = exp_utils

    def _write_hard(self, output, fn, colormap = None):
        """ 
        Write hard predictions into an image file (one bands, integer values). 
        """
        # post processing   
        data = self.exp_utils.postprocess_target(output)
        # set profile 
        profile = self.profile.copy()
        profile["dtype"] = "uint8"        
        profile["count"] = 1
        # write
        with rasterio.open(fn, "w", **profile) as f:
            f.write(data.astype(np.uint8), 1)
            if colormap is not None:
                f.write_colormap(1, colormap)

    def _write_soft(self, output, fn): 
        """Write soft predictions into an image file (n_classes bands, continuous values)"""  
        profile = self.profile.copy()
        profile["dtype"] = "float32"
        if output.ndim == 2:
            output = np.expand_dims(output, 0)
        profile["count"] = output.shape[0]
        # write
        with rasterio.open(fn, "w", **profile) as f:
            f.write(output.astype(np.float32))

    def _rescale_profile(self, in_profile, template_scale, dest_scale):
        """Adapt the profile to match the resolution corresponding to a destination scale"""
        out_profile = in_profile.copy()
        scale = template_scale/dest_scale
        if scale != 1:
            # adapt the dimensions and the transform
            out_profile['height'] = int(in_profile['height'] / scale)
            out_profile['width'] = int(in_profile['width'] / scale)
            t = in_profile['transform']
            out_profile['transform'] = rasterio.Affine(t.a * scale, t.b, t.c, t.d, t.e * scale, t.f)
        return out_profile

    def save_seg_result(self, output_dir, save_hard = True, output_hard = None, name_hard = 'predictions', 
                        save_soft = True, output_soft = None, name_soft = 'predictions_soft',
                        suffix = '', colormap = None):
        """Save segmentation predictions (hard: 1 band, integer values, soft: n_classes bands, continuous values) into 
        image file(s)"""  
            
        if save_hard:     
            output_fn = os.path.join(output_dir, '{}{}_{}.tif'.format(name_hard, suffix, self.tilenum))
            self._write_hard(output_hard, output_fn, colormap)
        if save_soft:
            output_fn = os.path.join(output_dir, '{}{}_{}.tif'.format(name_soft, suffix, self.tilenum))
            self._write_soft(output_soft, output_fn)
            
    def save_regr_result(self, output_dir, output, name = 'regr_predictions', suffix = ''):
        output_fn = os.path.join(output_dir, '{}{}_{}.tif'.format(name, suffix, self.tilenum))
        self._write_soft(output, output_fn)

    # def save_output(self, output_dir, save_hard, output_hard, save_soft = False, output_soft = None, suffix = None,
    #                 colormap = None):
    #     """
    #     Saves predictions into files.

    #     Args:
    #         - output_dir (str): directory where the predictions must be stored
    #         - save_hard (bool): whether to write hard predictions or not
    #         - output_hard (2-D ndarray): hard predictions (height x width)
    #         - save_soft (bool): whether to write soft predictions or not
    #         - output_soft (3-D ndarray): soft predictions (n_classes x height x width)
    #         - suffix (str): suffix to append to the filenames (before the extension). An underscore is automatically 
    #             prepended to the suffix.
    #     """
    #     #suffix, template_fn, tilenum_extractor, template_scale = self._setup_template_fn(suffix)
    #     suffix = '' if suffix is None  else '_' + suffix
    #     self._save_seg_result(output_dir, save_hard, save_soft, output_hard, output_soft, 
    #                     suffix = suffix, 
    #                     colormap = colormap)