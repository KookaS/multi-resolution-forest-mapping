# multi-resolution-forest-mapping
Multiple U-Net encoders mapping different resolution images into the same feature space for semantic segmentation of the Forest Tree Line in the Alps. The data used is from Swisstopo.

Original code from: https://github.com/thienanhng/ForestMapping/blob/main/launch.sh

## install

    conda env create -f environment.yml

or

    conda create -n myenv python
    conda install -n myenv gdal rasterio pandas torch torchvision tqdm 
    conda activate myenv
