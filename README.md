# multi-resolution-forest-mapping
Multiple U-Net encoders mapping different resolution images into the same feature space for semantic segmentation of the Forest Tree Line in the Alps. The data used is from Swisstopo.

Original code from: https://github.com/thienanhng/ForestMapping/blob/main/launch.sh

## install

    conda env create -f myenv.yml

or

    conda create -n myenv python=3.10.0

packages: gdal rasterio pandas torch torchvision tqdm numpy psutil opencv-python
    conda install -n myenv <package_name>
    conda activate myenv

## guide

Train is the training process to train the two encoders and decoder.
Infer is the validation process to evaluate the performance of the trained model.

## jupyter notebook

    jupyter notebook --no-browser --port=8080
    ssh -L 8080:localhost:8081 <REMOTE_USER>@<REMOTE_HOST>

## args

Check in infer.py and train.py the arguments for more help

## csv

for images SI2017 and ALTI, target TLM5c:

input_0,input_1,target
    /media/data/charrez/SwissImage/2017_25cm/DOP25_LV95_2614_1093_2017_1.tif,/media/data/charrez/SwissALTI3D/SWISSALTI3D_0.5_TIFF_CHLV95_LN02_2614_1093.tif,/media/data/charrez/TLMRaster/5c/TLM5c_2614_1093.tif

# TODO

    - fix plot
    - test different grayscale and noise
    - check predictions are good between 1946 and 2017
    - metric segmentation, metric accord between 1946 and 2017 on binary classification with mean(arr1 == arr2), forest classification, segmentation. Compare probabilities with L2