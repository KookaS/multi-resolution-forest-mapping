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