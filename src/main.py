import os
from PIL import Image

if __name__ == "__main__":
    data_path = '/media/data/charrez/'
    dataset_path = 'SwissIMAGE/'
    path = data_path+dataset_path

    entries = os.listdir(path)

    files_path = path + entries[0] + '/'
    tif_files = os.listdir(files_path)

    file_path = files_path + tif_files[0]
    tif_file = Image.open(file_path)
    tif_file.show()