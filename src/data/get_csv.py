import os
import csv
import rasterio
import numpy as np
from tqdm import tqdm

DIR = {'SI2017': '/home/tanguyen/Documents/Projects/2020/ForestMapping/Data/SwissImage/2017_25cm',
    'TLM2c': '/home/tanguyen/Documents/Projects/2020/ForestMapping/Data/TLMRaster/F',
    'TLM4c': '/home/tanguyen/Documents/Projects/2020/ForestMapping/Data/TLMRaster/OF_F_SF',
    'TLM5c': '/home/tanguyen/Documents/Projects/2020/ForestMapping/Data/TLMRaster/5c',
    'ALTI': '/home/tanguyen/Documents/Projects/2020/ForestMapping/Data/SwissALTI3D'}

PREFIX = {'SI2017': 'DOP25_LV95', 
    'TLM2c': 'TLM_F',
    'TLM4c': 'TLM_OF_F_SF',
    'TLM5c': 'TLM5c',
    'ALTI' : 'SWISSALTI3D_0.5_TIFF_CHLV95_LN02'}

SUFFIX = {'SI2017': '_2017_1.tif', 
    'TLM2c': '.tif',
    'TLM4c': '.tif',
    'TLM5c': '.tif',
    'ALTI' : '.tif'}

VAL_VIZ_ZONE = [ [(2568, 2572), (1095, 1101)],
                [(2568, 2572), (1136, 1140)],
                [(2628, 2632), (1140, 1143)]
                ]

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

def get_fn(dir, tk, tilenum_extractor):
    """
    Returns a path to a '.tif' file  given a parent folder (dir), a prefix (prefix) and a tile key (tk).
    Returns None if no file is found
    """
    for fn in os.listdir(dir):
        if os.path.splitext(fn)[-1] == '.tif':
            if tilenum_extractor(fn) == tk: # "_".join(os.path.splitext(fn)[0].replace(prefix, '').split("_")[:2]) == tk:
                return os.path.join(dir, fn)
    return None


def get_dataset_csv(input_sources, target_source, tilekeys_fns, output_fns):
    """
    Creates csv files listing input and target file paths, for any number of input and target sources

    Args:
        - input_sources (list of str): list of input sources
        - target_sources (str): target source
        - tilekeys_fns (str or list of str): paths to the csv files containing the tile keys for each set
        - output_fns (list of str): paths to the csv files to write
        - sets (str or list of str): names of the sets for which the csv files will be written (1 file per set)
    """

    # check that the directories exist
    sources = input_sources + [target_source]
    for source in sources:
        if not os.path.exists(DIR[source]):
            raise FileNotFoundError('{} does not exist'.format(DIR[source]))
        
    if not isinstance(output_fns, list):
        output_fns = [output_fns]
    if not isinstance(tilekeys_fns, list):
        tilekeys_fns = [tilekeys_fns]
    if len(output_fns) != len(tilekeys_fns):
        raise ValueError('output_fns and tilekeys_fns should have the same lengths')
    
    # read the tile keys for each set
    tilekeys = []
    for fn in tilekeys_fns:
        with open(fn, 'r') as f:
            tk_list = list(csv.reader(f))
            tilekeys.append([tk[0] for tk in tk_list])

    input_col_names = ['input'] if len(input_sources) == 1 else \
                                ['input_{}'.format(j) for j in range(len(input_sources))]

    for i in range(len(output_fns)):
        with open(output_fns[i], 'w') as f_out:
            writer = csv.writer(f_out, delimiter=',')
            writer.writerow(input_col_names + ['target'])
            print('Writing {}'.format(output_fns[i]))
            for tk in tqdm(tilekeys[i]):
                fns = []
                for source in sources:
                    fn = get_fn(DIR[source], tk, TILENUM_EXTRACTOR[source])
                    if fn is None:
                        raise FileNotFoundError('{} tile with tile key {} not found in directory {}'
                                                .format(source, tk, DIR[source]))
                    else:
                        fns.append(fn)
                writer.writerow(fns)



def get_viz_csv(input_csv_fn, output_csv_fn, tilenum_extractor, zones):
    """Filters a csv file list by keeping only tiles included in zones delimited by boundaries in arguments 'zones'"""
    with open(input_csv_fn, 'r') as f_in:
        reader = csv.reader(f_in)
        with open(output_csv_fn, 'w') as f_out:
            writer = csv.writer(f_out)
            for row in reader:
                fn = row[0]
                if os.path.splitext(fn)[-1] == '.tif':
                    x, y = tilenum_extractor(fn)
                    x, y = int(x), int(y)
                    for zone in zones:
                        (x_min, x_max), (y_min, y_max) = zone
                        if x >= x_min and x <= x_max and y >= y_min and y <= y_max:
                            writer.writerow(row)
                else:
                    writer.writerow(row)

def write_target_counts(csv_fn, n_classes, col = 'target'):
    """
    Creates a new version of file csv_fn containing class counts of the files in column col, supposing there are
    n_classes numbered from 0 to n_classes - 1.
    """
    new_csv_fn = csv_fn.replace('.csv', '_with_counts.csv')
    with open(csv_fn, 'r') as f_csv:
        reader = csv.reader(f_csv)
        with open(new_csv_fn, 'w') as f_new_csv:
            writer = csv.writer(f_new_csv)
            # update the column names
            column_names = next(reader)
            new_column_names = ['count_{}'.format(i) for i in range(n_classes)]
            writer.writerow(column_names + new_column_names)
            target_idx = column_names.index(col)
            # copy the rows and append the target counts
            for row in tqdm(reader):
                target_fn = row[target_idx]
                class_counts = [0] * n_classes
                with rasterio.open(target_fn, 'r') as f:
                    data = f.read()
                    unique, counts = np.unique(data, return_counts=True)
                    for i, target in enumerate(unique):
                        class_counts[target] = counts[i]
                writer.writerow(row + class_counts)
                
def get_filelist_csv(source, tilekeys_csv_fn, output_csv_fn):
    dir = DIR[source]
    prefix = PREFIX[source]
    suffix = SUFFIX[source]
    with open(tilekeys_csv_fn, 'r') as f_in:
        reader = csv.reader(f_in)
        with open(output_csv_fn, 'w') as f_out:
            writer = csv.writer(f_out)
            for row in reader:
                tilenum = row[0]
                fn = os.path.join(dir, '{}_{}{}'.format(prefix, tilenum, suffix))
                if os.path.exists(fn):
                    writer.writerow([fn])
                else:
                    raise FileNotFoundError('{} does not exist'.format(fn))

if __name__ == "__main__":
    
    source = 'TLM5c'
    set = 'test_viz'
    tilekeys_csv_fn = 'data/csv/TileKeys_{}.csv'.format(set)
    output_csv_fn = 'data/csv/{}_{}.csv'.format(source, set)
    get_filelist_csv(source, tilekeys_csv_fn, output_csv_fn)
    
    


