import csv
import os

#### Script to transfer files to scitas /scratch/izar/ ########################

file_list_fn = 'data/SI2017_ALTI_TLM_OF_F_val.csv'  # files to transfer
temp_file_list_fn = 'data/izar_transfer.csv'        # file where the destination path of each file is written

start = 0   # starting row of the file chunk to transfer
stop = 3000 # stopping row of the file chunk to transfer

with open(file_list_fn, 'r') as f_in:
    reader = csv.reader(f_in)
    next(reader) #ignore columns names
    with open(temp_file_list_fn, 'w') as f_out:
        writer = csv.writer(f_out)
        for i, row in enumerate(reader):
            if i >= start and i < stop:
                for fn in row:
                    if os.path.isfile(fn):
                        fn = fn.replace('/home/', '')
                        writer.writerow([fn])

cmd = 'rsync -az --progress --files-from={} /home tanguyen@izar.epfl.ch:/scratch/izar/ --ignore-existing'.format(temp_file_list_fn)
os.system(cmd)
