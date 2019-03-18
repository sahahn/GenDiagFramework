import nibabel as nib
import os
import numpy as np

def smart_load(path):

    if os.path.isfile(path):
        file = nib.load(path)
    
    elif 'nii.gz' in path and os.path.isfile(path.replace('.gz','')):
        file = nib.load(path.replace('.gz',''))

    elif 'nii' in path and os.path.isfile(path + '.gz'):
        file = nib.load(path + '.gz')

    else:
        file = None

    return file

def read_t_transform(loc):
    with open(loc, 'r') as f:
        
        lines = f.readlines()
        lines = lines[-3:]
        lines = [line.rstrip().replace(';', '').split() for line in lines]
        lines = [[float(l) for l in line] for line in lines]
        lines.append([0, 0, 0, 1])
        
    return np.array(lines)
    