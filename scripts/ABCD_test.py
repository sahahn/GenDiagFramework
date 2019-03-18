import keras
import numpy as np
from DataLoaders.ABCD_DataLoader import ABCD_DataLoader

main_dr = '/home/sage/GenDiagFramework/'

dl = ABCD_DataLoader(
                 init_location = '/home/sage/FS2/FS_data_release_2/',
                 label_location = main_dr + 'labels/adhd_scores.csv',
                 label_key='NDAR',
                 file_key='brain.finalsurfs.mgz',
                 input_size=(160,192,192,1),
                 load_segs=False,
                 segs_key='aparc.a2009s+aseg.mgz',
                 tal_transform=True,
                 tal_key='talairach.xfm',
                 limit=None,
                 in_memory=False,
                 memory_loc='/home/sage/temp/',
                 compress=False,
                 preloaded=False
                 )

X = dl.get_all()