import os
import glob
from tqdm import tqdm

data_folder = '/home/lyang/yl_code/dataset/MPZ14/inputmodels/obj'
shape_names = glob.glob(os.path.join(data_folder, '*.obj'))

for i, shape_name in tqdm(enumerate(shape_names), desc='NGF Training'):
    shape_name = os.path.splitext(os.path.basename(shape_name))[0]
    os.system(f'python source/train.py --mesh {shape_name} --lod 2000')
    if i > 20:
        break