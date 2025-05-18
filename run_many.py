import os
import glob
from tqdm import tqdm

data_folder = './data/MPZ14/inputmodels/obj'
shape_paths = sorted(glob.glob(os.path.join(data_folder, '*.obj')))

wanted = [
    "amphora",
    "bimba100K",
    "bunnyBotsch",
    "Chinese_lion100K",
    "bozbezbozzel50K",
    "fat_dragon",   
    "gargoyle100K",
    "igea100k",
    "rgb_dragon",
    "dragonstand_recon100K"
]
wanted = [w.lower() for w in wanted]
num_faces = 1000
for i, shape_path in enumerate(shape_paths):

    shape_name = os.path.splitext(os.path.basename(shape_path))[0]
    if (shape_name.lower() in wanted):
        print(shape_name)
        os.system(f'python source/train.py --mesh {shape_path} --lod {num_faces}')