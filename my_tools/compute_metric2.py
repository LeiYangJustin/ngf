import sys
import os
sys.path.append(os.getcwd())

import numpy as np
from glob import glob
import trimesh
from common_tools.io_tools import draw_colored_points_to_obj, read_json, write_json, write_line_file2
import argparse
import scipy.spatial.distance as distance_func

import matplotlib
from matplotlib import cm

def compute_point2point_error(gt_mesh, xx_mesh, sample_size=30000):
    gt_points, _ = trimesh.sample.sample_surface(gt_mesh, sample_size)
    xx_points, _ = trimesh.sample.sample_surface(xx_mesh, sample_size)
    error = distance_func.cdist(gt_points, xx_points) ## dim: sample_size x sample_size
    
    ## chamfer distance
    min_error_1 = error.min(axis = 0) ## min_error_1 agains each xx_points  
    min_error_2 = error.min(axis = 1) ## min_error_2 agains each gt_points

    cmf = (np.mean(min_error_1) + np.mean(min_error_2))/2
    hd = (min_error_1.max() + min_error_2.max())/2

    return cmf, hd

def compute_metrics(gt_mesh, gt_pq, xx_mesh, sample_size=30000):

    gt_points, fids = trimesh.sample.sample_surface(gt_mesh, sample_size)
    gt_normals = gt_mesh.face_normals[fids]

    xx_points, fids = trimesh.sample.sample_surface(xx_mesh, sample_size)
    xx_normals = xx_mesh.face_normals[fids]
    print("sampling done")

    d1 = None
    nae_deg1 = None
    if True:
        _, d1, gt_fids = gt_pq.on_surface(xx_points)
        gt_nvs = gt_mesh.face_normals[gt_fids]
        nae_deg1 = np.arccos((xx_normals * gt_nvs).sum(axis=-1))
        nae_deg1 = nae_deg1[~np.isnan(nae_deg1)]
        # print(d1.mean(), d1.max())
        # print(nae_deg1.mean()/np.pi*180, nae_deg1.max()/np.pi*180)

    d2 = None
    nae_deg2 = None
    if True:
        pq_mesh = trimesh.proximity.ProximityQuery(xx_mesh)
        print("finishing building pq_mesh")
        _, d2, xx_fids = pq_mesh.on_surface(gt_points)
        xx_nvs = xx_mesh.face_normals[xx_fids]
        nae_deg2 = np.arccos((xx_nvs * gt_normals).sum(axis=-1))
        nae_deg2 = nae_deg2[~np.isnan(nae_deg2)]
        # print(d2.mean(), d2.max())
        # print(nae_deg2.mean()/np.pi*180, nae_deg2.max()/np.pi*180)

    assert d1 is not None or d2 is not None

    if d1 is None:
        d1 = d2
        nae_deg1 = nae_deg2
    
    if d2 is None:
        d2 = d1
        nae_deg2 = nae_deg1

    metrics = {
        "details":{
            "chamfer_1": d1.mean(),
            "chamfer_2": d2.mean(),
            "HD_1": d1.max(),
            "HD_2": d2.max(),
            "nae_1": nae_deg1.mean()/np.pi*180,
            "nae_2": nae_deg2.mean()/np.pi*180,
        },
        "chamfer": (d1.mean() + d2.mean()) / 2,
        "HD": (d1.max() + d2.max()) / 2,
        "nae": (nae_deg1.mean()/np.pi*180 + nae_deg2.mean()/np.pi*180) / 2,
    }
    return metrics


def draw_error_map(gt_mesh, gt_pq, xx_mesh, sample_size=30000):
    xx_points, fids = trimesh.sample.sample_surface(xx_mesh, sample_size)
    _, d1, gt_fids = gt_pq.on_surface(xx_points)
    return xx_points, d1
    
def draw_error_to_mesh(proximity_query: trimesh.proximity.ProximityQuery, mesh):

    # norm = matplotlib.colors.Normalize(0.0, 0.01, clip=True)
    # mapper = cm.ScalarMappable(norm=norm, cmap=cm.jet)

    copymesh = mesh.copy()
    points = mesh.vertices
    _, d, _ = proximity_query.on_surface(points)
    print(d.mean(), d.max())

    clip_threshold = 0.003

    d = np.clip(d, 0.0, clip_threshold) / clip_threshold
    u = 1 - d
    v = np.zeros_like(d)
    # v = d
    uv = np.stack([u, v], axis=-1)

    from PIL import Image
    texture_img = Image.open(f'/home/lyang/yl_code/ngf/my_tools/asset/jet.png')
    copymesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=None, image=texture_img)
    
    # ## draw copymesh with texture
    # colors = [(r, g, b) for r, g, b, a in mapper.to_rgba(d)]
    # copymesh.visual.vertex_colors = colors*255
    return copymesh

def parse_args():
    parser = argparse.ArgumentParser(description="Modeling 3D shapes with neural patches")
    parser.add_argument("--model_name",
                        required=True,
                        type=str,
                        help="path to config"
                        )
    
    args = parser.parse_args()
    return args


# ## ngf
# data_method = "ngf"
# names = [
#     "dragon-lod2500-f20",
#     "vase-lod2000-f20", 
#     "Sapphos_Head-lod2000-f20",
#     "horsehead68380_remesh-lod2000-f20",
#     "horse_li-lod2000-f20",
#     "hawk126660-lod2000-f20",
#     "bunny_li-lod2000-f20",
#     "buddha-lod2000-f20",
#     "einstein-lod2000-f20",
#     "lucy120628-lod2000-f20",
#     "metatron-lod2000-f20",
#     "Ganesha-lod2000-f20",
#     "armadillo-lod2000-f20",
#     "55280-lod2000-f20",
#     "bimba_fix5-lod2000-f20"
# ]
# root = '/home/lyang/yl_code/ngf/results/'

# data_method = "nss"
# names = [
#     "dragon",
#     "sapphos",
#     # "horse",
#     # "hawk",
#     # "bunny",
#     # "buddha",
#     "einstein",
#     "metatron",
#     "ganesha",
#     "armadillo",
#     # =======
#     # "lucy",
#     # "vase", 
#     # "horsehead",
#     # "bimba_fix5"
# ]
# root = "/home/lyang/Downloads/anshul/teaser/normalized_neural_spline_results"

data_method = "nss" 
root = "~/Downloads/anshul/teaser/cc_and_bspline_examples"

if __name__ == "__main__":

    metric_dict = {}
    os.makedirs(root, exist_ok=True)
    os.makedirs(os.path.join(root, "error"), exist_ok=True)
    print_keys = ['chamfer', 'HD', 'nae']    

    # for model_name in names:
    model_name = "bimba_bspline"

    print('==================================================')
    print("model_name", model_name)
    
    # ## compute metrics
    # if data_method == "ngf":
    #     gt_file = os.path.join(root, f"normalized_target/{model_name}.stl")
    # else:
    #     gt_file = os.path.join(root, f"{model_name}_gt_normalized.obj")
    gt_file = os.path.join(root, 'bimba_head_fix6.obj')
    gt_mesh = trimesh.load(gt_file, process=False, maintain_order=True)
    gt_pq = trimesh.proximity.ProximityQuery(gt_mesh)
    print(gt_file)

    # if data_method == "ngf":
    #     mesh_file = os.path.join(root, f"stl/{model_name}.stl")
    # else:
    #     mesh_file = os.path.join(root, f"{model_name}_reconstructed_normalized.obj")
    # print(mesh_file)

    mesh_file = os.path.join(root, "bimba_head_bspline.obj")
    
    xx_mesh = trimesh.load(mesh_file, process=False, maintain_order=True)
    mesh_pq = trimesh.proximity.ProximityQuery(xx_mesh)
    
    metrics_to_gt = compute_metrics(gt_mesh, gt_pq, xx_mesh, sample_size=50000)
    metrics_to_xx = compute_metrics(xx_mesh, mesh_pq, gt_mesh, sample_size=50000)
    
    metric_dict[model_name] = {}
    for k in print_keys:
        metric_dict[model_name][k] = {}
        metric_dict[model_name][k]['gt_to_xx'] = metrics_to_gt[k]
        metric_dict[model_name][k]['xx_to_gt'] = metrics_to_xx[k]
        metric_dict[model_name][k]['avg'] = (metrics_to_gt[k] + metrics_to_xx[k])/2.0

    for k in print_keys:
        print(k, metric_dict[model_name][k]['avg'])
    write_json(metric_dict[model_name], os.path.join(root, f"metric_{model_name}.json"))
    
    # ## draw error map
    # # exit()
    # meshfile_errormap = f"{model_name}_errormap.obj"
    # meshfile_errormap = os.path.join(root, f"error/{meshfile_errormap}")
    # draw_error_to_mesh(gt_pq, xx_mesh).export(meshfile_errormap)
        


    write_json(metric_dict, os.path.join(root, f"metric.json"))
