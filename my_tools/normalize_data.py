import meshio
import os
import torch
import trimesh 
import math
def normalize_mesh(path, normalizer=None):
    mesh = meshio.read(path)

    v = torch.from_numpy(mesh.points[:, :3]).float().cuda()
    if 'triangle' in mesh.cells_dict:
        f = torch.from_numpy(mesh.cells_dict['triangle']).int().cuda()
    else:
        f = torch.from_numpy(mesh.cells_dict['quad']).int().cuda()

    if normalizer is None:
        vmin, vmax = v.min(dim=0)[0], v.max(dim=0)[0]
        center = (vmin + vmax) / 2
        extent = (vmax - vmin).max()
        normalizer = lambda x: (x - center) / (extent / 2)

    v = normalizer(v)

    save_path = os.path.join(os.path.dirname(path), 'normalized_' + os.path.basename(path))
    print(save_path)
    trimesh.Trimesh(vertices=v.cpu().numpy(), faces=f.cpu().numpy(), process=False, maintain_order=True).export(save_path)

if __name__ == '__main__':
    
    root_dir = "/home/lyang/Downloads/anshul/teaser/NeuralSplineResults"
    # root_dir = "/home/lyang/Downloads/anshul/teaser/curvature_compare"

    mesh_list = [
        # "armadillo_reconstructed_quad.obj",
        # "dragon_reconstructed_quad_2.obj",
        # "einstein_err.obj",
        # "ganesha_reconstructed_quad_2.obj",
        # "horse_reconstructed_quad.obj",
        # "metatron_reconstructed_quad.obj",
        # "sapphos_reconstructed_quad_500.obj",
        # "xyz_reconstructed.obj"
        "hawk_err_2.obj",
        "hawk_ngf_err.obj",
    ]

    for mesh_name in mesh_list:
        mesh_path = os.path.join(root_dir, mesh_name)
        # normalize_mesh(mesh_path)

        # mesh_path = os.path.join(root_dir, "normalized_ganesha_reconstructed_quad_2.obj")
        mesh = trimesh.load(mesh_path, process=False, maintain_order=True)
        ## rotate about x 180 degrees
        mesh.apply_transform(trimesh.transformations.rotation_matrix(math.pi/2, [1, 0, 0]))
        mesh.export(os.path.join(root_dir, f"normalized_{mesh_name}.obj"))