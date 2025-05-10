import os
import numpy as np
import open3d as o3d

cwd = os.getcwd()
# folder = "vis_tools/data"
# file_name = "output.pcd"

# file_name = "cloud.pcd"
file_name = "05_horizontal_wing_side.pcd"
folder = "data/"

pcd_path = os.path.join(cwd, folder, file_name)
if not os.path.exists(pcd_path):
    raise FileNotFoundError(f"File '{pcd_path}' not found!")

class Visualizer:
    def __init__(self):
        self.point_cloud = None
        self.size = 0


    def set_pcd(self, pcd_path):
        pcd = o3d.io.read_point_cloud(pcd_path)
        if pcd.is_empty():
            raise ValueError("Loaded point cloud is empty!")
        self.point_cloud = pcd
        self.get_size()


    def get_size(self):
        if self.point_cloud is None:
            raise ValueError("Point cloud is not set!")
        self.size = len(self.point_cloud.points)  # Get actual point count
        return self.size


    def visualize(self, point_normals = False):
        if self.point_cloud is None:
            raise ValueError("Point cloud is not set!")
        o3d.visualization.draw_geometries([self.point_cloud], point_show_normal=point_normals)
    

    def downsample(self, leaf_size):
        # cloud_ds = o3d.geometry.voxel_down_sample(self.point_cloud, leaf_size)
        cloud_ds = self.point_cloud.voxel_down_sample(leaf_size)
        return cloud_ds



pcd_vis = Visualizer()
pcd_vis.set_pcd(pcd_path)
print("Cloud Size: ", pcd_vis.size)
pcd_vis.visualize()


# Downsampling 
save_flag = False
save_path = os.path.join(cwd, "data")
file_name = "/dsc.pcd"
dsc = pcd_vis.downsample(0.5)

if save_flag == True:
    o3d.io.write_point_cloud(save_path + file_name, dsc, write_ascii=True)