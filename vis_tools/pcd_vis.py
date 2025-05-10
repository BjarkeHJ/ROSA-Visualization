import os
import numpy as np
import open3d as o3d

from open3d.visualization.rendering import OffscreenRenderer, MaterialRecord
from open3d.geometry import AxisAlignedBoundingBox

def point_cloud_to_spheres(pcd, radius=0.02, resolution=10):
    spheres = []
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    for i, point in enumerate(pcd.points):
        if not np.all(np.isfinite(point)):
            continue  # Skip NaN/Inf

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
        sphere.translate(point)

        if len(sphere.vertices) == 0:
            continue  # Skip invalid geometry

        sphere.compute_vertex_normals()

        if colors is not None and i < len(colors):
            sphere.paint_uniform_color(colors[i])
        else:
            sphere.paint_uniform_color([0.5, 0.5, 0.5])

        spheres.append(sphere)
    return spheres


class Visualizer:
    def __init__(self):
        self.point_clouds = []
        self.point_clouds_w_size = []

    def add_cloud(self, pcd_path, color=None, point_size=3.0):
        pcd = o3d.io.read_point_cloud(pcd_path)
        if pcd.is_empty():
            raise ValueError(f"Loaded Point Cloud at {pcd_path} is empty!")

        if color is not None:
            color = np.array(color).reshape(1, 3)
            pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (len(pcd.points), 1)))
        else:
            pcd.paint_uniform_color(np.random.rand(3))

        self.point_clouds.append(pcd)
        self.point_clouds_w_size.append((pcd, point_size)) # For image rendering 

    def visualize(self, show_normals=False):
        if not self.point_clouds:
            raise ValueError("No point clouds to visualize.")
        o3d.visualization.draw_geometries(self.point_clouds, point_show_normal=show_normals)

    def clear(self):
        self.point_clouds = []
        self.point_clouds_w_size = []

    def render_image(self, out_path, width=1024, height=768, transparent=True, camera_view=None):
        if not self.point_clouds_w_size:
            raise ValueError("No point clouds to render.")

        renderer = OffscreenRenderer(width, height)
        bg_color = [0, 0, 0, 0] if transparent else [1, 1, 1, 1]
        renderer.scene.set_background(bg_color)
        renderer.scene.scene.set_sun_light([0, 1, -1], [1, 1, 1], 75000)
        renderer.scene.scene.enable_sun_light(True)

        bbox = AxisAlignedBoundingBox()
        for i, (pcd, point_size) in enumerate(self.point_clouds_w_size):

            spheres = point_cloud_to_spheres(pcd, radius=point_size * 0.1)
            for j, sphere in enumerate(spheres):
                name = f"sphere_{i}_{j}"
                mat = MaterialRecord()
                mat.shader = "defaultLit"
                renderer.scene.add_geometry(name, sphere, mat)

            # mat = MaterialRecord()
            # mat.shader = "defaultUnlit"
            # mat.point_size = point_size
            # name = f"cloud_{i}"
            # renderer.scene.add_geometry(name, pcd, mat)

            bbox += pcd.get_axis_aligned_bounding_box()

        # Set camera
        if camera_view is not None:
            center = np.array(camera_view["center"])
            eye = np.array(camera_view["eye"])
            up = np.array(camera_view["up"])
        else:
            center = bbox.get_center()
            extent = bbox.get_extent().max()
            eye = center + np.array([extent, 0, 0])
            up = np.array([0, 0, 1])

        renderer.scene.camera.look_at(center, eye, up)

        img = renderer.render_to_image()
        o3d.io.write_image(out_path, img)
        print(f"Screenshot saved to: {out_path}")


# Path settings:
cwd = os.getcwd()

# raw_cloud_foler = "data/"
# raw_cloud_fname = "windmill.pcd"
# raw_cloud_fname = "windmill.pcd"
# raw_cloud_fname = "05_horizontal_wing_side.pcd"
# raw_cloud_fname = "08_nacelle_side.pcd"
# raw_cloud_fname = "09_wings_only_front.pcd"
# raw_cloud_path = os.path.join(cwd, raw_cloud_foler, raw_cloud_fname)
# if not os.path.exists(raw_cloud_path):
#     raise FileNotFoundError(f"File '{raw_cloud_path}' not found!")

folder = "vis_tools/data/"

ds_cloud_name = "input_ds.pcd"
ds_cloud_path = os.path.join(cwd, folder, ds_cloud_name)
if not os.path.exists(ds_cloud_path):
    raise FileNotFoundError(f"File '{ds_cloud_path}' not found!")

rosa_fname = "output_04.pcd"
rosa_cloud_path = os.path.join(cwd, folder, rosa_fname)
if not os.path.exists(rosa_cloud_path):
    raise FileNotFoundError(f"File '{rosa_cloud_path}' not found!")

# Visualizer:
pcd_vis = Visualizer()
pcd_vis.add_cloud(ds_cloud_path, color=[0,1,1], point_size=0.03)
pcd_vis.add_cloud(rosa_cloud_path, color=[1,0,0], point_size=0.1)
pcd_vis.visualize()


# Save Visualizations
image_folder = "vis_tools/images"
image_fname = "output_img.png"
out_img_path = os.path.join(cwd, image_folder, image_fname)

save_flag = True

# camera_view = {
#     "center": [0.5,0.5,0], #Direction of view
#     "eye": [0.55,0.55,0], #Position of view
#     "up": [0,0,1]
# }

camera_view = {
    "center": [-0.5, 0.5, -0.1], #Direction of view
    "eye": [0.6, -0.8, 1.0], #Position of view
    "up": [0,0,1]
}


if save_flag == True:
    pcd_vis.render_image(out_img_path, width=3840, height=2160, transparent=True, camera_view=camera_view)