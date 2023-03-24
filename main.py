import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from PIL import Image
import torch
import numpy as np
import open3d as o3d
import cv2
from transformers import GLPNImageProcessor, GLPNForDepthEstimation

feature_extractor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")

# load and resize the input image
image = Image.open("image.jpg")
new_height = 480 if image.height > 480 else image.height
new_height -= (new_height % 32)
new_width = int(new_height * image.width / image.height)
diff = new_width % 32
new_width = new_width - diff if diff < 16 else new_width + 32 - diff
new_size = (new_width, new_height)
image = image.resize(new_size)

width, height = new_size

# prepare image for the model
inputs = feature_extractor(images=image, return_tensors="pt")

# get the prediction from the model
with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

# remove borders
pad = 16
output = predicted_depth.squeeze().cpu().numpy() * 1000.0
output = output[pad:-pad, pad:-pad]
image = image.crop((pad, pad, image.width - pad, image.height - pad))

# apply bilateral filter to the depth image
output = output.astype(np.float32)
output = cv2.bilateralFilter(output, d=15, sigmaColor=15, sigmaSpace=0.1) 

# create rgbd image
depth_o3d = o3d.geometry.Image(output.astype(np.uint16))
image_o3d = o3d.geometry.Image(np.array(image))
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d, depth_o3d, depth_scale=1000.0, convert_rgb_to_intensity=False)

# camera settings
camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
camera_intrinsic.set_intrinsics(width, height, 500, 500, width/2, height/2)

# create point cloud
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)

# downsample point cloud
pcd = pcd.voxel_down_sample(voxel_size=0.01)

# radius outlier removal
cl, ind = pcd.remove_radius_outlier(nb_points=16, radius=0.05)
pcd = pcd.select_by_index(ind)

# estimate normals
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
pcd.orient_normals_to_align_with_direction()

# surface reconstruction
poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10, n_threads=1)[0]

# rotate the mesh
rotation_matrix = poisson_mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
poisson_mesh.rotate(rotation_matrix, center=(0, 0, 0))
 
#save the mesh
o3d.io.write_triangle_mesh(f'./mesh.obj', poisson_mesh)

#visualize the mesh
o3d.visualization.draw_geometries([poisson_mesh], mesh_show_back_face=True)
