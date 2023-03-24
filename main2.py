import cv2
import numpy as np
import trimesh
import open3d as o3d

# Step 1: Image Preprocessing
def preprocess_image(image):
    # Perform edge detection using the Canny edge detection algorithm
    edges = cv2.Canny(image, 100, 200)

    # Extract keypoints and descriptors using the ORB feature detector
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(edges, None)

    # Return the edges, keypoints, and descriptors
    return edges, keypoints, descriptors

# Step 2: 3D Mesh Generation 
def generate_3d_mesh(edges, features):
    # Convert the features array to the correct data type and reshape it
    features = features.astype(np.float64).reshape(-1, 3)

    # Create a point cloud from the features
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(features)

    # Perform surface reconstruction using Poisson surface reconstruction
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd)

    # Simplify the mesh using the Quadric Edge Collapse Decimation algorithm
    mesh.simplify_quadric_decimation(1000)

    # Smooth the mesh using the Taubin smoothing algorithm
    mesh.filter_smooth_taubin(30)

    # Return the mesh object
    return mesh
 
# Step 3: Texture Mapping
def texture_mapping(mesh, front_image, side_image, back_image):
    # Load the images
    front = cv2.imread(front_image)
    side = cv2.imread(side_image)
    back = cv2.imread(back_image)

    # Create a texture image by concatenating the input images
    texture_image = np.concatenate((front, side, back), axis=1)

    # Create a material object from the texture image
    material = o3d.geometry.Image(texture_image)

    # Assign the material to the mesh
    mesh.paint_uniform_color([1, 1, 1])
    mesh.texture = o3d.geometry.TriangleMesh.create_from_color_and_pcd(material, mesh)

    # Return the textured mesh
    return mesh

# Main Function
def main():

    # Load input images
    front_image = cv2.imread('front_ca.png')
    side_image = cv2.imread('side_ca.png')
    back_image = cv2.imread('back_ca.png')

    # Preprocess the input images and extract features
    front_edges, front_keypoints, front_descriptors = preprocess_image(front_image)
    side_edges, side_keypoints, side_descriptors = preprocess_image(side_image)
    back_edges, back_keypoints, back_descriptors = preprocess_image(back_image)

    # Extract feature locations from the keypoints
    front_locations = np.array([kp.pt for kp in front_keypoints])
    side_locations = np.array([kp.pt for kp in side_keypoints])
    back_locations = np.array([kp.pt for kp in back_keypoints])

    # Resize front and back edges arrays to match the shape of the side edges array
    front_edges_resized = cv2.resize(front_edges, (side_edges.shape[1], front_edges.shape[0]))
    back_edges_resized = cv2.resize(back_edges, (side_edges.shape[1], back_edges.shape[0]))

    # Concatenate edges and feature locations from multiple views
    edges = np.concatenate((front_edges_resized, side_edges, back_edges_resized), axis=0)
    features = np.concatenate((front_locations, side_locations, back_locations), axis=0)

    # Generate a 3D mesh from the input edges and features
    mesh = generate_3d_mesh(front_locations, side_locations, back_locations)

    # Texture map the mesh using the input images
    texture_mapped_mesh = texture_mapping(mesh, front_image, side_image, back_image)

    # Save the mesh to a file
    o3d.io.write_triangle_mesh('output_mesh.ply', texture_mapped_mesh)

    # Visualize the mesh
    o3d.visualization.draw_geometries([texture_mapped_mesh])


if __name__ == '__main__':
    main()
