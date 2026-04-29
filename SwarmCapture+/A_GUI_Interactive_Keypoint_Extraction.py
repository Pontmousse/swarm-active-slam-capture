import open3d as o3d
import numpy as np

# Target selection

# object_name = 'Turksat'
# ply_scale = 0.001

# object_name = 'Orion_Capsule'
# ply_scale = 0.0025

object_name = 'Motor'
ply_scale = 0.030

# object_name = 'Separation_Stage'
# ply_scale = 0.15


#%% Set up target pcd

ply_path = 'Targets/'+object_name+'/'+object_name+'.PLY'
mesh = o3d.io.read_triangle_mesh(ply_path)
mesh.compute_vertex_normals()
mesh.scale(ply_scale, center=mesh.get_center())

search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30)
number_of_points = 25000

target_pcd = mesh.sample_points_uniformly(number_of_points)
target_pcd.estimate_normals(search_param)
target_pcd.paint_uniform_color([0, 0, 0])  # Black color for keypoints

target_pcd.translate([0,0,0], relative=False)

#Debug
print('target point cloud x,y,z ranges min-max')
print(np.min(np.asarray(target_pcd.points), axis=0))
print(np.max(np.asarray(target_pcd.points), axis=0))

#%% GUI Interactive point selection
def pick_points(pcd):
    print("")
    print("1) Press [shift + left click] on a point to select it")
    print("2) Press [shift + right click] on the points to unselect them if desired")
    print("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()

picked_points_indices = pick_points(target_pcd)

print(f'You picked a total of {len(picked_points_indices)} points')
picked_points_coordinates = np.asarray(target_pcd.points)[picked_points_indices]


#%% Create sphere geometries for picked points to display

keypoints = o3d.geometry.PointCloud()
keypoints.points = o3d.utility.Vector3dVector(picked_points_coordinates)
keypoints.paint_uniform_color([1, 0, 0])  # Red color for keypoints


# Create spheres for keypoints
keypoint_spheres = []
for point in keypoints.points:
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    sphere.translate(point)
    sphere.paint_uniform_color([1, 0, 0])  # Red color for keypoints
    keypoint_spheres.append(sphere)
    
print('Created keypoint spheres')

#%% Display the generated points for vizualisation again

# Visualize point cloud and spheres
o3d.visualization.draw_geometries([target_pcd] + keypoint_spheres)


#%% Export the point coordinates in text file
output_file = 'Targets/'+object_name+'/'+object_name+'_keypoints_1.txt'

# Define custom comments
comments = [
    "# This file contains keypoint coordinates for the selected object.",
    f"# Object name: {object_name}",
    f"# PLY scale factor: {ply_scale}",
    "# Each line represents a keypoint in the format: x, y, z",
    ""
]

# Join comments with newline characters
comments_text = "\n".join(comments)

# Export the coordinates to a text file
np.savetxt(output_file, picked_points_coordinates, fmt='%.6f', delimiter=',',
           header=f"{comments_text}\n# x, y, z", comments='')

print(f'Exported keypoint coordinates with comments to {output_file}')













