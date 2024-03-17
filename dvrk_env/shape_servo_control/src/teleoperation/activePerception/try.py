import numpy as np
import open3d

query_points = np.random.uniform(low=[1,2], high=[1.5,3], size=(10,2))
print(query_points)

n_query = 1024
balls_xyz = np.array([[0,-0.5,0]])
x_ax = np.linspace(start=-0.1, stop=0.1, num=(int)(np.sqrt(n_query)))
y_ax = np.linspace(start=-0.4, stop=-0.6, num=(int)(np.sqrt(n_query)))
x_grid, y_grid = np.meshgrid(x_ax, y_ax) #(n_query.sqrt,n_query.sqrt)
print(f"xgrid: {x_grid}")
print(f"ygrid: {y_grid}")
grid_query_points = np.concatenate((x_grid[:,:,np.newaxis], y_grid[:,:,np.newaxis]), axis=-1) #(n_query.sqrt,n_query.sqrt,2)
grid_query_points = grid_query_points.reshape((-1,2))
print(f"grid_points: {grid_query_points}")
# which grid query points belong to attachment points
balls_xy = balls_xyz[:,:2][np.newaxis,:] #(1, num_attachment_points, 2)
grid_query_points_expand = grid_query_points[:,np.newaxis, :] #(n_query, 1, 2)

dists = np.linalg.norm(balls_xy - grid_query_points_expand, axis=-1) #(n_query, num_attachment_points)
print(f"dist: {dists}")
is_attachment_pt = (np.sum(dists<=0.01, axis=-1)>=1).astype(int)


grid_points = np.pad(grid_query_points, ((0,0), (0,1))) #(n_query, 3), set z-coordinate to 0
# print(f"3D grid_points: {grid_points}")

grid_label = is_attachment_pt #(n_query,)
# print(f"grid_label: {grid_label}")

if True:
    grid_label_expanded = grid_label[:,np.newaxis]
    red = np.array([1,0,0])
    green = np.array([0,1,0])
    red = np.tile(red, (len(grid_label),1))
    green = np.tile(green, (len(grid_label),1))
    colors = np.where(grid_label_expanded==1, red, green)
    print(f"color? {grid_label_expanded}")
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np.array(grid_points))
    pcd.colors = open3d.utility.Vector3dVector(colors)

    # pcd_att = open3d.geometry.PointCloud()
    # pcd_att.points = open3d.utility.Vector3dVector(np.array(balls_xyz))
    # pcd_att.colors = open3d.utility.Vector3dVector([[0,0,1]])
    open3d.visualization.draw_geometries([pcd]) 