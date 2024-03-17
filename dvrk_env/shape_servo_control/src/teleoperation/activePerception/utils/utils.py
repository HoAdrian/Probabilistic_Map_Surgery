import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import trimesh

'''
 Utils for probabilistic map over attachment points
'''



def get_point_label_nn(balls_xy, query_points, num_positive_points=50):
    '''
    balls_xy: (num_attachment_points, 2)
    query_points: (num_query, 2)
    num_positive_points: number of points that should be classified as attachment point

    returns the ground truth label of each point in the query points based on num_positive_points nearest neighbors
    '''
    balls_xy = balls_xy[:,:2][np.newaxis,:] #(1, num_attachment_points, 2)
    query_expand = query_points[:,np.newaxis, :] #(n_query, 1, 2)

    dists = np.linalg.norm(balls_xy - query_expand, axis=-1) #(n_query, num_attachment_points)
    
    dists_min = np.min(dists, axis=1)
    positive_pt_idxs = np.argpartition(dists_min, num_positive_points)[:num_positive_points]  # select num_positive_points closest points to the attachment point(s) 
    is_attachment_pt = np.zeros((dists_min.shape[0])).astype(int)
    is_attachment_pt[positive_pt_idxs] = 1

    return is_attachment_pt

def get_point_label_nn_for_each(balls_xy, query_points, num_positive_points_for_each=50):
    '''
    balls_xy: (num_attachment_points, 2)
    query_points: (num_query, 2)
    num_positive_points_for_each: number of points around each attachment point that should be classified as attachment point

    returns the ground truth label of each point in the query points based on num_positive_points_for_each nearest neighbors
    '''
    balls_xy = balls_xy[:,:2][np.newaxis,:] #(1, num_attachment_points, 2)
    query_expand = query_points[:,np.newaxis, :] #(n_query, 1, 2)

    dists = np.linalg.norm(balls_xy - query_expand, axis=-1) #(n_query, num_attachment_points)
    is_attachment_pt = np.zeros((dists.shape[0])).astype(int)
    
    for i in range(balls_xy.shape[1]):
        dists_one_attachment = dists[:,i]
        positive_pt_idxs = np.argpartition(dists_one_attachment, num_positive_points_for_each)[:num_positive_points_for_each]
        is_attachment_pt[positive_pt_idxs] = 1

    return is_attachment_pt

def get_point_label_radius(balls_xy, query_points, max_radius=0.01):
    '''
    balls_xy: (num_attachment_points, 2)
    query_points: (num_query, 2)
    max_radius: max distance a point of an attachment point can be from an attachment point

    returns the ground truth label of each point in the query points based on its distance from some attachment point
    '''
    balls_xy = balls_xy[:,:2][np.newaxis,:] #(1, num_attachment_points, 2)
    query_expand = query_points[:,np.newaxis, :] #(n_query, 1, 2)

    dists = np.linalg.norm(balls_xy - query_expand, axis=-1) #(n_query, num_attachment_points)
    is_attachment_pt = (np.sum(dists<=max_radius, axis=-1)>=1).astype(int)

    return is_attachment_pt

def plot_2D_points(points, values, vmin, vmax, title, path, name, has_vmin=True, has_vmax=True, vis=True):
    '''
    points: shape (num_poins,2)
    values: shape (num_poins,)
    vmin: min value of a point
    vmax: max value of a point
    # xlim: [lb, up] lower bound and upper bound on x value
    # ylim: [lb, up] lower bound and upper bound on y value
    path: where to save the image
    '''
    plt.figure(figsize=(7,7))
    if has_vmin and has_vmax:
        plt.scatter(points[:,0], points[:,1], c=values, cmap='jet', vmin=vmin, vmax=vmax)
    elif not has_vmin and has_vmax:
        plt.scatter(points[:,0], points[:,1], c=values, cmap='jet', vmax=vmax)
    elif has_vmin and not has_vmax:
        plt.scatter(points[:,0], points[:,1], c=values, cmap='jet', vmin=vmin)
    else:
        plt.scatter(points[:,0], points[:,1], c=values, cmap='jet')
    plt.colorbar()
    plt.title(title)
    #plt.xlim(xlim); plt.ylim(ylim)
    if path!=None and name!=None:
        os.makedirs(path, exist_ok=True)
        plt.savefig(f"{path}/{name}.png")
    if vis:
        plt.show()


def get_2D_grid_points(num_points, xy_lower_bound, xy_upper_bound):
    x_ax = np.linspace(start=xy_lower_bound[0], stop=xy_upper_bound[0], num=(int)(np.sqrt(num_points)))
    y_ax = np.linspace(start=xy_lower_bound[1], stop=xy_upper_bound[1], num=(int)(np.sqrt(num_points)))
    x_grid, y_grid = np.meshgrid(x_ax, y_ax)
    grid = np.concatenate((x_grid[:,:,np.newaxis], y_grid[:,:,np.newaxis]), axis=-1)
    grid_points = grid.reshape((-1,2)) # points in 2D
    return grid_points


def plot_xy(xs, ys_list, title, x_label, y_label):
    fig, ax = plt.subplots()
    for idx in range(len(ys_list)):
        ax.scatter(xs, ys_list[idx], s=8)
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.show()

def plot_xyz(xs, ys, zs, title):
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
    ax.scatter(xs, ys, zs, c=[0,0,1], s=8)
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(azim=45, elev=30)
    plt.show()

def compute_camera_intrinsics_matrix(image_width, image_heigth, horizontal_fov):
    '''
    compute this for Isaac Gym
    '''
    vertical_fov = (image_heigth / image_width * horizontal_fov) * np.pi / 180
    horizontal_fov *= np.pi / 180

    f_x = (image_width / 2.0) / np.tan(horizontal_fov / 2.0)
    f_y = (image_heigth / 2.0) / np.tan(vertical_fov / 2.0)

    K = np.array([[f_x, 0.0, image_width / 2.0], [0.0, f_y, image_heigth / 2.0], [0.0, 0.0, 1.0]]).astype(np.float64)

    return K

def compute_camera_extrinsics_matrix(cam_view_mat):
    '''
    Compute this for Isaac Gym
    '''
    return cam_view_mat.T.astype(np.float64)

def is_in_triangle(points_2D, triangle_2D):
    '''
    points_2D: (N,2)
    triangle_2D: (3,2) 
    '''
    A = triangle_2D[0,:][np.newaxis,:] #(1,2)
    B = triangle_2D[1,:][np.newaxis,:]
    C = triangle_2D[2,:][np.newaxis,:]

    points_2D -= C # shift coordinate frame to C
    v = A-C
    u = B-C
    basis = np.concatenate((v,u), axis=0) #(2,2)
    coeffs = np.linalg.solve(basis.T, points_2D.T).T #(N,2)
    return (coeffs[:,0]>=0) & (coeffs[:,1]>=0) & (np.sum(coeffs, axis=1)<=1)

def is_occluded(points, mesh_vertices, mesh_triangles, intrinsic_mat, homo_mat):
    '''
    points: (N,3), each row is x,y,z coordinates, we want to know whether a point is occluded by the mesh for each point. 
    mesh_vertices: (K, 3)
    mesh_triangles: (M, 3)
    intrinsic_mat: (3,3)
    homo_mat: (4,4), world to cam
    '''
    world2cam_R = homo_mat[:3, :3]
    world2cam_t = homo_mat[:3, 3:4]

    img_points,_ = cv2.projectPoints(points, world2cam_R,  world2cam_t, intrinsic_mat, np.zeros((5, 1), np.float64) )
    img_points = img_points[:,0,:]
    img_vertices,_ = cv2.projectPoints(mesh_vertices, world2cam_R,  world2cam_t, intrinsic_mat, np.zeros((5, 1), np.float64) )
    img_vertices = img_vertices[:,0,:]
    triangles_2D = img_vertices[mesh_triangles] #(num_triangles, 3, 2)

    is_occluded = np.zeros((points.shape[0]))!=0
    for triangle in triangles_2D:
        #print(triangle.shape)
        is_occluded = (is_in_triangle(img_points, triangle)) | (is_occluded)
    
    return is_occluded, img_points, img_vertices

# def is_occluded(points, mesh_vertices, mesh_triangles, intrinsic_mat, homo_mat):
#     '''
#     points: (N,3), each row is x,y,z coordinates, we want to know whether a point is occluded by the mesh for each point. 
#     mesh_vertices: (K, 3)
#     mesh_triangles: (M, 3)
#     intrinsic_mat: (3,3)
#     homo_mat: (4,4), world to cam
#     '''
#     img_points =  projectPoints(points, homo_mat, intrinsic_mat)
#     img_vertices =  projectPoints(mesh_vertices, homo_mat, intrinsic_mat)
#     triangles_2D = img_vertices[mesh_triangles] #(num_triangles, 3, 2)

#     is_occluded = np.zeros((points.shape[0]))!=0
#     for triangle in triangles_2D:
#         #print(triangle.shape)
#         is_occluded = (is_in_triangle(img_points, triangle)) | (is_occluded)
    
#     return is_occluded, img_points, img_vertices

# def projectPoints(points, homo_mat, intrinsic_mat):
#     points = points.T #(3,N)
#     points = np.concatenate((points, np.ones((1,points.shape[1]))), axis=0) #(4,N)
#     trans = homo_mat[:3, :4] #(3,4)
#     points_cam = np.matmul(trans, points) #(3,N)
#     points_cam_z = points_cam[2:3,:] #(1,N)
#     points_cam_z = np.where(points_cam_z==0, 1, points_cam_z)
#     points_img = (np.matmul(intrinsic_mat, points_cam)/(points_cam_z))[:2, :] #(2,N)
#     return points_img.T

def get_rays(points, ray_origin):
    '''
    points: shape (N,3)
    ray_origin: shape (3,)
    get rays from the ray_origin to each point in points
    '''
    ray_origin = np.tile(ray_origin.reshape(1,3), (len(points),1))
    ray_directions = points - ray_origin #(N,3)

    # stack rays into line segments for visualization as Path3D
    lines = np.hstack((ray_origin,ray_origin + ray_directions)).reshape(-1, 2, 3)
    ray_visualize = trimesh.load_path(lines)

    return ray_origin, ray_directions, ray_visualize


def compute_occluded_points(full_pc, tri_indices, points, ray_origin, vis = False):
    '''
    We compute the ray from the ray origin to each point in points, and then determine whether the ray intersects the triangular mesh specified
    by vertices=full_pc and faces=tri_indices

    full_pc: shape (num_vertices,3)
    tri_indices: shape (num_triangles,3)
    points: shape (N,3)
    ray_origin: shape (1,3)

    return indices of points that are occluded by the mesh
    '''
    ray_origin, ray_directions, ray_visualize = get_rays(points, ray_origin)

    tissue_mesh = trimesh.Trimesh(vertices=full_pc,
                            faces=np.array(tri_indices).reshape(-1,3).astype(np.int32))

    ### run the mesh- ray test
    locations, index_ray, index_tri = tissue_mesh.ray.intersects_location(
        ray_origins=ray_origin,
        ray_directions=ray_directions)

    index_ray = set(index_ray)
    index_ray = np.array(list(index_ray))

    return index_ray
    


if __name__=="__main__":
    ########## test grid points labels 
    # xy_lower_bound = np.array([-0.1, -0.5])
    # xy_upper_bound = np.array([0.1, -0.3])
    # n_grid_points = 1024
    # grid_points = get_2D_grid_points(n_grid_points, xy_lower_bound, xy_upper_bound)

    # balls_xy = np.array([[-0.05, -0.350], [0.0, -0.47]])

    # labels = get_point_label_nn_for_each(balls_xy, grid_points, num_positive_points_for_each=50)
    # print(np.sum(labels))

    # plot_2D_points(grid_points, labels, vmin=0, vmax=1, title="test label", path=None, name=None, has_vmin=True, has_vmax=True)

    ######### test is in triangle
    # vertices = np.array([[1,0], [0,1], [1,1]]) #(num_vertices, d)
    # triangles = np.array([[0,1,2]]) #(num_triangles, 3)
    # triangles_vertices = vertices[triangles] #(num_triangles, 3, d)
    # #print(vertices[triangles])
    # points_2D = np.array([[0.5, 0.6], [0.4, 0.6], [0.4, 0.3]])
    # print(is_in_triangle(points_2D, triangles_vertices[0]))

    # vertices = np.array([[1,0], [0,1], [0,0]]) #(num_vertices, d)
    # triangles = np.array([[0,1,2]]) #(num_triangles, 3)
    # triangles_vertices = vertices[triangles] #(num_triangles, 3, d)
    # #print(vertices[triangles])
    # points_2D = np.array([[0.5, 0.6], [0.4, 0.6], [0.4, 0.3]])
    # print(is_in_triangle(points_2D, triangles_vertices[0]))

    ####### test is occluded
    # homo_mat = np.array([[1,0,0,0], \
    #                     [0,1,0,0],  \
    #                     [0,0,1,0],  \
    #                     [0,0,0,1]   \
    #                     ]).astype(np.float64)
    # intrinsic_mat = np.array([ \
    #                     [1,0,0],  \
    #                     [0,1,1],  \
    #                     [0,0,1]   \
    #                     ]).astype(np.float64)
    # vertices = np.array([[1,0,0], [0,1,0], [-1,-1,0]]).astype(np.float64) #(num_vertices, d)
    # triangles = np.array([[0,1,2]]).astype(int) #(num_triangles, 3)

    # points = get_2D_grid_points(1024, [-1,-1], [1,1])
    # points = np.pad(points, ((0,0), (0,1)), constant_values=(0,0)).astype(np.float64) #(1024,3)
    
    
    # is_occluded_arr = is_occluded(points, vertices, triangles, intrinsic_mat, homo_mat)
    # labels = np.zeros((1024,))
    # labels = np.where(is_occluded_arr, 1, 0)
    # print(f"num occluded: ", np.sum(labels))
    # plot_2D_points(points, labels, vmin=0, vmax=1, title="occlude", path=None, name=None, has_vmin=True, has_vmax=True)

    #projectPoints(points, homo_mat, intrinsic_mat)

    ######## test get rays ######
    points = get_2D_grid_points(1024, [-1,-1], [1,1])
    points = np.pad(points, ((0,0), (0,1)), constant_values=(0,0)).astype(np.float64) #(1024,3)
    ray_origin = np.array([0,-0.0,0.1])
    get_rays(points, ray_origin)

    vertices = np.array([[1,0,0], [0,1,0], [-1,-1,0]]).astype(np.float64) #(num_vertices, d)
    triangles = np.array([[0,1,2]]).astype(int) #(num_triangles, 3)

    compute_occluded_points(vertices, triangles, points, ray_origin, vis = True)




















########### Unused code I spend so much time on that I don't want to throw away ################################
# cam_pos_np = np.array([cam_positions.x, cam_positions.y, cam_positions.z])
# cam_target_np = np.array([cam_targets.x, cam_targets.y, cam_targets.z])
# V = np.array(gym.get_camera_view_matrix(sim, envs[0], cam_handle))
# homo_mat = compute_camera_extrinsics_matrix(V)
# intrinsic_mat = compute_camera_intrinsics_matrix(cam_prop.width, cam_prop.height, cam_prop.horizontal_fov)
# homo_mat = homo_mat.astype(np.float64)

# intrinsic_mat = intrinsic_mat.astype(np.float64)
# print(f"homo: {homo_mat}")
# print(f"intrinsics: {intrinsic_mat}")

# vis_3D_mesh=False
# if vis_3D_mesh:
#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')
    
#     cam_x = homo_mat[:3,0]/10
#     cam_y = homo_mat[:3,1]/10
#     cam_z = homo_mat[:3,2]/10
#     #cam_pos_np = -homo_mat[:3,3]
#     ax.scatter(cam_pos_np[0], cam_pos_np[1], cam_pos_np[2])
#     ax.scatter(cam_pos_np[0]+cam_x[0], cam_pos_np[1]+cam_x[1], cam_pos_np[2]+cam_x[2], color="red", s=60)
#     ax.scatter(cam_pos_np[0]+cam_y[0], cam_pos_np[1]+cam_y[1], cam_pos_np[2]+cam_y[2], color="green", s=60)
#     ax.scatter(cam_pos_np[0]+cam_z[0], cam_pos_np[1]+cam_z[1], cam_pos_np[2]+cam_z[2], color="blue", s=60)
#     ax.plot_trisurf(trimesh_mesh.vertices[:, 0], trimesh_mesh.vertices[:,1], trimesh_mesh.vertices[:,2], triangles=trimesh_mesh.faces)

# homo_mat[:3,3] = -np.matmul(homo_mat[:3, :3].T,-cam_pos_np)
# homo_mat[:3, :3] = homo_mat[:3, :3].T




# is_occluded_arr, img_points, img_vertices = is_occluded(grid_3D, obj_particles, tri_indices, intrinsic_mat, homo_mat) #(1024,)
# print("num points occluded:", np.sum(is_occluded_arr.astype(int)))
# print(f"img points: {img_points}")
# print(f"img vertices: {img_vertices}")

# vis_2D_mesh=True
# if vis_2D_mesh:
#     fig = plt.figure()
#     ax = fig.add_subplot()
#     mesh = trimesh.Trimesh(vertices=img_vertices, faces=tri_indices)
#     #ax.scatter(img_points[:,0], img_points[:,1], color="red", s=60)
#     ax.triplot(mesh.vertices[:, 0], mesh.vertices[:,1], triangles=mesh.faces)