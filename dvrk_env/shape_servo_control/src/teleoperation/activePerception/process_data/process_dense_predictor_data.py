import sys
# import roslib.packages as rp
# pkg_path = rp.get_pkg_dir('shape_servo_control')
# sys.path.append(pkg_path + '/src')

import os
import numpy as np
import pickle
import open3d
import argparse

#from util.isaac_utils import *
sys.path.append("../../pc_utils")
from compute_partial_pc import farthest_point_sample_batched

'''
downsample each partial pointcloud to 256 points
'''

def process_and_save(data, data_processed_path, index):
    # partial_pc = data["partial_pc"]
    # pcd = np.expand_dims(partial_pc, axis=0) # shape (1, n, d)
    # down_sampled_pcd = farthest_point_sample_batched(point=pcd, npoint=256)
    # down_sampled_pcd = np.squeeze(down_sampled_pcd, axis=0)
    # data["partial_pc"] = down_sampled_pcd

    balls_xyz = data["balls_xyz"] #(num_attachment_points, 3)
    soft_xyz = data["soft_xyz"]

    xy_lower_bound = soft_xyz[:2] - 0.1
    xy_upper_bound = soft_xyz[:2] + 0.1

    n_query = 1024 # number of query points
    attachment_radius = 0.01
    num_positive_points = 50
    vis = False

    ######## sample random query points on the floor ##################
    query_points = np.random.uniform(low=xy_lower_bound, high=xy_upper_bound, size=(n_query,2)) #(n_query, 2)

    # which random query points belong to attachment points
    balls_xy = balls_xyz[:,:2][np.newaxis,:] #(1, num_attachment_points, 2)
    query_points_expand = query_points[:,np.newaxis, :] #(n_query, 1, 2)

    dists = np.linalg.norm(balls_xy - query_points_expand, axis=-1) #(n_query, num_attachment_points)
    dists_min = np.min(dists, axis=1)
    positive_pt_idxs = np.argpartition(dists_min, num_positive_points)[:num_positive_points]  # select num_positive_points closest points to the attachment point(s) 
    is_attachment_pt = np.zeros((dists_min.shape[0])).astype(int)
    is_attachment_pt[positive_pt_idxs] = 1
    # is_attachment_pt = (np.sum(dists<=attachment_radius, axis=-1)>=1).astype(int)

    data["random_query_points"] = np.pad(query_points, ((0,0), (0,1))) #(n_query, 3), set z-coordinate to 0
    data["random_query_points_labels"] = is_attachment_pt #(n_query,)

    if vis:
        target=is_attachment_pt
        num_pts = n_query #query_points.shape[0]
        target_expanded = target[:,np.newaxis]
        red = np.array([1,0,0])
        green = np.array([0,1,0])
        red = np.tile(red, (num_pts,1))
        green = np.tile(green, (num_pts,1))
        colors = np.where(target_expanded==1, red, green)
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(np.pad(query_points, ((0,0), (0,1))))
        pcd.colors = open3d.utility.Vector3dVector(colors)
        open3d.visualization.draw_geometries([pcd]) 

    ######## grid query points on the floor ##################
    x_ax = np.linspace(start=xy_lower_bound[0], stop=xy_upper_bound[0], num=(int)(np.sqrt(n_query)))
    y_ax = np.linspace(start=xy_lower_bound[1], stop=xy_upper_bound[1], num=(int)(np.sqrt(n_query)))
    x_grid, y_grid = np.meshgrid(x_ax, y_ax) #(n_query.sqrt,n_query.sqrt)
    grid_query_points = np.concatenate((x_grid[:,:,np.newaxis], y_grid[:,:,np.newaxis]), axis=-1) #(n_query.sqrt,n_query.sqrt,2)
    grid_query_points = grid_query_points.reshape((-1,2))
    
    # which grid query points belong to attachment points
    balls_xy = balls_xyz[:,:2][np.newaxis,:] #(1, num_attachment_points, 2)
    grid_query_points_expand = grid_query_points[:,np.newaxis, :] #(n_query, 1, 2)

    dists = np.linalg.norm(balls_xy - grid_query_points_expand, axis=-1) #(n_query, num_attachment_points)
    dists_min = np.min(dists, axis=1)
    positive_pt_idxs = np.argpartition(dists_min, num_positive_points)[:num_positive_points]  # select num_positive_points closest points to the attachment point(s) 
    is_attachment_pt = np.zeros((dists_min.shape[0])).astype(int)
    is_attachment_pt[positive_pt_idxs] = 1
    # is_attachment_pt = (np.sum(dists<=attachment_radius, axis=-1)>=1).astype(int)

    data["grid_query_points"] = np.pad(grid_query_points, ((0,0), (0,1))) #(n_query, 3), set z-coordinate to 0
    data["grid_query_points_labels"] = is_attachment_pt #(n_query,)

    if vis:
        target=is_attachment_pt
        num_pts = n_query #grid_query_points.shape[0]
        target_expanded = target[:,np.newaxis]
        red = np.array([1,0,0])
        green = np.array([0,1,0])
        red = np.tile(red, (num_pts,1))
        green = np.tile(green, (num_pts,1))
        colors = np.where(target_expanded==1, red, green)
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(np.pad(grid_query_points, ((0,0), (0,1))))
        pcd.colors = open3d.utility.Vector3dVector(colors)
        open3d.visualization.draw_geometries([pcd]) 

    ################# points close to the attachment point on the pointcloud ##############################
    partial_pc = data["partial_pc"]
    dists_partial_pc = np.linalg.norm(partial_pc[:,np.newaxis,:] - balls_xyz[:,:][np.newaxis,:], axis=-1) #(n_query, num_attachment_points)
    dists_min = np.min(dists_partial_pc, axis=1)
    positive_pt_idxs = np.argpartition(dists_min, num_positive_points)[:num_positive_points]  # select num_positive_points closest points to the attachment point(s) 
    is_attachment_pt = np.zeros((dists_min.shape[0])).astype(int)
    is_attachment_pt[positive_pt_idxs] = 1
    #is_attachment_pt = (np.sum(dists_partial_pc<=attachment_radius*4, axis=-1)>=1).astype(int) #6

    data["partial_pc_labels"] = is_attachment_pt #(n_query,)

    if vis:
        target=is_attachment_pt
        num_pts = partial_pc.shape[0]
        target_expanded = target[:,np.newaxis]
        red = np.array([1,0,0])
        green = np.array([0,1,0])
        red = np.tile(red, (num_pts,1))
        green = np.tile(green, (num_pts,1))
        colors = np.where(target_expanded==1, red, green)
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(np.array(partial_pc))
        pcd.colors = open3d.utility.Vector3dVector(colors)
        open3d.visualization.draw_geometries([pcd]) 



    with open(os.path.join(data_processed_path, "processed sample " + str(index) + ".pickle"), 'wb') as handle:
        pickle.dump(data, handle, protocol=3)   


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Options')
    is_train = True
    suffix = "train" if is_train else "test"
    parser.add_argument('--data_recording_path', type=str, help="where you recorded the data")
    parser.add_argument('--data_processed_path', type=str, help="where you want to record the processed data")
    parser.add_argument('--vis', default="False", type=str, help="if False: visualize processed data instead of saving processed data")

    args = parser.parse_args()

    data_recording_path = args.data_recording_path
    data_processed_path = args.data_processed_path
    vis = args.vis=="True"
    os.makedirs(data_processed_path, exist_ok=True)
    

   
    if not vis:
    # ############################### process and save ######################################
        num_samples = len(os.listdir(data_recording_path))
        print("num_groups to process: ", num_samples)
        
        processed_sample_idx = 0
        for sample_idx in range(num_samples):
            with open(os.path.join(data_recording_path, f"processed sample {sample_idx}.pickle"), 'rb') as handle:
                data = pickle.load(handle)
            #partial_pc = data["partial_pc"]
            process_and_save(data, data_processed_path, processed_sample_idx)
            print(f"processed_sample_idx: {processed_sample_idx}")
            processed_sample_idx += 1
            if sample_idx%1000==0:
                print(f"finished processing group[{sample_idx}]")
    else:
        ############################### visualize part of the processed point clouds ############################
        for i in range(100):
            with open(os.path.join(data_processed_path, "processed sample " + str(i) + ".pickle"), 'rb') as handle:
                print(os.path.join(data_processed_path, "processed sample " + str(i) + ".pickle"))
                data = pickle.load(handle)   
            print(f"sample {i}\tshape:", data["partial_pc"].shape) 
            pcd = data["partial_pc"]
            pcd2 = open3d.geometry.PointCloud()
            pcd2.points = open3d.utility.Vector3dVector(np.array(pcd)) 
            pcd2.paint_uniform_color([1, 0, 0])
            open3d.visualization.draw_geometries([pcd2])  


































##################################old version#################################################
# import sys
# import roslib.packages as rp
# pkg_path = rp.get_pkg_dir('shape_servo_control')
# sys.path.append(pkg_path + '/src')

# import os
# import math
# import numpy as np
# import pickle
# import timeit
# #import open3d
# import argparse

# from util.isaac_utils import *
# import random

# def process_and_save(data, data_processed_path, index):
#     pcd = data["partial_pc"]
#     pcd = down_sampling(pcd, num_pts=256)
#     data["partial_pc"] = pcd
     
#     with open(os.path.join(data_processed_path, "processed sample " + str(index) + ".pickle"), 'wb') as handle:
#         pickle.dump(data, handle, protocol=3)   
#     assert(pcd.shape[0]==256)


# if __name__ == "__main__":
#     ### CHANGE ####
#     is_train = True
#     suffix = "train" if is_train else "test"
#     data_processed_path = f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_AE_balls_corrected/processed_data_{suffix}_straight3D_partial_flat_2ball_varied"
#     os.makedirs(data_processed_path, exist_ok=True)
#     data_recording_path = f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_AE_balls_corrected/demos_{suffix}_straight3D_partial_flat_2ball_varied"
    
#     num_data_pt = len(os.listdir(data_recording_path))
#     print("num_data_pt", num_data_pt)
#     #assert(num_data_pt==50000)

#     # ############################### process and save ######################################
#     for i in range(0, num_data_pt, 1):
#         #print("gh")
#         with open(os.path.join(data_recording_path, f"sample {i}.pickle"), 'rb') as handle:
#             data = pickle.load(handle)
#         process_and_save(data, data_processed_path, i)
#         if i%1000==0:
#             print(f"finished processing samples[{i}]")

#     assert(len(os.listdir(data_processed_path))==len(os.listdir(data_recording_path)))

#     ############################### visualize processed point clouds ############################
#     # for i in range(100):
#     #     with open(os.path.join(data_processed_path, "processed sample " + str(i) + ".pickle"), 'rb') as handle:
#     #         print(os.path.join(data_processed_path, "processed sample " + str(i) + ".pickle"))
#     #         data = pickle.load(handle)   
#     #     print(f"sample {i}\tshape:", data["partial_pc"].shape) 
#     #     print(f"num_balls: ", len(data["balls_xyz"])) 
#     #     pcd = data["partial_pc"]
#     #     pcd2 = open3d.geometry.PointCloud()
#     #     pcd2.points = open3d.utility.Vector3dVector(np.array(pcd)) 
#     #     pcd2.paint_uniform_color([1, 0, 0])
#     #     open3d.visualization.draw_geometries([pcd2])  