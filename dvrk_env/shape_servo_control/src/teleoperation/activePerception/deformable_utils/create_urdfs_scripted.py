import os
import pickle
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--object_mesh_path', type=str, help="where you get the mesh")
    parser.add_argument('--object_urdf_path', type=str, help="root path of the urdf, where you save the urdf")
    parser.add_argument('--rand_seed', default=2021, type=int, help="random seed")
    parser.add_argument('--num_config', default=1, type=int, help="number of configurations of attachment points (number of tissue urdf)")
    parser.add_argument('--max_num_balls', default=1, type=int, help="maximum number of attachment points for all configurations")
    
    args = parser.parse_args()

    object_mesh_path =args.object_mesh_path
    base_mesh_path = object_mesh_path
    object_urdf_path = args.object_urdf_path
    os.makedirs(object_urdf_path,exist_ok=True)

    np.random.seed(args.rand_seed)

    num_urdf = args.num_config
    max_num_balls = args.max_num_balls

    density = 100
    poissons = 0.3
    scale = 1.0
    attach_dist = 0.01

    with open(os.path.join(object_mesh_path, "primitive_dict.pickle"), 'rb') as handle:
        mesh_data = pickle.load(handle)
    base_thickness = mesh_data["base"]["thickness"]

    for i in range(num_urdf):
        mesh_object_name = f"box"
        base_name = f"base"
        height = mesh_data[mesh_object_name]["height"]
        width = mesh_data[mesh_object_name]["width"]
        thickness = mesh_data[mesh_object_name]["thickness"]
        youngs =  mesh_data[mesh_object_name]["youngs"]

        object_name = f"tissue_{i}"
    
        urdf_file_path = object_urdf_path + '/' + object_name + '.urdf'
        f = open(urdf_file_path, 'w')

        balls_xyz = []
        for b in range(max_num_balls):
      
            # big region
            if b==0:
                sample_x = -(width/4)
                sample_y = height/4
            else:
                sample_x = width/4
                sample_y = height/4

            z = base_thickness/2
            balls_xyz.append([sample_x, sample_y, z])
        
        print("balls_relative_xyz: ", balls_xyz)

        
        urdf_str = f"""<?xml version="1.0" encoding="utf-8"?>    
        
        <robot name="{object_name}">
            <link name="{object_name}">    
                <fem>
                    <origin rpy="0.0 0.0 0.0" xyz="0 0 0" />
                    <density value="{density}" />
                    <youngs value="{youngs}"/>
                    <poissons value="{poissons}"/>
                    <damping value="0.0" />
                    <attachDistance value="{attach_dist}"/>
                    <tetmesh filename="{os.path.join(object_mesh_path, mesh_object_name+".tet")}"/>
                    <scale value="{scale}"/>
                </fem>
            </link>
        """

        for b in range(max_num_balls):
            
            urdf_str +=f"""
                <link name="ball_{b}">
                    <visual>
                        <origin xyz="{balls_xyz[b][0]} {balls_xyz[b][1]} {-(thickness+base_thickness)*scale/2+0.0015:.5f}"/>              
                        <geometry>
                            <mesh filename="{os.path.join(base_mesh_path, base_name+".obj")}" scale="{scale} {scale} {scale}"/>
                        </geometry>
                    </visual>
                    <collision>
                        <origin xyz="{balls_xyz[b][0]} {balls_xyz[b][1]} {-(thickness+base_thickness)*scale/2+0.0015:.5f}"/>           
                        <geometry>
                            <mesh filename="{os.path.join(base_mesh_path, base_name+".obj")}" scale="{scale} {scale} {scale}"/>
                        </geometry>
                    </collision>
                    <inertial>
                        <mass value="5000000000000"/>
                        <inertia ixx="10.0" ixy="0.0" ixz="0.0" iyy="10.0" iyz="0.0" izz="10.0"/>
                    </inertial>
                </link>
                
                <joint name = "attach_{b}" type = "fixed">
                    <origin xyz = "{0} {0} 0.0" rpy = "0 0 0"/>
                    <parent link ="{object_name}"/>
                    <child link = "ball_{b}"/>
                </joint>  

            """

        urdf_str+="""
        </robot>
        """
        
        f.write(urdf_str)
        f.close()

        balls_data = {"balls_relative_xyz": balls_xyz}
        with open(os.path.join(object_urdf_path, f"{object_name}_balls_relative_xyz.pickle"), 'wb') as handle:
            pickle.dump(balls_data, handle, protocol=3) 
        

