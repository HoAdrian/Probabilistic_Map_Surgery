#!/bin/bash


####################################################################
# this file contains an example list of commands that you can run #
###################################################################


##############################################################################################
############## collect meshes and urdf for train & validation data of dense predictor ########
##############################################################################################

#conda activate py39
cd deformable_utils
python3 create_meshes.py --object_mesh_path="/home/dvrk/active_data/mix/cutting_mesh"
python3 create_urdfs.py --object_mesh_path="/home/dvrk/active_data/mix/cutting_mesh" --object_urdf_path="/home/dvrk/active_data/mix/cutting_urdf_train" --rand_seed=2023 --num_config=10000 --max_num_balls=1
python3 create_urdfs.py --object_mesh_path="/home/dvrk/active_data/mix/cutting_mesh" --object_urdf_path="/home/dvrk/active_data/mix/cutting_urdf_test" --rand_seed=1997 --num_config=1000 --max_num_balls=1
cd ..

# conda deactivate
# conda activate rlgpu
cd data_collection
python3 loop_collect_rigid_state.py --data_recording_path="/home/dvrk/active_data/mix/rigid_state_train" --object_urdf_path="/home/dvrk/active_data/mix/cutting_urdf_train" --object_mesh_path="/home/dvrk/active_data/mix/cutting_mesh" --headless=True --save_data True --rand_seed=2023
python3 loop_collect_rigid_state.py --data_recording_path="/home/dvrk/active_data/mix/rigid_state_test" --object_urdf_path="/home/dvrk/active_data/mix/cutting_urdf_test" --object_mesh_path="/home/dvrk/active_data/mix/cutting_mesh" --headless=True --save_data True --rand_seed=1997

python3 loop_collect_pcd.py --data_recording_path="/home/dvrk/active_data/mix/processed_data_train" --rigid_state_path="/home/dvrk/active_data/mix/rigid_state_train" --headless=True --save_data True --rand_seed=2023
python3 loop_collect_pcd.py --data_recording_path="/home/dvrk/active_data/mix/processed_data_test" --rigid_state_path="/home/dvrk/active_data/mix/rigid_state_test" --headless=True --save_data True --rand_seed=1997
cd ..

cd processed_data
python3 process_dense_predictor_data.py --data_recording_path="/home/dvrk/active_data/mix/processed_data_train" --data_processed_path="/home/dvrk/active_data/mix/processed_data_train"
python3 process_dense_predictor_data.py --data_recording_path="/home/dvrk/active_data/mix/processed_data_test" --data_processed_path="/home/dvrk/active_data/mix/processed_data_test"
cd ..


cd dense_predictor_data

python3 dense_predictor_trainer.py --weight_path="/home/dvrk/active_data/mix/dense_pred_weights" --tdp="/home/dvrk/active_data/mix/processed_data_train" --vdp="/home/dvrk/active_data/mix/processed_data_test" --epochs=161 --batch_size=40 --plot_category="dense_predictor"
python3 test_dense.py --weight_path="/home/dvrk/active_data/mix/dense_pred_weights/epoch_20" --vdp="/home/dvrk/active_data/mix/processed_data_train"
python3 test_dense.py --weight_path="/home/dvrk/active_data/mix/dense_pred_weights/epoch_60" --vdp="/home/dvrk/active_data/mix/processed_data_train"
cd ..


########################################################################################
############################## Online Learning #########################################
########################################################################################
cd online_learning
python3 loop_online_learning.py --data_recording_path="/home/dvrk/active_data/mix/online_learning" --object_urdf_path="/home/dvrk/active_data/mix/cutting_urdf_train" --object_mesh_path="/home/dvrk/active_data/mix/cutting_mesh" --headless=False --save_data False --rand_seed=2023 --weight_path="/home/dvrk/active_data/mix/dense_pred_weights/epoch_60"
python3 loop_online_learning.py --data_recording_path="/home/dvrk/active_data/mix/online_learning" --object_urdf_path="/home/dvrk/active_data/mix/cutting_urdf_test" --object_mesh_path="/home/dvrk/active_data/mix/cutting_mesh" --headless=False --save_data False --rand_seed=2023 --weight_path="/home/dvrk/active_data/mix/dense_pred_weights/epoch_60"
cd ..