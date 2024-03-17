
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import argparse
import open3d

from architecture import DensePredictor
from dataset_loader import DensePredictorDataset
import pickle


def test(model, device, test_loader, epoch):
    model.eval()
    size = len(test_loader.dataset)
    num_batch = len(test_loader)
    test_loss = 0
    correct = 0
    false_negative = 0
    false_positive = 0
    pos_size = 0
    neg_size = 0
    with torch.no_grad():
        for sample in test_loader:
           
            pc = sample[0].to(device)
            query = sample[1].to(device)
            target = sample[2].to(device)
            
            output = model(pc, query)

            loss = F.nll_loss(output, target)
            test_loss += loss.item()
            pred = torch.argmax(torch.exp(output), dim=1)
            correct += (pred==target).type(torch.float).sum().item() /  pred.shape[-1]

            false_negative += torch.logical_and(pred==0, target==1).type(torch.float).sum().item() #/  torch.sum(target==1)
            false_positive += torch.logical_and(pred==1, target==0).type(torch.float).sum().item() #/  torch.sum(target==0)
            pos_size += torch.sum(target==1).item()
            neg_size += torch.sum(target==0).item()
    
    avg_test_loss = test_loss/num_batch
    correct /= size
    false_positive /= neg_size
    false_negative /= pos_size
    print(f"Test: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} False positive:{false_positive} False negative:{false_negative}\n")
    print("============================")


    return avg_test_loss, correct, false_positive, false_negative



if __name__ == "__main__":
    torch.manual_seed(2021)
    
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--weight_path', type=str, help="where to load the weight")
    parser.add_argument('--vdp', type=str, help="validation data path")

    args = parser.parse_args()

    weight_path = args.weight_path
    vdp = args.vdp

    device = torch.device("cuda")

    test_dataset = DensePredictorDataset(dataset_path=vdp)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=40, shuffle=True)

    model = DensePredictor(num_classes=2).to(device)  # single robot
    model.load_state_dict(torch.load(weight_path))
    #test_loss, test_acc, fp, fn = test(model, device, test_loader, None)

    #10
    with open(os.path.join(vdp, f"processed sample {0}.pickle"), 'rb') as handle:
            data = pickle.load(handle)

    partial_pc = data["partial_pc"] #(num_pts,3)
    target = data["grid_query_points_labels"] #(num_pts,)
    query_points = data["grid_query_points"]# shape (num_pts, 3)
    #print(f"target: {target}")
    
    num_pts = len(partial_pc)
    partial_pc_torch = torch.tensor(partial_pc).to(device).float().unsqueeze(0).permute((0,2,1))
    query_torch = torch.tensor(query_points).to(device).float().unsqueeze(0).permute((0,2,1))
    target_torch = torch.tensor(target).to(device).long().unsqueeze(0)
    out = model(partial_pc_torch, query_torch) #(B ,2, num_pts)
    pred_prob, _ = torch.max(torch.exp(out), dim=1)
    pred_prob = pred_prob.squeeze().detach()
    pred = torch.argmax(torch.exp(out), dim=1).squeeze()

    print(f"num pos pred points: {torch.sum(pred_prob[pred==1])}")
    print(f"pred prob of pos points: {pred_prob[pred==1]}")

    ## vis pc
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np.array(partial_pc))
    pcd.colors = open3d.utility.Vector3dVector(np.array([[1,0,0]]))
    open3d.visualization.draw_geometries([pcd]) 


    # vis ground truth
    target_expanded = target[:,np.newaxis]
    red = np.array([1,0,0])
    green = np.array([0,1,0])
    red = np.tile(red, (num_pts,1))
    green = np.tile(green, (num_pts,1))
    colors = np.where(target_expanded==1, red, green)
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np.array(query_points))
    pcd.colors = open3d.utility.Vector3dVector(colors)
    #open3d.visualization.draw_geometries([pcd]) 

    # vis pred
    pred_expanded = pred.cpu().numpy()[:,np.newaxis]
    red = np.array([1,0,0])
    green = np.array([0,1,0])
    red = np.tile(red, (num_pts,1))
    #red[:,0] = pred_prob.detach().cpu().numpy().astype(float)
    green = np.tile(green, (num_pts,1))
    colors = np.where(pred_expanded==1, red, green)
    pcd_pred = open3d.geometry.PointCloud()
    pcd_pred.points = open3d.utility.Vector3dVector(np.array(query_points))
    pcd_pred.colors = open3d.utility.Vector3dVector(colors)
    open3d.visualization.draw_geometries([pcd_pred.translate((0.25,0,0)), pcd]) 


    
