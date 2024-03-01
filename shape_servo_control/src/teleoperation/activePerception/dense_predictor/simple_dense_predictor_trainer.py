
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import argparse
import logging
import socket

from architecture import SimpleDensePredictor
from dataset_loader import SimpleDensePredictorDataset

import matplotlib.pyplot as plt

def plot_curves(xs, ys_1, ys_2, x_label="epochs", y_label="losses", label_1="train_losses", label_2="test_losses", title="train test losses dense pred", path="./figures"):
    fig, ax = plt.subplots()
    ax.plot(xs, ys_1, label=label_1)
    ax.plot(xs, ys_2, label=label_2)
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    os.makedirs(path, exist_ok=True)
    fig.savefig(f"{path}/{title}.png")
    plt.cla()
    plt.close(fig) 

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    num_batch = 0
    correct = 0
    false_negative = 0
    false_positive = 0
    pos_size = 0
    neg_size = 0
    size = len(train_loader.dataset)
    for batch_idx, sample in enumerate(train_loader):
        num_batch += 1
    
        pc = sample[0].to(device) #(B, 3, num_pts)
        target = sample[1].to(device) #(B, num_query_pts)
        
        
        optimizer.zero_grad()
        output = model(pc) #(B, 2, num_query_pts)

        loss = F.nll_loss(output, target)
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pred = torch.argmax(torch.exp(output), dim=1) #(B, num_query_pts)
        correct += (pred==target).type(torch.float).sum().item() /  pred.shape[-1]
        false_negative += torch.logical_and(pred==0, target==1).type(torch.float).sum().item() #/  torch.sum(target==1)
        false_positive += torch.logical_and(pred==1, target==0).type(torch.float).sum().item() #/  torch.sum(target==0)
        pos_size += torch.sum(target==1).item()
        neg_size += torch.sum(target==0).item()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(sample), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    avg_train_loss = train_loss/num_batch
    correct /= size
    false_positive /= neg_size
    false_negative /= pos_size

    print('====> Train Epoch: {} Average loss: {:.6f} Accuracy:{:.6f} False positive:{} False negative:{}\n'.format(epoch, avg_train_loss, (100*correct), false_positive, false_negative))  
    logger.info('Train: Average loss: {:.6f}'.format(avg_train_loss))  

    return avg_train_loss, correct, false_positive, false_negative





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
            target = sample[1].to(device)
            
            output = model(pc)

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

    logger.info('Test: Average loss: {:.6f}'.format(avg_test_loss))
    logger.info('Test: Accuracy: {:.0f}%\n'.format(correct))

    print("============================")


    return avg_test_loss, correct, false_positive, false_negative



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)



if __name__ == "__main__":
    torch.manual_seed(2021)
    

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--weight_path', type=str, help="where to save the weight")
    parser.add_argument('--tdp',type=str, help="training data path")
    parser.add_argument('--vdp', type=str, help="validation data path")
    parser.add_argument('--batch_size', type=int, help="batch size")
    parser.add_argument('--epochs', type=int, help="number of epochs")
    parser.add_argument('--plot_category', type=str, help="the name of the folder containing the plot")

    args = parser.parse_args()

    weight_path = args.weight_path
    tdp = args.tdp
    vdp = args.vdp
    batch_size = args.batch_size
    epochs = args.epochs
    plot_category = args.plot_category
    os.makedirs(weight_path, exist_ok=True)

    logger = logging.getLogger(weight_path)
    logger.propagate = False    # no output to console
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(weight_path, "log.txt"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f"Machine: {socket.gethostname()}")


    device = torch.device("cuda")


    # train_len = round(len(os.listdir(dataset_path))*0.95)   
    # test_len = round(len(os.listdir(dataset_path))*0.05)  
    # total_len = train_len + test_len

    # dataset = DensePredictorDataset(dataset_path=dataset_path)
    # train_dataset = torch.utils.data.Subset(dataset, range(0, train_len))
    # test_dataset = torch.utils.data.Subset(dataset, range(train_len, total_len))

    train_dataset = SimpleDensePredictorDataset(dataset_path=tdp)
    test_dataset = SimpleDensePredictorDataset(dataset_path=vdp)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    
    print("training data: ", len(train_dataset))
    print("test data: ", len(test_dataset))
    
    logger.info(f"Train len: {len(train_dataset)}")    
    logger.info(f"Test len: {len(test_dataset)}") 


    model = SimpleDensePredictor(num_classes=2).to(device)  # single robot
    model.apply(weights_init)
  
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 80, gamma=0.1)
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    train_fps = []
    test_fps = []
    train_fns = []
    test_fns = []
    
    for epoch in range(0, epochs):
        logger.info(f"Epoch {epoch}")
        logger.info(f"Lr: {optimizer.param_groups[0]['lr']}")
        train_loss, train_acc, train_fp, train_fn = train(model, device, train_loader, optimizer, epoch)
        scheduler.step()
        test_loss, test_acc, test_fp, test_fn = test(model, device, test_loader, epoch)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        train_fps.append(train_fp)
        train_fns.append(train_fn)
        test_fps.append(test_fp)
        test_fns.append(test_fn)
        
        if epoch % 20 == 0:            
            xs = [i for i in range(epoch+1)]
            plot_curves(xs, train_losses, test_losses, x_label="epochs", y_label="losses", label_1="train_losses", label_2="test_losses", title="train test losses", path=f"./figures/{args.plot_category}")
            plot_curves(xs, train_accs, test_accs, x_label="epochs", y_label="accuracies", label_1="train_accuracies", label_2="test_accuracies", title="train test accuracies", path=f"./figures/{args.plot_category}") 
            plot_curves(xs, train_fps, test_fps, x_label="epochs", y_label="false positive rate", label_1="train", label_2="test", title="train test false positive rate", path=f"./figures/{args.plot_category}")
            plot_curves(xs, train_fns, test_fns, x_label="epochs", y_label="false negative rate", label_1="train", label_2="test", title="train test false negative rate", path=f"./figures/{args.plot_category}")               
            torch.save(model.state_dict(), os.path.join(weight_path, "epoch_" + str(epoch)))

