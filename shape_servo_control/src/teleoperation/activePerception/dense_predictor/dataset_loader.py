import torch
import os
import numpy as np
import ast
import random
from torch.utils.data import Dataset
import pickle
from farthest_point_sampling import *


class DensePredictorDataset(Dataset):
    """predict attachment point using segmentation"""


    def __init__(self, dataset_path):
        """
        Args:

        """ 
        random.seed(2021)
        self.dataset_path = dataset_path

        self.filenames = os.listdir(self.dataset_path)
        random.shuffle(self.filenames)


    def load_pickle_data(self, filename):
        if os.path.getsize(os.path.join(self.dataset_path, filename)) == 0: 
            print(filename)
        with open(os.path.join(self.dataset_path, filename), 'rb') as handle:
            return pickle.load(handle)            

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):        
        sample = self.load_pickle_data(self.filenames[idx])

        pc = torch.tensor(sample["partial_pc"]).permute(1,0).float()  # shape (3, num_pts) -dataLoader-> (B, 3, num_pts)
        query_points = torch.tensor(sample["grid_query_points"]).permute(1,0).float() # shape (3, num_pts) -dataLoader-> (B, 3, num_pts)
        query_points_labels = torch.tensor(sample["grid_query_points_labels"]).long() # shape (num_pts,) -dataLoader-> (B, num_pts)
        
        # query_points = torch.tensor(sample["random_query_points"]).permute(1,0).float() # shape (3, num_pts) -dataLoader-> (B, 3, num_pts)
        # query_points_labels = torch.tensor(sample["random_query_points_labels"]).float() # shape (num_pts,) -dataLoader-> (B, num_pts)
        
        sample = (pc, query_points, query_points_labels) 

        
        return sample    


class SimpleDensePredictorDataset(Dataset):
    """predict attachment point using segmentation without additional query points"""


    def __init__(self, dataset_path):
        """
        Args:

        """ 
        random.seed(2021)
        self.dataset_path = dataset_path

        self.filenames = os.listdir(self.dataset_path)
        random.shuffle(self.filenames)


    def load_pickle_data(self, filename):
        if os.path.getsize(os.path.join(self.dataset_path, filename)) == 0: 
            print(filename)
        with open(os.path.join(self.dataset_path, filename), 'rb') as handle:
            return pickle.load(handle)            

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):        
        sample = self.load_pickle_data(self.filenames[idx])

        pc = torch.tensor(sample["partial_pc"]).permute(1,0).float()  # shape (3, num_pts) -dataLoader-> (B, 3, num_pts)
        labels = torch.tensor(sample["partial_pc_labels"]).long() # shape (num_pts,) -dataLoader-> (B, num_pts)

        sample = (pc, labels) 

        return sample                



        