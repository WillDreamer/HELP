import numpy as np
import torch
from torch_geometric.data import DataLoader,Data
from torch.utils.data import DataLoader as Loader
from tqdm import tqdm
import math
from scipy.linalg import block_diag
import lib.utils as utils
import pandas as pd
import os

def softmax(x):
    
    max = np.max(x,axis=-1,keepdims=True) #returns max of each row and keeps same dims
    e_x = np.exp(x - max) #subtracts each row with its max value
    sum = np.sum(e_x,axis=-1,keepdims=True) #returns sum of each row and keeps same dims
    f_x = e_x / sum 
    return f_x

class ParseData(torch.utils.data.Dataset):
    def __init__(self, args, is_train=True):
        self.args = args
        self.datapath = args.datapath
        self.random_seed = args.random_seed
        self.batch_size = args.batch_size
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        self.features_list = []
        self.graph_list = []
        self.target_list = []
        self.edge_list = []
        for idd, file in enumerate(os.listdir(self.args.datapath)):
            if 'npy' in file:
                features = np.load(self.args.datapath  + file,allow_pickle=True).item()['pheromone'].mean(axis=1) # [T,N,N]
                graphs = np.load(self.args.datapath  + file,allow_pickle=True).item()['city'].values.astype('float') # [N,N]
                target = np.load(self.args.datapath  + file,allow_pickle=True).item()['target']
                route = np.load(self.args.datapath  + file,allow_pickle=True).item()['route']

                route_min = np.zeros((route.shape[0],route.shape[2])) #[T,N]
                for idx,i in enumerate(np.argmin(target.squeeze(-1),axis=1)):
                    route_min[idx,:] = route[idx,i,:]

                for k in range(features.shape[0]):
                    transProb = 100/graphs
                    for i in range(len(transProb)):
                        for j in range(len(transProb)):
                            if i != j:
                                features[k,i,j] = pow(features[k,i,j], 2) * pow(transProb[i,j], 1)
                            else:
                                features[k,i,j] = 0
                for i in range(features.shape[0]):
                    # min_f = np.min(features[i,:,:])
                    # max_f = np.min(features[i,:,:],axis=-1,keepdims=True)
                    sum = np.sum(features[i,:,:],axis=-1,keepdims=True)
                    features[i,:,:] = (features[i,:,:]) / sum
                
                num_states = graphs.shape[0]
                graphs = graphs * (1 - np.identity(num_states))
                graphs/=100
                threshold = np.percentile(graphs, 30)
                edge = np.where(graphs < threshold, 1, 0)
                   
                self.features_list.append(features)
                self.graph_list.append(graphs)
                self.target_list.append(target)
                self.edge_list.append(edge)
            
        if is_train:
            self.features_list = self.features_list[:int(0.8*len(self.features_list))]
            self.graph_list =self.graph_list[:int(0.8*len(self.graph_list))]
            self.target_list = self.target_list[:int(0.8*len(self.target_list))]
            self.edge_list = self.edge_list[:int(0.8*len(self.edge_list))]
        else:
            self.features_list = self.features_list[int(0.8*len(self.features_list)):]
            self.graph_list =self.graph_list[int(0.8*len(self.graph_list)):]
            self.target_list = self.target_list[int(0.8*len(self.target_list)):]
            self.edge_list = self.edge_list[int(0.8*len(self.edge_list)):]

        
    def __len__(self):
        return len(self.features_list)

    def __getitem__(self, idx):
        x = torch.from_numpy(np.asarray(np.array(self.features_list[idx][40, :, :]),'float32'))
        y = torch.from_numpy(np.asarray(np.array(self.features_list[idx][41:, :, :]),'float32')) # truth shape  [T, N, N]
        dis = torch.from_numpy(np.asarray(np.array(self.graph_list[idx]),'float32'))
        target = torch.from_numpy(np.asarray(np.array(self.target_list[idx]),'float32'))
        edge = torch.from_numpy(np.asarray(np.array(self.edge_list[idx]),'float32'))
        dataset = Data(x=x, y=y, pos=dis,adj=edge)
        
        return dataset,target



    