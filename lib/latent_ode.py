from lib.base_models import Baseline
import torch
import numpy as np
import lib.utils as utils

import torch.nn.functional as F

import torch.nn as nn
class LearnedPositionEncoding(nn.Embedding):
    def __init__(self,d_model, max_len,out_dim, dropout = 0.1):
        super().__init__(max_len, d_model)
        self.dropout = nn.Dropout(p = dropout)
        self.route_proj = nn.Sequential(
              nn.Linear(max_len,out_dim//2),
              nn.ReLU(),
              nn.Linear(out_dim//2,out_dim)
		)
        

    def forward(self, x):
        
        weight = self.weight.data.unsqueeze(1)
        x = x + weight[:x.size(0),:]
        x = self.dropout(x) 
        x = x.permute(0,2,1)
        x = self.route_proj(x)
        return x


class CoupledODE(Baseline):
    def __init__(self, encoder_z0,ode_hidden_dim,hidden_dim,rec_hidden_dim,decoder_node, diffeq_solver,device):

        super(CoupledODE, self).__init__(device=device)

        self.diffeq_solver = diffeq_solver
        self.decoder_node = decoder_node
        self.ode_hidden_dim =ode_hidden_dim
        self.encoder_z0 = encoder_z0

        # Shared with edge ODE
        self.embedding = LearnedPositionEncoding(hidden_dim,ode_hidden_dim,rec_hidden_dim)


    def get_reconstruction(self, graph_batch, num_atoms):
        #regularization
        # ps_emb = self.embedding(graph_batch.pos.T)[:,1:,:]

        #Encoder:
        first_point_enc = self.encoder_z0(graph_batch) 

        graph_batch.x = first_point_enc
        time_steps_to_predict = graph_batch.y.shape[0]

        sol_y = self.diffeq_solver(graph_batch,time_steps_to_predict)
        ps_emb = 0

        # Decoder:
        pred_node = self.decoder_node(sol_y[:,:,:-1*num_atoms]) #[ N, time_length, N+D]

        return pred_node,ps_emb,sol_y[:,:,:-1*num_atoms]


