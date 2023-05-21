import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
import numpy as np
import lib.utils as utils
import torch.nn.functional as F
from scipy.linalg import block_diag
from torch_scatter import scatter_add
import scipy.sparse as sp



class DiffeqSolver(nn.Module):
    def __init__(self, ode_func, method,args,
            odeint_rtol = 1e-3, odeint_atol = 1e-4, device = torch.device("cpu")):
        super(DiffeqSolver, self).__init__()

        self.ode_method = method
        self.device = device
        self.ode_func = ode_func
        self.args = args
        self.num_atoms = args.num_atoms
        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol

    def forward(self, graph_batch,time_steps_to_predict):
        '''
        :param first_point:  [K*N,D]
        :param edge_initials: [K*N*N,D]
        :param time_steps_to_predict: [t]
        :return:
        '''
        # print(graph_batch.x.shape,graph_batch.adj.shape)
        node_edge_initial = torch.cat([graph_batch.x,graph_batch.adj],1)  #[N,128+N]

        # Results
        pred_y = odeint(self.ode_func, node_edge_initial, torch.range(1,time_steps_to_predict),
            rtol=self.odeint_rtol, atol=self.odeint_atol, method = self.ode_method) #[time_length, K*N + K*N*N, D]

        pred_y = pred_y.permute(1,0,2) #[ N, time_length, N+D]

        return pred_y



class CoupledODEFunc(nn.Module):
    def __init__(self, node_ode_func_net,num_atom, dropout,device = torch.device("cpu")):
        """
        input_dim: dimensionality of the input
        latent_dim: dimensionality used for ODE. Analog of a continous latent state
        """
        super(CoupledODEFunc, self).__init__()

        self.device = device
        self.node_ode_func_net = node_ode_func_net  #input: x, edge_index
        self.num_atom = num_atom
        self.nfe = 0
        self.dropout = nn.Dropout(dropout)


    def forward(self, t_local, z, backwards = False):
        """
        Perform one step in solving ODE. Given current data point y and current time point t_local, returns gradient dy/dt at this time point
        t_local: current time point
        z:  [H,E] concat by axis0. H is [K*N,D], E is[K*N*N,D], z is [K*N + K*N*N, D]
        """
        self.nfe += 1
        node_attributes = z[:,:-1*self.num_atom]
        edge_attributes = z[:,-1*self.num_atom:]
        grad_edge = edge_attributes
        grad_node = self.node_ode_func_net(node_attributes,grad_edge) # [N,D]
        # Concat two grad
        grad = self.dropout(torch.cat([grad_node,grad_edge],1)) # [N, N+D]

        return grad











