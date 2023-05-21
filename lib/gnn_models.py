import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
import math
import lib.utils as utils
from torch_geometric.nn.inits import glorot
from torch_scatter import scatter_add
import torch_sparse
from torch import FloatTensor

class GNN(nn.Module):
    '''
    wrap up multiple layers
    '''
    def __init__(self, args, in_dim, n_hid,dropout = 0.2):
        super(GNN, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.adapt_w = nn.Sequential(
            nn.Linear(in_dim,n_hid//2),
            nn.ReLU(),
            nn.Linear(n_hid//2,n_hid),
        )
        # utils.init_network_weights(self.adapt_w)
        
    def forward(self, graph):  
        x = graph.x
        h_t = self.drop(F.relu(self.adapt_w(x)))  #initial input for encoder
        return h_t


class Node_GCN(nn.Module):
    def __init__(
            self,
            feat_dim: int,
            hidden_dim: int,
            class_dim: int,
            k: int = 2,
            dropout: float = 0.5,
            use_relu: bool = True
    ):
        super(Node_GCN, self).__init__()
        self.dropout = dropout
        self.k = k
        self.act = F.relu if use_relu else lambda x: x
        self.use_relu = use_relu
        self.w_embed = nn.Parameter(
            torch.zeros(size=(feat_dim, hidden_dim)),
            requires_grad=True
        )
        self.w_classify = nn.Parameter(
            torch.zeros(size=((2 ** (self.k + 1) - 1) * hidden_dim, class_dim)),
            requires_grad=True
        )
        self.params = [self.w_embed, self.w_classify]
        self.initialized = False
        self.a1 = None
        self.a2 = None
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.w_embed)
        nn.init.xavier_uniform_(self.w_classify)

    @staticmethod
    def _indicator(sp_tensor: torch.sparse.Tensor) -> torch.sparse.Tensor:
        csp = sp_tensor.coalesce()
        return torch.sparse_coo_tensor(
            indices=csp.indices(),
            values=torch.where(csp.values() > 0, 1, 0),
            size=csp.size(),
            dtype=torch.float
        )

    @staticmethod
    def _spspmm(sp1: torch.sparse.Tensor, sp2: torch.sparse.Tensor) -> torch.sparse.Tensor:
        assert sp1.shape[1] == sp2.shape[0], 'Cannot multiply size %s with %s' % (sp1.shape, sp2.shape)
        # sp1, sp2 = sp1.coalesce(), sp2.coalesce()
        index1, value1 = sp1.indices(), sp1.values()
        index2, value2 = sp2.indices(), sp2.values()
        m, n, k = sp1.shape[0], sp1.shape[1], sp2.shape[1]
        indices, values = torch_sparse.spspmm(index1, value1, index2, value2, m, n, k)
        return torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=(m, k),
            dtype=torch.float
        )

    @classmethod
    def _adj_norm(cls, adj: torch.sparse.Tensor) -> torch.sparse.Tensor:
        n = adj.size(0)
        d_diag = torch.pow(torch.sum(adj, dim=1).values(), -0.5)
        d_diag = torch.where(torch.isinf(d_diag), torch.full_like(d_diag, 0), d_diag)
        d_tiled = torch.sparse_coo_tensor(
            indices=[list(range(n)), list(range(n))],
            values=d_diag,
            size=(n, n)
        )
        return cls._spspmm(cls._spspmm(d_tiled, adj), d_tiled)

    def _prepare_prop(self, adj):
        n = adj.size(0)
        device = adj.device
        self.initialized = True
        sp_eye = torch.sparse_coo_tensor(
            indices=[list(range(n)), list(range(n))],
            values=[1.0] * n,
            size=(n, n),
            dtype=torch.float
        ).to(device)
        # initialize A1, A2
        self.a1 = adj - sp_eye
        # a2 = self._spspmm(adj, adj) - adj - sp_eye
        self.a2 = torch.sparse.mm(adj, adj) - adj - sp_eye
        # # norm A1 A2
        # self.a1 = self._adj_norm(a1)
        # self.a2 = self._adj_norm(a2)

    def forward(self, x: FloatTensor, adj: torch.sparse.Tensor) -> FloatTensor:
        if not self.initialized:
            self._prepare_prop(adj)
        # H2GCN propagation
        rs = [self.act(torch.mm(x, self.w_embed))]
        for i in range(self.k):
            r_last = rs[-1]
            r1 = torch.spmm(self.a1, r_last)
            r2 = torch.spmm(self.a2, r_last)
            rs.append(self.act(torch.cat([r1, r2], dim=1)))
        r_final = torch.cat(rs, dim=1)
        r_final = F.dropout(r_final, self.dropout, training=self.training)
        return torch.softmax(torch.mm(r_final, self.w_classify), dim=1)


class GAT(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, dropout, alpha=0.1, n_heads=2):

        super(GAT, self).__init__()
        self.dropout = dropout 
        
        self.attentions = [GraphAttentionLayer(n_feat, n_hid, dropout=dropout, alpha=alpha, concat=True) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)  
        self.out_att = GraphAttentionLayer(n_hid * n_heads, n_class, dropout=dropout,alpha=alpha, concat=False)
        self.norm = torch.nn.LayerNorm(n_feat)
    
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)  
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)  
        x = F.dropout(x, self.dropout, training=self.training)   
        x = F.elu(self.out_att(x, adj))  
        x = self.norm(x)
        return F.log_softmax(x, dim=1)  



class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features   
        self.out_features = out_features   
        self.dropout = dropout    
        self.alpha = alpha     
        self.concat = concat   
        
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))  
        nn.init.xavier_uniform_(self.W.data, gain=1.414) 
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)   

        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, inp, adj):

        h = torch.mm(inp, self.W)   # [N, out_features]
        N = h.size()[0]   
        
        a_input = torch.cat([h.repeat(1, N).view(N*N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2*self.out_features)
        # [N, N, 2*out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        
        zero_vec = -1e12 * torch.ones_like(e)   
        attention = torch.where(adj>0, e, zero_vec)   # [N, N]
        attention = F.softmax(attention, dim=1)   
        attention = F.dropout(attention, self.dropout, training=self.training)   
        h_prime = torch.matmul(attention, h)  # [N, N].[N, out_features] => [N, out_features]
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime 
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


