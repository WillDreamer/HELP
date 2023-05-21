from lib.likelihood_eval import *
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
import torch.nn as nn
import torch.nn.functional as F
import torch
import lib.utils as utils
import numpy as np
import random
import pandas as pd
# from concorde.tsp import TSPSolver
import os


class Baseline(nn.Module):
	def __init__(self, device):

		super(Baseline, self).__init__()
		self.device = device

	def compute_all_losses(self, graph_batch ,num_atoms,epo,final=False,test=False):
		'''
		:param batch_dict_encoder:
		:param batch_dict_decoder: dict: 1. time 2. data: (K*N, T2, D)
		:param batch_dict_graph: #[K,T2,N,N], ground_truth graph with log normalization
		:param num_atoms:
		:param kl_coef:
		:return:
		'''

		pred_node,ps_emb,node_repre = self.get_reconstruction(graph_batch, num_atoms = num_atoms) #[ N, time_length, N]
		# pred_node [ N , time_length, d]
		truth = graph_batch.y.permute(1,0,2)
		
		# higher temperature makes it smoother
		time_len = truth.shape[1]
		tem_seq = torch.flip(torch.linspace(0.1,0.9,time_len),dims=[0])
		weight = torch.nn.functional.softmax(torch.linspace(1,time_len,time_len))
		loss = 0
		for ti in range(time_len):
			loss+=weight[ti]*torch.nn.functional.kl_div(F.log_softmax(pred_node[:,ti,:]),F.softmax(truth[:,ti,:]), reduction='sum')
			# loss+=weight[ti]*torch.nn.functional.kl_div(F.log_softmax(pred_node[:,ti,:]/tem_seq[ti]),F.softmax(truth[:,ti,:]/tem_seq[ti]), reduction='sum')

		
		results = {}
		results["loss"] = loss

		if final:
			edge = graph_batch.pos.detach().cpu().numpy()
			a = 1e3*np.identity(edge.shape[0])
			edge+=a
			results["accurate"] = 0
			results["accurate_tour"] = [0]
			# os.system('rm -rf *.res')
			best_fit = 1e10
			for i in range(10):
				city = random.randint(0, num_atoms-1)
				cityTabu = list(range(num_atoms))
				cityTabu.remove(city)
				trans_p = pred_node[:,-1,:].detach().cpu().numpy()
				
				trans_p = pd.DataFrame(data=trans_p,columns=range(num_atoms),index=range(num_atoms))
				
				antCityList = utils.select([city],cityTabu,trans_p)
				edge = pd.DataFrame(data=edge,columns=range(num_atoms),index=range(num_atoms))
				fit = utils.calFitness(antCityList,edge*100)
				if fit < best_fit:
					best_fit = fit
					results["our_tour"] = antCityList
			results["our_solution"] = best_fit

		return results






