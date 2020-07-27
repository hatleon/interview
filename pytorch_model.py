import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from .Model import Model

class TransE(Model):
	def __init__(self, config):
		super(TransE, self).__init__(config)
		self.ent_embeddings = nn.Embedding(self.config.entTotal, self.config.hidden_size)
		self.rel_embeddings = nn.Embedding(self.config.relTotal, self.config.hidden_size)
		self.weight_embeddings = nn.Parameter(self.config.hidden_size)
		self.criterion = nn.MarginRankingLoss(self.config.margin, False)
		self.init_weights()
		
	def init_weights(self):
		nn.init.xavier_uniform(self.ent_embeddings.weight.data)
		nn.init.xavier_uniform(self.rel_embeddings.weight.data)

	def _calc(self, h, t, r):
		return torch.norm(h + r - t, self.config.p_norm, -1)
	
	def loss(self, p_score, n_score):
		if self.config.use_gpu:
			y = Variable(torch.Tensor([-1]).cuda())
		else:
			y = Variable(torch.Tensor([-1]))
		return self.criterion(p_score, n_score, y)

	def forward(self):
		h_pos = self.ent_embeddings(triple_pos[:, 0])  // m*n batch_size为m表示实体数量，n为隐藏层维度大小
		t_pos = self.ent_embeddings(triple_pos[:, 1])
		r_pos = self.rel_embeddings(triple_pos[:, 2])
		weight_pos = self.unsqueeze(self.weight_embeddings, -1)  // 扩充成n*1维
		weight_pos = torch.matmul(r_pos, weight_pos)  // 矩阵乘得权重大小，结果为m*1维

		h_neg = self.ent_embeddings(triple_neg[:, 0])
		t_neg = self.ent_embeddings(triple_neg[:, 1])
		r_neg = self.rel_embeddings(triple_neg[:, 2])
		weight_neg = self.unsqueeze(self.weight_embeddings, -1)
		weight_neg = torch.matmul(r_neg, weight_neg)

		p_score = r_pos + t_pos - weight_pos*h_pos  //数乘 r+t-weight*h
		n_score = r_neg + t_neg - weight_neg*h_neg
		return self.loss(p_score, n_score)	

	def predict(self):
		h = self.ent_embeddings(self.batch_h)
		t = self.ent_embeddings(self.batch_t)
		r = self.rel_embeddings(self.batch_r)
		score = self._calc(h, t, r)
		return score.cpu().data.numpy()	