import torch
import torch.nn as nn
import torch.nn.functional as F
from models.AGCN.graphconv import SGC_LL, DenseMol
# from models.AGCN.graph_topology import SequentialGraphMol
from utils import *



class SimpleAGCN(nn.Module):
    """
    A simple example of AGCN implemented in PyTorch
    """

    def __init__(self, n_features, batch_size, hyper_parameters):
        super(SimpleAGCN, self).__init__()

        # n_features = 5
        # batch_size = 2
        # hyper_parameters = {
        #     'max_hop_K': 3,
        #     'final_feature_n': 1,
        #     'l_n_filters': [64, 128, 256, 512],
        #     'batch_size': 32
        # }

        # self.max_atom = max_atom
        self.K = hyper_parameters['max_hop_K']
        self.final_feature_n = hyper_parameters['final_feature_n']
        self.l_n_filters = hyper_parameters['l_n_filters']

        # assign the number of feature at output of the SGC_LL layer
        n_filters_1 = self.l_n_filters[0]
        n_filters_2 = self.l_n_filters[1]
        # n_filters_3 = self.l_n_filters[2]
        # n_filters_4 = self.l_n_filters[3]

        # Network Architecture - 4 SGC layers, similar to the original AGCN
        # self.graph_model = nn.ModuleList([
        #     # SequentialGraphMol(n_features, batch_size, self.max_atom),
        #     SGC_LL(n_filters_1, n_features, batch_size, K=self.K, activation='relu'),
        #     SGC_LL(n_filters_2, n_filters_1, batch_size, K=self.K, activation='relu'),
        #     SGC_LL(n_filters_3, n_filters_2, batch_size, K=self.K, activation='relu'),
        #     SGC_LL(n_filters_4, n_filters_3, batch_size, K=self.K, activation='relu')
        #     # GraphGatherMol(batch_size, activation=torch.tanh)
        # ])

        self.sgc1 = SGC_LL(n_filters_1, n_features, batch_size, K=self.K, activation='leaky_relu')
        self.sgc2 = SGC_LL(n_filters_2, n_filters_1, batch_size, K=self.K, activation='leaky_relu')
        # self.sgc3 = SGC_LL(n_filters_3, n_filters_2, batch_size, K=self.K, activation='relu')
        # self.sgc4 = SGC_LL(n_filters_4, n_filters_3, batch_size, K=self.K, activation='relu')
        self.dense = DenseMol(self.final_feature_n, n_filters_2, activation='leaky_relu')

    def forward(self, x):

        # for layer in self.graph_model:
        #     x = layer(x, La)

        # 拉普拉斯矩阵
        La = batch_compute_laplacian(x)

        x1, La1, W1 = self.sgc1(x, La)
        x2, La2, W2 = self.sgc2(x1, La1)
        # x3, La3, W3 = self.sgc3(x2, La2)
        # x4, La4, W4 = self.sgc4(x3, La3)

        x_out = self.dense(x2)

        return x_out

