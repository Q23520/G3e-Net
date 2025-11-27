import torch
import torch.nn as nn
import torch.nn.functional as F
from models.AGCN.graphconv import SGC_LL, DenseMol, GraphGatherMol

class GraphTopologyMol:
    """管理批量分子的图拓扑信息，处理原子特征和拉普拉斯矩阵"""

    def __init__(self, n_feat=75, batch_size=50, max_atom=128):
        """
        参数:
        n_feat: int - 每个原子的特征数量
        batch_size: int - 批大小
        max_atom: int - 分子最大原子数
        """
        self.n_feat = n_feat
        self.batch_size = batch_size
        self.max_atom = max_atom

    def pad_data2sparse(self, graph):
        """填充原子特征矩阵"""
        feature = graph.node_features  # shape: (N, F)
        pad_size = self.max_atom - feature.shape[0]
        feature_pad = F.pad(feature, (0, 0, 0, pad_size), "constant", 0)
        return feature_pad, torch.tensor([feature.shape[0], -1], dtype=torch.int32)

    def pad_Lap2sparse(self, graph):
        """填充拉普拉斯矩阵"""
        L = graph.Laplacian  # shape: (N, N)
        pad_size = self.max_atom - L.shape[0]
        L_pad = F.pad(L, (0, pad_size, 0, pad_size), "constant", 0)
        return L_pad, torch.tensor(L.shape, dtype=torch.int32)

    def batch_to_tensor(self, batch):
        """将批量图数据转换为 PyTorch 张量"""
        mol_atoms, mol_slice, Laplacians, L_slice = [], [], [], []

        for graph in batch:
            data, shape = self.pad_data2sparse(graph)
            mol_atoms.append(data)
            mol_slice.append(shape)

            L, shape = self.pad_Lap2sparse(graph)
            Laplacians.append(L)
            L_slice.append(shape)

        return {
            "atom_features": torch.stack(mol_atoms),
            "data_slice": torch.stack(mol_slice),
            "laplacians": torch.stack(Laplacians),
            "L_slice": torch.stack(L_slice)
        }


class SequentialGraphMol(nn.Module):
    """基于 PyTorch 的分子图神经网络模型"""

    def __init__(self, n_feat, batch_size=50, max_atom=128):
        """
        参数:
        n_feat: int - 每个节点的特征数
        """
        super(SequentialGraphMol, self).__init__()
        self.graph_topology = GraphTopologyMol(n_feat, batch_size, max_atom)
        self.output = None
        self.layers = nn.ModuleList()
        self.res_L_set = []
        self.res_W_set = []

    def add(self, layer):
        """添加层到模型"""
        self.layers.append(layer)

    def forward(self, batch):
        """前向传播"""
        batch_data = self.graph_topology.batch_to_tensor(batch)
        self.output = batch_data["atom_features"]

        for layer in self.layers:
            if isinstance(layer, SGC_LL):
                self.output, res_L, res_W = layer(
                    self.output,
                    batch_data["laplacians"],
                    batch_data["data_slice"]
                )
                self.res_L_set.append(res_L)
                self.res_W_set.append(res_W)
            else:
                self.output = layer(self.output)
        return self.output

    def get_resL_set(self):
        return self.res_L_set

    def get_resW_set(self):
        return self.res_W_set
