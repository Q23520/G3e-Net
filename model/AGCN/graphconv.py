import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SGC_LL(nn.Module):
    def __init__(self,
                 output_dim,
                 input_dim,
                 batch_size,
                 activation,
                 dropout=None,
                 K=2,
                 save_lap=False,
                 save_output=False,
                 **kwargs):
        super(SGC_LL, self).__init__()

        self.dropout = dropout

        self.activation_fn = self.get_activation_fn(activation)

        self.batch_size = batch_size
        self.nb_filter = output_dim
        self.n_atom_feature = input_dim
        self.vars = nn.ParameterDict()
        self.bias = True
        self.K = K
        self.save_lap = save_lap
        self.save_output = save_output  # 如果是 True，则将结果保存到模型中并作为残差使用

        self.build()

    def build(self):
        """ 初始化权重和偏置 """
        self.vars['weight'] = nn.Parameter(torch.randn(self.n_atom_feature * self.K, self.nb_filter) * np.sqrt(2.0 / (self.n_atom_feature * self.K + self.nb_filter)))
        if self.bias:
            self.vars['bias'] = nn.Parameter(torch.zeros(self.nb_filter))
        self.vars['M_L'] = nn.Parameter(torch.randn(self.n_atom_feature, self.n_atom_feature) * np.sqrt(2.0 / self.n_atom_feature))
        self.vars['alpha'] = nn.Parameter(torch.tensor(1.0))
        self.alpha = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=True)

    def forward(self, x, La):
        """
        :param x: 包含节点特征、拉普拉斯矩阵等的字典
        :return: 输出张量
        """
        node_features = x
        Laplacian = La
        # mol_slice = x['data_slice']
        # L_slice = x['lap_slice']

        # 执行光谱卷积和拉普拉斯更新
        node_features, L_updated, W_updated = self.specgraph_LL(node_features, Laplacian, self.vars, self.K, self.n_atom_feature)

        # 应用激活函数和 Dropout
        activated_nodes = []
        for i in range(self.batch_size):
            activated_mol = self.activation_fn(node_features[i])
            # activated_mol = F.leaky_relu(node_features[i])

            if self.dropout is not None:
                activated_mol = F.dropout(activated_mol, p=self.dropout, training=self.training)
            activated_nodes.append(activated_mol)

        activate_nodes = torch.stack(activated_nodes)
        Laplacian_updated = torch.stack(L_updated)
        Weight_updated = torch.stack(W_updated)
        return activate_nodes, Laplacian_updated, Weight_updated

    def specgraph_LL(self, node_features, Laplacian, vars, K, Fin):
        """
        该函数执行：
        1）使用更新的马哈拉诺比斯权重 M_L 学习拉普拉斯矩阵
        2）通过切比雪夫近似进行谱卷积

        Args:
            node_features: 每个分子的节点特征
            Laplacian: 初始拉普拉斯矩阵
            mol_slice: 用于提取特征的索引
            L_slice: 用于提取拉普拉斯矩阵的索引
            vars: 训练变量：w, b 和 M_L
            K: 切比雪夫多项式的阶数
            Fin: 输入特征维度

        Returns:
            经过卷积后的特征列表，更新的拉普拉斯矩阵，更新的相似度矩阵
        """
        batch_size = len(node_features)
        x_conved = []
        M_L = vars['M_L']
        # alpha = vars['alpha']
        res_L_updated = []
        res_W_updated = []

        for mol_id in range(batch_size):
            x = node_features[mol_id]  # max_atom x Fin
            LL = Laplacian[mol_id]  # max_atom x max_atom
            max_atom = LL.shape[0]  # 获取当前分子节点数量
            ################### AGCN ######################
            res_L, res_W = self.compute_laplacian(x, M_L)
            res_L = self.clip_by_average_norm(res_L, 1)  # 归一化
            res_L = F.leaky_relu(res_L, inplace=True)  # 进行激活
            res_W_updated.append(res_W)
            L_all = res_L + LL  # 最终拉普拉斯矩阵
            res_L_updated.append(res_L)

            # 切比雪夫多项式变换
            x0 = x
            x = x0.unsqueeze(0)  # x -> 1 x M x Fin

            # 切比雪夫递推公式: T_k(x) = 2x T_{k-1}(x) - T_{k-2}(x)
            if K > 1:
                x1 = torch.mm(L_all, x0)
                x = torch.cat([x, x1.unsqueeze(0)], dim=0)  # K x M x Fin

            for k in range(2, K):
                x2 = 2 * torch.mm(L_all, x1) - x0  # M x Fin
                x = torch.cat([x, x2.unsqueeze(0)], dim=0)
                x0, x1 = x1, x2

            # 重新调整形状
            # M = x_indices[0].item()
            M = node_features.shape[1]
            x = x.view(K, M, Fin)
            x = x.permute(1, 2, 0).contiguous().view(M, K * Fin)  # M x (Fin*K)

            # 线性变换
            w = vars['weight']  # 变换权重矩阵 (Fin*K x Fout)
            b = vars['bias']
            x = torch.mm(x, w) + b  # M x Fout

            # 填充到 max_atom 维度
            pad_size = max_atom - M
            x = F.pad(x, (0, 0, 0, pad_size), "constant", 0)  # M x Fout
            x_conved.append(x)

        return x_conved, res_L_updated, res_W_updated

    # 计算新的拉普拉斯矩阵
    def compute_laplacian(self, x, M, sigma=1.0):
        # Mahalanobis 变换
        x_w = torch.mm(x, M)

        # 计算欧几里得距离矩阵（广播方式）
        diff = x_w[:, None, :] - x_w[None, :, :]
        dist_matrix = torch.norm(diff, dim=-1)

        # 计算相似度矩阵 W
        W = torch.exp(-dist_matrix)

        # 计算度矩阵 D（对角形式）
        d = W.sum(dim=1) + 1e-7  # 避免除 0
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(d))

        # 计算归一化拉普拉斯矩阵
        I = torch.eye(W.size(0)).to(W.device)
        L = I - D_inv_sqrt @ W @ D_inv_sqrt  # 矩阵乘法

        # # 计算马氏距离
        # diff = x[:, None, :] - x[None, :, :]
        # dist_matrix = torch.sqrt(torch.sum(torch.matmul(diff, M) * diff, dim=-1))
        #
        # # 计算高斯核
        # W = torch.exp(-dist_matrix ** 2 / (2 * sigma ** 2))
        #
        # # 计算度矩阵 D
        # d = W.sum(dim=1) + 1e-7  # 加上小常数以避免除以零
        # D_inv_sqrt = torch.diag(1.0 / torch.sqrt(d))
        #
        # # 计算归一化残差图拉普拉斯矩阵
        # I = torch.eye(W.size(0)).to(x.device)
        # L = I - D_inv_sqrt @ W @ D_inv_sqrt  # 归一化拉普拉斯矩阵

        return L, W


    def get_activation_fn(self, activation):
        """根据字符串输入获取激活函数"""
        if activation == 'relu':
            return F.leaky_relu
        elif activation == 'sigmoid':
            return torch.sigmoid
        elif activation == 'tanh':
            return torch.tanh
        elif activation == 'leaky_relu':
            return F.leaky_relu
        else:
            raise ValueError("未知的激活函数: {}".format(activation))


    def clip_by_average_norm(self, tensor, clip_norm):
        """
        使用 PyTorch 实现 TensorFlow 的 tf.clip_by_average_norm。

        :param tensor: 输入张量
        :param clip_norm: 限制的平均 L2 范数
        :return: 裁剪后的张量
        """
        num_elements = tensor.numel()  # 计算元素总数
        avg_norm = torch.norm(tensor) / num_elements  # 计算平均 L2 范数
        factor = clip_norm / (avg_norm + 1e-6)  # 计算裁剪比例，防止除零
        factor = torch.clamp(factor, max=1.0)  # 限制最大不超过 1.0
        return tensor * factor  # 缩放张量



class DenseMol(nn.Module):
    def __init__(self, output_dim, input_dim, activation, init='xavier_uniform', bias=True):
        super(DenseMol, self).__init__()

        self.output_dim = output_dim
        self.input_dim = input_dim
        self.bias = bias

        # Initialize weights and bias
        self.W = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.bias:
            self.b = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.b = None

        # Weight initialization
        self.reset_parameters(init)

        # Activation function
        self.activation = self.get_activation_function(activation)

    def reset_parameters(self, init):
        """Initialize the weights and biases."""
        if init == 'xavier_uniform':
            nn.init.xavier_uniform_(self.W)
        elif init == 'he_uniform':
            nn.init.kaiming_uniform_(self.W, mode='fan_in', nonlinearity='relu')
        elif init == 'glorot_uniform':
            nn.init.xavier_uniform_(self.W)
        else:
            raise ValueError("Unknown initializer: {}".format(init))

        if self.bias is not None:
            nn.init.zeros_(self.b)

    def get_activation_function(self, activation):
        """Return the corresponding activation function."""
        if activation == 'relu':
            return F.relu
        elif activation == 'sigmoid':
            return torch.sigmoid
        elif activation == 'tanh':
            return torch.tanh
        elif activation == 'leaky_relu':
            return F.leaky_relu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        """Forward pass."""
        output = []
        for xx in x:
            xx = torch.matmul(xx, self.W)
            if self.bias is not None:
                xx += self.b
            if self.activation:
                xx = self.activation(xx)
            output.append(xx)
        output = torch.stack(output)
        return output


class GraphGatherMol(nn.Module):
    def __init__(self, batch_size, activation='relu'):
        """
        参数:
          batch_size: int
            批次中分子的数量。
          activation: str 或 callable
            聚合后要应用的激活函数。若为字符串，可选择'relu'或'sigmoid'等。
        """
        super(GraphGatherMol, self).__init__()
        self.batch_size = batch_size

        # 根据传入的 activation 参数选择激活函数
        if isinstance(activation, str):
            if activation.lower() == 'relu':
                self.activation = F.relu
            elif activation.lower() == 'sigmoid':
                self.activation = torch.sigmoid
            else:
                raise ValueError("不支持的激活函数: " + activation)
        else:
            self.activation = activation

    def forward(self, x):
        """
        前向传播方法

        参数:
          x: dict，包含以下键：
             - 'node_features': 一个列表，每个元素是一个 torch.Tensor，
               形状为 (n_atoms, n_feat)，表示每个分子的原子特征。
             - 'data_slice': 一个 torch.Tensor，形状为 (batch_size, 2)。
               每一行存放切片信息（例如，有效原子数量和有效特征数）。

        返回:
          torch.Tensor，形状为 (batch_size, n_feat)，表示每个分子的聚合特征表示。
        """
        node_features = x['node_features']
        data_slice = x['data_slice']
        mol_reps = self.graph_gather_mol(node_features, data_slice, self.batch_size)
        return self.activation(mol_reps)

    def graph_gather_mol(self, atoms, mol_slice, batch_size):
        """
        实现将每个分子的原子特征聚合为分子级特征表示。

        参数:
          atoms: list of torch.Tensor
            每个元素的形状为 (n_atoms, n_feat)，表示一个分子的原子特征。
          mol_slice: torch.Tensor, 形状为 (batch_size, 2)
            每个分子的切片信息，通常包含有效原子数和有效特征数。
          batch_size: int
            批次大小。

        返回:
          torch.Tensor, 形状为 (batch_size, n_feat)
            每个分子的聚合特征。
        """
        mol_feature = []
        for mol_id in range(batch_size):
            x = atoms[mol_id]  # 当前分子的原子特征，形状 (n_atoms, n_feat)
            # 获取该分子的切片信息，假设切片信息包含 [有效原子数, 有效特征数]
            slice_info = mol_slice[mol_id]  # shape: (2,)
            valid_atoms = int(slice_info[0].item())
            valid_features = int(slice_info[1].item())
            # 根据切片信息提取有效部分
            x = x[:valid_atoms, :valid_features]
            # 聚合操作：沿原子维度求和，得到分子的特征表示
            f_mol = torch.sum(x, dim=0)
            mol_feature.append(f_mol)
        # 将所有分子的特征堆叠成一个张量，形状为 (batch_size, n_feat)
        mol_reps = torch.stack(mol_feature, dim=0)
        return mol_reps
