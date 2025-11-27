import torch
from torch import nn
from models.backbone import DGCNN_Cline
from models.AGCN.basic_AGCN import SimpleAGCN


class DGCNN_Centerline_Graph(nn.Module):
    """DGCNN backbone that fuses observation/boundary points with centerline cues."""
    
    def __init__(self, k, dropout, output_channels, n_features, batch_size,
                 hyper_parameters, emb_dims=4096, feature_transform=False, channel=3):
        super(DGCNN_Centerline_Graph, self).__init__()
        
        self.main_net = DGCNN_Cline(k, dropout, output_channels, emb_dims, feature_transform, channel)
        self.centerline_net = SimpleAGCN(n_features, batch_size, hyper_parameters)
        self.centerline_net_bound = SimpleAGCN(n_features, batch_size, hyper_parameters)
    
    def forward(self, pv_o, pv_b, centerline_o, centerline_b):
        """Forward pass that returns fused features for downstream heads."""
        input_centerline_o = centerline_o[:, :, :5]
        input_centerline_b = centerline_b[:, :, :5]
        
        cline_feature_o = self.centerline_net(input_centerline_o)
        cline_feature_b = self.centerline_net_bound(input_centerline_b)
        
        pv_o_feature = self.process_data_batch_optimized(pv_o, centerline_o, cline_feature_o)
        pv_o_feature = pv_o_feature.permute(0, 2, 1)
        
        pv_b_feature = self.process_data_batch_optimized_boundary(pv_b, centerline_b, cline_feature_b)
        pv_b_feature = pv_b_feature.permute(0, 2, 1)
        
        y_branch = self.main_net(pv_o_feature, pv_b_feature)
        return y_branch
    
    def process_data_batch_optimized(self, data, centerline, feature):
        """
        Vectorised observation pre-processing used by the abdominal model variant.

        Args:
            data: (B, N, M) tensor with observation samples
            feature: (B, F, D) centerline feature bank
            centerline: (B, C, P) raw centerline data (unused but kept for parity)
        """
        B, N, M = data.shape
        _, F, D = feature.shape
        
        # Gather neighbor indices
        idx1 = data[:, :, 3].long()
        idx2 = data[:, :, 4].long()
        idx3 = data[:, :, 5].long()
        idx4 = data[:, :, 6].long()
        
        # Guard against out-of-range values
        idx1 = torch.clamp(idx1, 0, F - 1)
        idx2 = torch.clamp(idx2, 0, F - 1)
        idx3 = torch.clamp(idx3, 0, F - 1)
        idx4 = torch.clamp(idx4, 0, F - 1)
        
        # Fetch centerline embeddings
        part1 = feature.gather(dim=1, index=idx1.unsqueeze(-1).expand(-1, -1, D))
        part2 = feature.gather(dim=1, index=idx2.unsqueeze(-1).expand(-1, -1, D))
        part3 = feature.gather(dim=1, index=idx3.unsqueeze(-1).expand(-1, -1, D))
        part4 = feature.gather(dim=1, index=idx4.unsqueeze(-1).expand(-1, -1, D))
        
        # Reciprocal distances (avoid division by zero)
        min_value = 1e-10
        weights = data[:, :, 7:11]
        clamped_weights = torch.clamp(weights, min=min_value)
        inv_weights = 1 / clamped_weights
        
        # Interpolate feature
        weighted_sum = (part1 * inv_weights[:, :, 0:1] +
                       part2 * inv_weights[:, :, 1:2] +
                       part3 * inv_weights[:, :, 2:3] +
                       part4 * inv_weights[:, :, 3:4])
        
        # Keep xyz, append interpolated descriptor
        result = torch.cat((data[:, :, :3], weighted_sum), dim=-1)
        return result
    
    def process_data_batch_optimized_boundary(self, data, centerline, feature):
        """Boundary pre-processing which mirrors observation flow but keeps extra scalars."""
        B, N, M = data.shape
        _, F, D = feature.shape
        
        idx1 = data[:, :, 3].long()
        idx2 = data[:, :, 4].long()
        idx3 = data[:, :, 5].long()
        idx4 = data[:, :, 6].long()
        
        idx1 = torch.clamp(idx1, 0, F - 1)
        idx2 = torch.clamp(idx2, 0, F - 1)
        idx3 = torch.clamp(idx3, 0, F - 1)
        idx4 = torch.clamp(idx4, 0, F - 1)
        
        part1 = feature.gather(dim=1, index=idx1.unsqueeze(-1).expand(-1, -1, D))
        part2 = feature.gather(dim=1, index=idx2.unsqueeze(-1).expand(-1, -1, D))
        part3 = feature.gather(dim=1, index=idx3.unsqueeze(-1).expand(-1, -1, D))
        part4 = feature.gather(dim=1, index=idx4.unsqueeze(-1).expand(-1, -1, D))
        
        min_value = 1e-10
        weights = data[:, :, 7:11]
        clamped_weights = torch.clamp(weights, min=min_value)
        inv_weights = 1 / clamped_weights
        
        weighted_sum = (part1 * inv_weights[:, :, 0:1] +
                       part2 * inv_weights[:, :, 1:2] +
                       part3 * inv_weights[:, :, 2:3] +
                       part4 * inv_weights[:, :, 3:4])
        
        # Preserve xyz + boundary scalar before descriptor
        result = torch.cat((data[:, :, :3], data[:, :, -1:], weighted_sum), dim=-1)
        return result


class CarotidArtery_Centerline_Graph(nn.Module):
    """Carotid-specific flavor that only references three indices per sample."""
    
    def __init__(self, k, dropout, output_channels, n_features, batch_size,
                 hyper_parameters, emb_dims=4096, feature_transform=False, channel=3):
        super(CarotidArtery_Centerline_Graph, self).__init__()
        
        self.main_net = DGCNN_Cline(k, dropout, output_channels, emb_dims, feature_transform, channel)
        self.centerline_net = SimpleAGCN(n_features, batch_size, hyper_parameters)
        self.centerline_net_bound = SimpleAGCN(n_features, batch_size, hyper_parameters)
    
    def forward(self, pv_o, pv_b, centerline_o, centerline_b):
        """Forward pass identical to the abdominal variant but with carotid inputs."""
        input_centerline_o = centerline_o[:, :, :5]
        input_centerline_b = centerline_b[:, :, :5]
        
        cline_feature_o = self.centerline_net(input_centerline_o)
        cline_feature_b = self.centerline_net_bound(input_centerline_b)
        
        pv_o_feature = self.process_data_batch_optimized(pv_o, centerline_o, cline_feature_o)
        pv_o_feature = pv_o_feature.permute(0, 2, 1)
        
        pv_b_feature = self.process_data_batch_optimized_boundary(pv_b, centerline_b, cline_feature_b)
        pv_b_feature = pv_b_feature.permute(0, 2, 1)
        
        y_branch = self.main_net(pv_o_feature, pv_b_feature)
        return y_branch
    
    def process_data_batch_optimized(self, data, centerline, feature):
        """Carotid observation processing (3 indices instead of 4)."""
        B, N, M = data.shape
        _, F, D = feature.shape
        
        idx1 = data[:, :, 3].long()
        idx2 = data[:, :, 4].long()
        idx3 = data[:, :, 5].long()
        
        idx1 = torch.clamp(idx1, 0, F - 1)
        idx2 = torch.clamp(idx2, 0, F - 1)
        idx3 = torch.clamp(idx3, 0, F - 1)
        
        part1 = feature.gather(dim=1, index=idx1.unsqueeze(-1).expand(-1, -1, D))
        part2 = feature.gather(dim=1, index=idx2.unsqueeze(-1).expand(-1, -1, D))
        part3 = feature.gather(dim=1, index=idx3.unsqueeze(-1).expand(-1, -1, D))
        
        min_value = 1e-10
        weights = data[:, :, 6:9]
        clamped_weights = torch.clamp(weights, min=min_value)
        inv_weights = 1 / clamped_weights
        
        weighted_sum = (part1 * inv_weights[:, :, 0:1] +
                       part2 * inv_weights[:, :, 1:2] +
                       part3 * inv_weights[:, :, 2:3])
        
        result = torch.cat((data[:, :, :3], weighted_sum), dim=-1)
        return result
    
    def process_data_batch_optimized_boundary(self, data, centerline, feature):
        """Carotid boundary processing (mirrors observation but keeps extra scalar)."""
        B, N, M = data.shape
        _, F, D = feature.shape
        
        idx1 = data[:, :, 3].long()
        idx2 = data[:, :, 4].long()
        idx3 = data[:, :, 5].long()
        
        idx1 = torch.clamp(idx1, 0, F - 1)
        idx2 = torch.clamp(idx2, 0, F - 1)
        idx3 = torch.clamp(idx3, 0, F - 1)
        
        part1 = feature.gather(dim=1, index=idx1.unsqueeze(-1).expand(-1, -1, D))
        part2 = feature.gather(dim=1, index=idx2.unsqueeze(-1).expand(-1, -1, D))
        part3 = feature.gather(dim=1, index=idx3.unsqueeze(-1).expand(-1, -1, D))
        
        min_value = 1e-10
        weights = data[:, :, 6:9]
        clamped_weights = torch.clamp(weights, min=min_value)
        inv_weights = 1 / clamped_weights
        
        weighted_sum = (part1 * inv_weights[:, :, 0:1] +
                       part2 * inv_weights[:, :, 1:2] +
                       part3 * inv_weights[:, :, 2:3])
        
        result = torch.cat((data[:, :, :3], data[:, :, -1:], weighted_sum), dim=-1)
        return result


class DGCNN_CenterGraph(nn.Module):
    """Wrapper that exposes the DGCNN + centerline fusion used by abdominal training."""
    
    def __init__(self, k, dropout, output_channels, n_features, batch_size,
                 hyper_parameters, emb_dims):
        super(DGCNN_CenterGraph, self).__init__()
        self.DCP = DGCNN_Centerline_Graph(
            k, dropout, output_channels, n_features, batch_size,
            hyper_parameters, emb_dims, feature_transform=False, channel=3
        )
    
    def forward(self, input_bo, input_ob, centerline_bo, centerline_ob):
        """Assemble tensors for boundary/observation pairs and return velocity/pressure."""
        # Boundary inputs
        x_b, y_b, z_b, ti_b, b1i_b, b2i_b, b3i_b, td_b, b1d_b, b2d_b, b3d_b, t_b, mind_b = input_bo
        pv_b = torch.cat([x_b, y_b, z_b, ti_b, b1i_b, b2i_b, b3i_b, td_b, b1d_b, b2d_b, b3d_b, mind_b], -1)
        
        # Observation inputs
        x_o, y_o, z_o, ti_o, b1i_o, b2i_o, b3i_o, td_o, b1d_o, b2d_o, b3d_o, t_o = input_ob
        pv_o = torch.cat([x_o, y_o, z_o, ti_o, b1i_o, b2i_o, b3i_o, td_o, b1d_o, b2d_o, b3d_o], -1)
        
        # Centerline stacks
        x_c_b, y_c_b, z_c_b, cla_b, d_b = centerline_bo
        c_b = torch.cat([x_c_b, y_c_b, z_c_b, cla_b, d_b], -1)
        
        x_c_o, y_c_o, z_c_o, cla_o, d_o = centerline_ob
        c_o = torch.cat([x_c_o, y_c_o, z_c_o, cla_o, d_o], -1)
        
        logits = self.DCP(pv_o, pv_b, c_o, c_b)
        
        u, v, w, p = logits[:, :, 0:1], logits[:, :, 1:2], logits[:, :, 2:3], logits[:, :, 3:4]
        return u, v, w, p


class CarotidArtery_CenterGraph(nn.Module):
    """Carotid-friendly wrapper with reduced feature dimensions."""
    
    def __init__(self, k, dropout, output_channels, n_features, batch_size,
                 hyper_parameters, emb_dims):
        super(CarotidArtery_CenterGraph, self).__init__()
        self.DCP = CarotidArtery_Centerline_Graph(
            k, dropout, output_channels, n_features, batch_size,
            hyper_parameters, emb_dims, feature_transform=False, channel=3
        )
    
    def forward(self, input_bo, input_ob, centerline_bo, centerline_ob):
        """Package carotid boundary/observation tensors for downstream prediction."""
        # Boundary inputs
        x_b, y_b, z_b, ti_b, b1i_b, b2i_b, td_b, b1d_b, b2d_b, mind_b = input_bo
        pv_b = torch.cat([x_b, y_b, z_b, ti_b, b1i_b, b2i_b, td_b, b1d_b, b2d_b, mind_b], -1)
        
        # Observation inputs
        x_o, y_o, z_o, ti_o, b1i_o, b2i_o, td_o, b1d_o, b2d_o = input_ob
        pv_o = torch.cat([x_o, y_o, z_o, ti_o, b1i_o, b2i_o, td_o, b1d_o, b2d_o], -1)
        
        # Centerline stacks
        x_c_b, y_c_b, z_c_b, cla_b, d_b = centerline_bo
        c_b = torch.cat([x_c_b, y_c_b, z_c_b, cla_b, d_b], -1)
        
        x_c_o, y_c_o, z_c_o, cla_o, d_o = centerline_ob
        c_o = torch.cat([x_c_o, y_c_o, z_c_o, cla_o, d_o], -1)
        
        logits = self.DCP(pv_o, pv_b, c_o, c_b)
        
        u, v, w, p = logits[:, :, 0:1], logits[:, :, 1:2], logits[:, :, 2:3], logits[:, :, 3:4]
        return u, v, w, p

