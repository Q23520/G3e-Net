from typing import Sequence

import torch
from torch import nn

from utils.PointNetUtils import STN3d
from utils.G3e_Net_utils import get_graph_feature


class EdgeConvStack(nn.Module):
    """
    Multi-layer edge-convolution stack that mirrors the repeated pattern
    used in the original PINet implementation.
    """

    def __init__(
        self,
        in_channels: int,
        layer_channels: Sequence[int],
        k: int,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.k = k
        self.layers = nn.ModuleList()
        prev_channels = in_channels

        for out_channels in layer_channels:
            block = nn.Sequential(
                nn.Conv2d(prev_channels * 2, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(negative_slope=negative_slope),
            )
            self.layers.append(block)
            prev_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor with shape (B, C, N)

        Returns:
            Concatenated features from each edge-conv stage, shape (B, sum(layer_channels), N)
        """
        feats = []
        current = x
        for layer in self.layers:
            edge_feature = get_graph_feature(current, k=self.k)
            current = layer(edge_feature)
            feats.append(current.max(dim=-1, keepdim=False)[0])
        return torch.cat(feats, dim=1)


class DGCNN_Cline(nn.Module):
    """
    Dual-branch dynamic graph CNN used as the main backbone in nn.py.
    """

    def __init__(
        self,
        k: int,
        dropout: float,
        output_channels: int,
        emb_dims: int = 4096,
        feature_transform: bool = False,
        channel: int = 3,
        obs_channels: int = 4,
        boundary_channels: int = 5,
        activation_slope: float = 0.2,
    ) -> None:
        super().__init__()
        del feature_transform  # kept for API compatibility, not used in the original impl

        self.stn = STN3d(channel)
        self.obs_stack = EdgeConvStack(obs_channels, (64, 128, 256), k, activation_slope)
        self.boundary_stack = EdgeConvStack(boundary_channels, (64, 64, 128), k, activation_slope)

        self.obs_projection = nn.Sequential(
            nn.Conv1d(64 + 128 + 256, emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(emb_dims),
            nn.LeakyReLU(negative_slope=activation_slope),
        )

        self.boundary_projection = nn.Sequential(
            nn.Conv1d(64 + 64 + 128, emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(emb_dims),
            nn.LeakyReLU(negative_slope=activation_slope),
        )

        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.prediction_head = nn.Sequential(
            nn.Conv1d(emb_dims * 2, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=activation_slope),
            nn.Dropout(p=dropout),
            nn.Conv1d(1024, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=activation_slope),
            nn.Dropout(p=dropout),
            nn.Conv1d(512, output_channels, kernel_size=1),
        )

    def forward(self, observation: torch.Tensor, boundary: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observation: observation graph input of shape (B, C_obs, N_obs)
            boundary: boundary graph input of shape (B, C_bnd, N_bnd)
        """
        obs_aligned = self._apply_tnet(observation)
        obs_features = self.obs_stack(obs_aligned)
        obs_features = self.obs_projection(obs_features)

        boundary_aligned = self._apply_tnet(boundary)
        boundary_features = self.boundary_stack(boundary_aligned)
        boundary_features = self.boundary_projection(boundary_features)

        boundary_global = self.global_pool(boundary_features)
        boundary_tiled = boundary_global.expand(-1, -1, obs_features.size(-1))

        fused = torch.cat((obs_features, boundary_tiled), dim=1)
        return self.prediction_head(fused)

    def _apply_tnet(self, features: torch.Tensor) -> torch.Tensor:
        """
        Align the first three spatial channels with STN while keeping the rest intact.
        """
        trans = self.stn(features[:, :3, :])
        permuted = features.permute(0, 2, 1)
        spatial = permuted[:, :, :3]
        residual = permuted[:, :, 3:] if permuted.size(-1) > 3 else None

        aligned = torch.bmm(spatial, trans)
        if residual is not None:
            aligned = torch.cat((aligned, residual), dim=2)
        return aligned.permute(0, 2, 1)

