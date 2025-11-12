import torch.nn as nn
import torch
import torch.nn.functional as F
from .pointnet2_utils import PointNetSetAbstractionMsg,PointNetFeaturePropagation

class Pointnet2MSGBackbone(nn.Module):
    r"""
        PointNet2 backbone for pointwise feature extraction( multi-scale grouping).

        Parameters
        ----------
    """

    def __init__(self, npoint_per_layer, radius_per_layer, input_feature_dims=0, use_xyz=True):
        super(Pointnet2MSGBackbone, self).__init__()
        assert len(npoint_per_layer) == len(radius_per_layer) == 4
        self.nscale = len(radius_per_layer[0])

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointNetSetAbstractionMsg(
                npoint=npoint_per_layer[0],
                radius_list=radius_per_layer[0],
                nsample_list=[32] * self.nscale,
                in_channel=input_feature_dims,
                mlp_list=[[32, 32, 64] for _ in range(self.nscale)],
            )
        )
        c_out_0 = 64*self.nscale
        c_in = c_out_0
        self.SA_modules.append(
            PointNetSetAbstractionMsg(
                npoint=npoint_per_layer[1],
                radius_list=radius_per_layer[1],
                nsample_list=[32] * self.nscale,
                in_channel=c_in,
                mlp_list=[[64, 64, 128] for _ in range(self.nscale)],
            )
        )
        c_out_1 = 128*self.nscale
        c_in = c_out_1
        self.SA_modules.append(
            PointNetSetAbstractionMsg(
                npoint=npoint_per_layer[2],
                radius_list=radius_per_layer[2],
                nsample_list=[32] * self.nscale,
                in_channel=c_in,
                mlp_list=[[128, 128, 256] for _ in range(self.nscale)],
            )
        )
        c_out_2 = 256*self.nscale
        c_in = c_out_2
        self.SA_modules.append(
            PointNetSetAbstractionMsg(
                npoint=npoint_per_layer[3],
                radius_list=radius_per_layer[3],
                nsample_list=[32] * self.nscale,
                in_channel=c_in,
                mlp_list=[[256, 256, 512] for _ in range(self.nscale)],
            )
        )
        c_out_3 = 512*self.nscale

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointNetFeaturePropagation(in_channel = 128 + input_feature_dims, mlp=[128, 128, 128]))
        self.FP_modules.append(PointNetFeaturePropagation(in_channel = 256 + c_out_0, mlp=[256, 128]))
        self.FP_modules.append(PointNetFeaturePropagation(in_channel = 512 + c_out_1, mlp=[256, 256]))
        self.FP_modules.append(PointNetFeaturePropagation(in_channel = c_out_3 + c_out_2, mlp=[512, 512]))


    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud):
        # type: (Pointnet2MSG, torch.cuda.FloatTensor) -> pt_utils.Seq
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: torch.cuda.FloatTensor
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)

            Returns:
            ----------
            new_features : torch.Tensor
                (B, 128, N) tensor. Pointwise feature.
        """
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return l_features[0]