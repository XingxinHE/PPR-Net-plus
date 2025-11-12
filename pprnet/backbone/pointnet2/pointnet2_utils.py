import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
import warp as wp
import fpsample

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def farthest_point_sample_cpu(xyz, npoint):
    pc = xyz.cpu().numpy()
    B, _, _ = xyz.shape
    new_xyz = []
    for b in range(B):
        fps_samples_idx = fpsample.bucket_fps_kdtree_sampling(pc[b, :], npoint)
        fps_samples_idx = torch.from_numpy(fps_samples_idx).type(torch.long).to(xyz.device)
        new_xyz.append(torch.index_select(xyz[b, :, :], 0, fps_samples_idx))
    new_xyz = torch.stack(new_xyz)
    return new_xyz

@wp.kernel
def query_ball_point_warp_kernel(grids: wp.array(dtype=wp.uint64),
                                 points: wp.array2d(dtype=wp.vec3),
                                 new_points: wp.array2d(dtype=wp.vec3),
                                 group_idx: wp.array2d(dtype=wp.int32),
                                 radius: float,
                                 nsample: int):

    batch, tid = wp.tid()

    # query point
    p = new_points[batch, tid]

    # create grid query around point
    query = wp.hash_grid_query(grids[batch], p, radius)
    index = int(0)
    isample = int(0)
    while (wp.hash_grid_query_next(query, index)):

        neighbor = points[batch, index]

        # compute distance to neighbor point
        dist = wp.length(p - neighbor)
        if (dist <= radius):
            group_idx[batch, tid * nsample + isample] = index
            isample = isample + 1

        if isample >= nsample:
            break

def query_ball_point_warp(radius, nsample, xyz, new_xyz):
    device = xyz.device
    device_str = str(device)
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape

    xyz_wp = wp.from_torch(xyz, dtype=wp.vec3)
    new_xyz_wp = wp.from_torch(new_xyz, dtype=wp.vec3)
    grids = []
    grids_id = []
    for b in range(B):
        grid = wp.HashGrid(dim_x=128, dim_y=128, dim_z=128, device=device_str)
        xyz_b_wp = wp.from_torch(xyz[b, :], dtype=wp.vec3)
        grid.build(xyz_b_wp, radius)
        grids.append(grid)
        grids_id.append(grid.id)

    group_idx = torch.ones(B, S * nsample, dtype=torch.int32).to(device) * N
    group_idx = wp.from_torch(group_idx, dtype=wp.int32)
    grids_id = wp.array(grids_id, dtype=wp.uint64).to(device_str)

    wp.launch(query_ball_point_warp_kernel, dim = (B, S),
              inputs = [grids_id, xyz_wp, new_xyz_wp, group_idx, radius, nsample], device=device_str)
    group_idx = wp.to_torch(group_idx).view(B, S, nsample)
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, N, C]
            points: input points data, [B, N, D]
        Return:
            new_xyz: sampled points position data, [B, S, C]
            new_points_concat: sample points feature data, [B, S, D']
        """
        start = time()
        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = farthest_point_sample_cpu(xyz, S)
        #timeit("farthest_point_sample", start)
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point_warp(radius, K, xyz, new_xyz)
            #timeit("query_ball_point_warp", start)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_points_concat = torch.cat(new_points_list, dim=1).permute(0, 2, 1)
        #timeit("forward", start)

        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        new_points = new_points.permute(0, 2, 1)

        return new_points
