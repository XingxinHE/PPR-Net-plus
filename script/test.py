import warp as wp
import numpy as np
import torch
wp.init()
from time import perf_counter
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

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def validate(radius, xyz, new_xyz, group_idx):
    check = 1
    for b in range(B):
        for i in range(m):
            diff = new_xyz[b, i, :] - xyz[b, group_idx[b, i, :], :]
            check = check * torch.prod(torch.linalg.norm(diff, dim=-1) < radius)
    return check >= 1

B = 32
n = 2**14
m = int(n / 4)
nsample = 32
radius = 0.1
xyz = torch.rand((B, n, 3), dtype=torch.float32, device = 'cuda')
new_xyz = xyz[:, :m, :].clone()

time = perf_counter()
result = query_ball_point_warp(0.1, nsample=nsample, xyz=xyz, new_xyz=new_xyz)
print((perf_counter() - time))
print(validate(radius, xyz, new_xyz, result))

time = perf_counter()
result = query_ball_point(0.1, nsample=nsample, xyz=xyz, new_xyz=new_xyz)
print((perf_counter() - time))

print(validate(radius, xyz, new_xyz, result))
