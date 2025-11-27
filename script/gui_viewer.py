import polyscope as ps
import torch
import numpy as np
import trimesh

from pprnet.data.IPA_pose_dataset import IPAPoseDataset
from pprnet import ROOT_DIR
from torchvision import transforms
DATASET_DIR = f'E:\\h5_dataset\\bunny\\'
from pprnet.data.pointcloud_transforms import PointCloudShuffle, ToTensor

from pprnet.pprnet import PPRNet, load_checkpoint, save_checkpoint
from pprnet.object_type import ObjectType
from sklearn.cluster import MeanShift
from pprnet.utils import eulerangles
import warp as wp
import os

NUM_POINT = 2**14
transforms = transforms.Compose(
    [
        PointCloudShuffle(NUM_POINT),
        ToTensor()
    ]
)

test_dataset = IPAPoseDataset(DATASET_DIR, [245, 246], [3, 81], transforms=transforms)
data = test_dataset[50]
pcl = data["point_clouds"].cpu().numpy()
ps.init()
wp.init()

ps.set_front_dir("neg_y_front")
ps.set_up_dir("neg_z_up")

type_bunny = ObjectType(type_name='bunny', class_idx=0, symmetry_type='finite',
                        lambda_p=[[0.0263663, 0.0, 0.0], [0.0, 0.0338224, 0.0], [-0.0, 0.0, 0.0484393]],
                        G=[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])

backbone_config = {
    'npoint_per_layer': [4096, 1024, 256, 64],
    'radius_per_layer': [[10, 20, 30], [30, 45, 60], [60, 80, 120], [120, 160, 240]]
}
loss_weights = {
    'trans_head': 200.0 / 1000.0,  # implicit convert mm to m by deviding 1000
    'rot_head': 200.0,
    'vis_head': 50.0
}
PROJECT_NAME = "ppr"
LOG_NAME = 'log0_batch8_scale3_test_log'
log_dir = os.path.join(ROOT_DIR, 'logs', PROJECT_NAME, LOG_NAME)

net = PPRNet(type_bunny, backbone_config, True, loss_weights, True)
load_checkpoint(log_dir + "/checkpoint.tar", net)
device = 'cuda'
net = net.to(device)
labels = {
    'rot_label': data['rot_label'].to(device).view(1, NUM_POINT, 3, 3),
    'trans_label': data['trans_label'].to(device).view(1, NUM_POINT, 3),
    'vis_label': data['vis_label'].to(device).view(1, NUM_POINT)
}
inputs = {
    'point_clouds': data['point_clouds'].to(device).view(1, NUM_POINT, 3),
    'labels': labels
}

with torch.no_grad():
    pred_results = net.forward(inputs)[0]

pred_trans_val = pred_results[0][0].cpu().numpy()
pred_mat_val = pred_results[1][0].cpu().numpy()
pred_vis_val = pred_results[2][0].cpu().numpy()

# remove low vis points
vs_picked_idx = pred_vis_val > 0.45
input_point = pcl[vs_picked_idx]
pred_trans_val = pred_trans_val[vs_picked_idx]
pred_mat_val = pred_mat_val[vs_picked_idx]

# cluster
ms = MeanShift(bandwidth=40, bin_seeding=True, cluster_all=False, min_bin_freq=40)
ms.fit(pred_trans_val)
labels = ms.labels_
cluster_centers = ms.cluster_centers_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

color_cluster = [np.array([np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)]) for i in
                 range(n_clusters)]
color_per_point = np.ones([pred_trans_val.shape[0], pred_trans_val.shape[1]])
for idx in range(color_per_point.shape[0]):
    if labels[idx] != -1:
        color_per_point[idx, :] = color_cluster[labels[idx]] / 255.0

pred_trans_cluster = [[] for _ in range(n_clusters)]
pred_mat_cluster = [[] for _ in range(n_clusters)]
for idx in range(pred_trans_val.shape[0]):
    if labels[idx] != -1:
        pred_trans_cluster[labels[idx]].append(np.reshape(pred_trans_val[idx], [1, 3]))
        pred_mat_cluster[labels[idx]].append(np.reshape(pred_mat_val[idx], [1, 3, 3]))
pred_trans_cluster = [np.concatenate(cluster, axis=0) for cluster in pred_trans_cluster]
pred_mat_cluster = [np.concatenate(cluster, axis=0) for cluster in pred_mat_cluster]

cluster_center_pred = [np.mean(cluster, axis=0) for cluster in pred_trans_cluster]

cluster_mat_pred = []
for mat_cluster in pred_mat_cluster:
    all_quat = np.zeros([mat_cluster.shape[0], 4])
    for idx in range(mat_cluster.shape[0]):
        all_quat[idx] = eulerangles.mat2quat(mat_cluster[idx])
    quat = eulerangles.average_quat(all_quat)
    cluster_mat_pred.append(eulerangles.quat2mat(quat))

pcl = ps.register_point_cloud(name = "centroid_points", points=input_point / 1000.0)
pcl.add_color_quantity("cluster", color_per_point, enabled = True)



bunny_mesh = trimesh.load(f"{ROOT_DIR}/models/SileaneBunny.obj")

group = ps.create_group("bunny")
for cluster_id in range(n_clusters):
    rot_mat = cluster_mat_pred[cluster_id]
    center = cluster_center_pred[cluster_id]
    T = np.eye(4)
    T[:3, :3] = rot_mat
    T[:3, 3] = center / 1000.0

    copy_bunny = bunny_mesh.copy()
    copy_bunny.apply_transform(T)
    obj = ps.register_surface_mesh(f"bunny {cluster_id}", copy_bunny.vertices, copy_bunny.faces, color = color_cluster[cluster_id] / 255.0)
    obj.add_to_group(group)
group.set_hide_descendants_from_structure_lists(True)
group.set_show_child_details(False)
ps.show()