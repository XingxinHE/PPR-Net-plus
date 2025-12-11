import polyscope as ps
import torch
import numpy as np
import trimesh
import os
import sys

# Add root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pprnet.data.IPA_pose_dataset import IPAPoseDataset
from pprnet import ROOT_DIR
from torchvision import transforms
from pprnet.data.pointcloud_transforms import PointCloudShuffle, ToTensor
from pprnet.pprnet import PPRNet, load_checkpoint
from pprnet.object_type import ObjectType
from sklearn.cluster import MeanShift
from pprnet.utils import eulerangles
import warp as wp

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATASET_DIR = os.path.join(ROOT_DIR, 'dataset', 'teris', 'h5')
# Use the test range from train_teris.py
TEST_CYCLE_RANGE = [121, 150]
TEST_SCENE_RANGE = [1, 8]

NUM_POINT = 2**14 # 16384

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
transforms = transforms.Compose([
    PointCloudShuffle(NUM_POINT),
    ToTensor()
])

print(f"Loading dataset from {DATASET_DIR}...")
test_dataset = IPAPoseDataset(DATASET_DIR, TEST_CYCLE_RANGE, TEST_SCENE_RANGE, transforms=transforms)
print(f"Dataset loaded. Size: {len(test_dataset)}")

# Pick a sample
SAMPLE_IDX = 5 # Change this to see different samples
data = test_dataset[SAMPLE_IDX]

pcl = data["point_clouds"].cpu().numpy() # (N, 3)

# -----------------------------------------------------------------------------
# Model Loading
# -----------------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define Object Type (Teris)
# Using the symmetry definition from test_capture_zivid.py
G_sym = [
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]], # Identity
    #[[-1, 0, 0], [0, 1, 0], [0, 0, -1]] # 180 deg around Y (Flip)
]
type_teris = ObjectType(type_name='teris', class_idx=0, symmetry_type='finite',
                        lambda_p=[[0.039965, 0.0, 0.0], [0.0, 0.028565, 0.0], [0.0, 0.0, 0.018634]],
                        G=G_sym)

backbone_config = {
    'npoint_per_layer': [4096, 1024, 256, 64],
    'radius_per_layer': [[10, 20, 30], [30, 45, 60], [60, 80, 120], [120, 160, 240]]
}
loss_weights = {'trans_head': 1.0, 'rot_head': 1.0, 'vis_head': 1.0}

net = PPRNet(type_teris, backbone_config, True, loss_weights, True)
net.to(device)

# Load Checkpoint
CHECKPOINT_PATH = os.path.join(ROOT_DIR, 'logs', "ppr_teris", "log_teris_test", "checkpoint.tar")
if os.path.exists(CHECKPOINT_PATH):
    load_checkpoint(CHECKPOINT_PATH, net)
    print(f"Loaded model from {CHECKPOINT_PATH}")
else:
    print(f"Checkpoint not found at {CHECKPOINT_PATH}")
    sys.exit(1)

net.eval()

# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------
print("Running inference...")
labels = {
    'rot_label': data['rot_label'].to(device).view(1, NUM_POINT, 3, 3),
    'trans_label': data['trans_label'].to(device).view(1, NUM_POINT, 3),
    'vis_label': data['vis_label'].to(device).view(1, NUM_POINT)
}
inputs = {
    'point_clouds': data['point_clouds'].to(device).view(1, NUM_POINT, 3),
    'labels': labels
}

wp.init()
with torch.no_grad():
    pred_results = net.forward(inputs)[0]

pred_trans_val = pred_results[0][0].cpu().numpy() # (N, 3)
pred_mat_val = pred_results[1][0].cpu().numpy()   # (N, 3, 3)
pred_vis_val = pred_results[2][0].cpu().numpy()   # (N,)

# -----------------------------------------------------------------------------
# Post-processing (Clustering)
# -----------------------------------------------------------------------------

# remove low vis points
vs_picked_idx = pred_vis_val > 0.45
input_point = pcl[vs_picked_idx]
pred_trans_val = pred_trans_val[vs_picked_idx]
pred_mat_val = pred_mat_val[vs_picked_idx]

if len(pred_trans_val) == 0:
    print("No points with high visibility found.")
else:
    # Position Clustering
    ms = MeanShift(bandwidth=40, bin_seeding=True, cluster_all=False, min_bin_freq=40)
    ms.fit(pred_trans_val)
    labels = ms.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Found {n_clusters} clusters.")

    color_cluster = [np.array([np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)]) for i in
                     range(n_clusters)]
    color_per_point = np.ones([pred_trans_val.shape[0], 3])
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

    # Rotation Clustering (The new logic)
    cluster_mat_pred = []
    for mat_cluster in pred_mat_cluster:
        all_quat = np.zeros([mat_cluster.shape[0], 4])
        for idx in range(mat_cluster.shape[0]):
            q = eulerangles.mat2quat(mat_cluster[idx])
            # Canonicalize quaternion
            if q[0] < 0:
                q = -q
            all_quat[idx] = q

        # MeanShift on quaternions
        ms_rot = MeanShift(bandwidth=0.2, bin_seeding=True, cluster_all=False)
        ms_rot.fit(all_quat)

        rot_labels = ms_rot.labels_
        unique_labels = set(rot_labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)

        if len(unique_labels) > 0:
            counts = [np.sum(rot_labels == l) for l in unique_labels]
            best_label = list(unique_labels)[np.argmax(counts)]
            best_quats = all_quat[rot_labels == best_label]
            quat = eulerangles.average_quat(best_quats)
            print(f"Rotation clustering: Found {len(unique_labels)} modes. Using mode with {len(best_quats)}/{len(all_quat)} votes.")
        else:
            print("Rotation clustering failed. Falling back to global average.")
            quat = eulerangles.average_quat(all_quat)

        cluster_mat_pred.append(eulerangles.quat2mat(quat))

    # -----------------------------------------------------------------------------
    # Visualization
    # -----------------------------------------------------------------------------
    ps.init()
    ps.set_front_dir("neg_y_front")
    ps.set_up_dir("neg_z_up")

    def visualize_frame(name, t, R, length=0.05):
        p_x = t + R[:, 0] * length
        ps.register_curve_network(f"{name}_x", np.array([t, p_x]), np.array([[0, 1]]), color=np.array([1, 0, 0]), radius=0.002)
        p_y = t + R[:, 1] * length
        ps.register_curve_network(f"{name}_y", np.array([t, p_y]), np.array([[0, 1]]), color=np.array([0, 1, 0]), radius=0.002)
        p_z = t + R[:, 2] * length
        ps.register_curve_network(f"{name}_z", np.array([t, p_z]), np.array([[0, 1]]), color=np.array([0, 0, 1]), radius=0.002)

    # Register input point cloud
    # Note: Dataset is in mm, but gui_viewer divides by 1000.0 for visualization in meters
    pcl_vis = ps.register_point_cloud(name="centroid_points", points=input_point / 1000.0)
    pcl_vis.add_color_quantity("cluster", color_per_point, enabled=True)

    mesh_path = os.path.join(ROOT_DIR, 'models', 'T.obj')
    if os.path.exists(mesh_path):
        teris_mesh = trimesh.load(mesh_path)
        group = ps.create_group("teris")

        for cluster_id in range(n_clusters):
            rot_mat = cluster_mat_pred[cluster_id]
            center = cluster_center_pred[cluster_id]

            T = np.eye(4)
            T[:3, :3] = rot_mat
            T[:3, 3] = center / 1000.0

            copy_teris = teris_mesh.copy()
            copy_teris.apply_transform(T)

            obj = ps.register_surface_mesh(f"teris {cluster_id}", copy_teris.vertices, copy_teris.faces, color=color_cluster[cluster_id] / 255.0)
            obj.add_to_group(group)

            visualize_frame(f"frame_{cluster_id}", center / 1000.0, rot_mat)

        group.set_hide_descendants_from_structure_lists(True)
        group.set_show_child_details(False)
    else:
        print(f"Mesh not found at {mesh_path}")

    ps.show()
