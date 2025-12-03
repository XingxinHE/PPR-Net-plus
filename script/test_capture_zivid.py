import rtde_receive
import rtde_control
import math
import time
import numpy as np
import torch
import polyscope as ps
import trimesh
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pprnet.pprnet import PPRNet, load_checkpoint, save_checkpoint

from pprnet.pprnet import PPRNet
from pprnet.object_type import ObjectType
from pprnet import ROOT_DIR
import warp as wp

def transform_to_ur(rtde_r, points_sampled):
    # scale from mm to m
    points_sampled = points_sampled / 1000.0


    # Get TCP pose (x, y, z, rx, ry, rz) [m, rad]
    tcp_pose = rtde_r.getActualTCPPose()

    # Convert to 4x4 Matrix
    t_base_ee = np.eye(4)
    t_base_ee[:3, 3] = tcp_pose[:3]

    # Convert axis-angle (rx, ry, rz) to rotation matrix
    r_vec = np.array(tcp_pose[3:])
    theta = np.linalg.norm(r_vec)
    if theta > 1e-6:
        k = r_vec / theta
        K = np.array([
            [0, -k[2], k[1]],
            [k[2], 0, -k[0]],
            [-k[1], k[0], 0]
        ])
        R_base_ee = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    else:
        R_base_ee = np.eye(3)

    t_base_ee[:3, :3] = R_base_ee

    print("TCP Pose (Base -> EE):")
    print(t_base_ee)


    # Hand-Eye Calibration (EE -> Camera)
    # TODO: Replace with actual calibration
    # Placeholder: Identity (Camera is at EE origin)
    t_ee_cam = np.eye(4)

    # Base -> Camera
    t_base_cam = t_base_ee @ t_ee_cam

    # Transform Point Cloud to Base Frame
    # points (N, 3) -> (N, 4)
    points_hom = np.hstack((points_sampled, np.ones((len(points_sampled), 1))))
    points_base = (t_base_cam @ points_hom.T).T[:, :3]

    return points_base, t_base_cam

def transform_to_camera(points_base, t_base_cam):
    t_cam_base = np.linalg.inv(t_base_cam)
    points_hom = np.hstack((points_base, np.ones((len(points_base), 1))))
    points_cam = (t_cam_base @ points_hom.T).T[:, :3]
    return points_cam


def drop_points_below_plane(points, plane_z=0.0):
    # points: (N, 3)
    mask = points[:, 2] >= plane_z
    return points[mask]



# -----------------------------------------------------------------------------
# 1. Move UR Robot
# -----------------------------------------------------------------------------
ROBOT_IP = "172.16.0.7"
print(f"Connecting to robot at {ROBOT_IP}...")

try:
    rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
    print("Connected to UR robot.")

    # Target joint values in degrees
    target_joints_deg = [143.98, -86.56, -90.38, -100.50, 99.73, 54.82]
    # Convert to radians
    target_joints_rad = [math.radians(d) for d in target_joints_deg]

    print(f"Moving to joint positions (rad): {target_joints_rad}")
    # moveJ(q, speed, acceleration, asynchronous)
    # Using moderate speed and acceleration
    success = rtde_c.moveJ(target_joints_rad, 0.5, 0.3)

    if success:
        print("Movement completed.")
    else:
        print("Movement failed or interrupted.")

except Exception as e:
    print(f"Robot connection/movement failed: {e}")
    # Continue for testing other parts if robot is not available (e.g. in simulation/dev)
    # In production, you might want to raise e

# -----------------------------------------------------------------------------
# 2. Capture Point Cloud from Zivid
# -----------------------------------------------------------------------------
import zivid

print("Initializing Zivid...")
try:
    app = zivid.Application()
    camera = app.connect_camera()

    settings_path = "./zivid_settings.yml"
    if not os.path.exists(settings_path):
        print(f"Warning: {settings_path} not found. Using default settings.")
        settings = zivid.Settings()
        settings.acquisitions.append(zivid.Settings.Acquisition())
    else:
        settings = zivid.Settings.load(settings_path)

    print("Capturing frame...")
    with camera.capture_2d_3d(settings) as frame:
        point_cloud = frame.point_cloud()

        # Get XYZ data
        points = point_cloud.copy_data("xyz").reshape((-1, 3))

        # Filter NaNs
        flag = np.logical_not(np.any(np.isnan(points), axis=1))
        points = points[flag, :]

        # Convert to mm if Zivid returns meters (Zivid usually returns mm, but let's verify)
        # Assuming Zivid returns mm based on standard usage, but if it's meters, multiply by 1000.
        # Usually Zivid SDK returns XYZ in millimeters.

        print(f"Captured {len(points)} points.")

except Exception as e:
    print(f"Zivid capture failed: {e}")
    raise e


# -----------------------------------------------------------------------------
# 4. Load PPR-Net and Inference
# -----------------------------------------------------------------------------
print("Loading PPR-Net...")

# Preprocessing: Downsample to 16384 points
NUM_POINT = 16384
if len(points) >= NUM_POINT:
    choice = np.random.choice(len(points), NUM_POINT, replace=False)
else:
    choice = np.random.choice(len(points), NUM_POINT, replace=True)
points_sampled = points[choice, :]

# Transform to Robot Base Frame
points_sampled, t_base_cam = transform_to_ur(rtde_r, points_sampled)
# Drop points below table
points_sampled = drop_points_below_plane(points_sampled, plane_z=0.0)


# Transform back to Camera Frame
points_sampled = transform_to_camera(points_sampled, t_base_cam)

# -----------------------------------------------------------------------------
# Coordinate System Correction
# -----------------------------------------------------------------------------
# The network was trained on data in a custom coordinate frame:
# X: Left, Y: Up, Z: Forward
# The Zivid camera (and standard CV) uses:
# X: Right, Y: Down, Z: Forward
# We need to flip X and Y to match the training distribution.
points_sampled[:, 0] = -points_sampled[:, 0]
points_sampled[:, 1] = -points_sampled[:, 1]


# Resample again after filtering if needed, or pad
if len(points_sampled) < NUM_POINT:
    choice = np.random.choice(len(points_sampled), NUM_POINT, replace=True)
    points_sampled = points_sampled[choice, :]
elif len(points_sampled) > NUM_POINT:
    choice = np.random.choice(len(points_sampled), NUM_POINT, replace=False)
    points_sampled = points_sampled[choice, :]



# Prepare input tensor
# Network expects (B, N, 3)
# Note: The network was trained on data in mm (or scaled units).
# If points_sampled is in meters (from transform_to_ur), we might need to scale it back to mm for the network?
# The training script says: 'trans_head': 200.0 / 1000.0, # implicit convert mm to m by deviding 1000
# And IPAPoseDataset scales data by 1000.0 (mm).
# So the network expects input in mm.
points_sampled_mm = points_sampled * 1000.0

input_points = torch.from_numpy(points_sampled_mm).float().unsqueeze(0) # (1, 16384, 3)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_points = input_points.to(device)

# Define Model
type_teris = ObjectType(type_name='teris', class_idx=0, symmetry_type='finite',
                        lambda_p=[[0.039965, 0.0, 0.0], [0.0, 0.028565, 0.0], [0.0, 0.0, 0.018634]],
                        G=[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])

backbone_config = {
    'npoint_per_layer': [4096, 1024, 256, 64],
    'radius_per_layer': [[10, 20, 30], [30, 45, 60], [60, 80, 120], [120, 160, 240]]
}

# Initialize Warp
wp.init()

# Dummy weights for initialization
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

net.eval()

# Inference
print("Running inference...")
with torch.no_grad():
    dummy_labels = {
        'rot_label': torch.zeros(1, NUM_POINT, 3, 3).to(device),
        'trans_label': torch.zeros(1, NUM_POINT, 3).to(device),
        'vis_label': torch.zeros(1, NUM_POINT).to(device)
    }

    inputs = {
        'point_clouds': input_points,
        'labels': dummy_labels
    }

    pred_results = net.forward(inputs)[0]

    pred_trans_val = pred_results[0][0].cpu().numpy() # (N, 3)
    pred_mat_val = pred_results[1][0].cpu().numpy()   # (N, 3, 3)
    pred_vis_val = pred_results[2][0].cpu().numpy()   # (N,)

    # -----------------------------------------------------------------------------
    # Coordinate System Correction (Output)
    # -----------------------------------------------------------------------------
    # Convert back from Network Frame (X-Left, Y-Up) to Standard Frame (X-Right, Y-Down)

    # Translation: Flip X and Y
    pred_trans_val[:, 0] = -pred_trans_val[:, 0]
    pred_trans_val[:, 1] = -pred_trans_val[:, 1]

    # Rotation: R_std = T_flip @ R_net
    T_flip = np.diag([-1.0, -1.0, 1.0])
    # Use broadcasting: (3, 3) @ (N, 3, 3) -> (N, 3, 3)
    pred_mat_val = np.matmul(T_flip, pred_mat_val)

    # Input points for visualization also need to be flipped back
    points_sampled_mm[:, 0] = -points_sampled_mm[:, 0]
    points_sampled_mm[:, 1] = -points_sampled_mm[:, 1]

# -----------------------------------------------------------------------------
# 5. Post-processing and Visualization
# -----------------------------------------------------------------------------
from sklearn.cluster import MeanShift
from pprnet.utils import eulerangles

# remove low vis points
vs_picked_idx = pred_vis_val > 0.45
input_point = points_sampled_mm[vs_picked_idx]
pred_trans_val = pred_trans_val[vs_picked_idx]
pred_mat_val = pred_mat_val[vs_picked_idx]

if len(pred_trans_val) == 0:
    print("No points with high visibility found.")
else:
    # cluster
    # Bandwidth might need adjustment depending on the scale (mm vs m) and object size
    ms = MeanShift(bandwidth=40, bin_seeding=True, cluster_all=False, min_bin_freq=40)
    ms.fit(pred_trans_val)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
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

    cluster_mat_pred = []
    for mat_cluster in pred_mat_cluster:
        all_quat = np.zeros([mat_cluster.shape[0], 4])
        for idx in range(mat_cluster.shape[0]):
            all_quat[idx] = eulerangles.mat2quat(mat_cluster[idx])
        quat = eulerangles.average_quat(all_quat)
        cluster_mat_pred.append(eulerangles.quat2mat(quat))

    # Visualization
    ps.init()
    ps.set_front_dir("neg_y_front")
    ps.set_up_dir("neg_z_up")

    def visualize_frame(name, t, R, length=0.05):
        # t: (3,)
        # R: (3, 3)
        # X axis (Red)
        p_x = t + R[:, 0] * length
        ps.register_curve_network(f"{name}_x", np.array([t, p_x]), np.array([[0, 1]]), color=np.array([1, 0, 0]), radius=0.002)
        # Y axis (Green)
        p_y = t + R[:, 1] * length
        ps.register_curve_network(f"{name}_y", np.array([t, p_y]), np.array([[0, 1]]), color=np.array([0, 1, 0]), radius=0.002)
        # Z axis (Blue)
        p_z = t + R[:, 2] * length
        ps.register_curve_network(f"{name}_z", np.array([t, p_z]), np.array([[0, 1]]), color=np.array([0, 0, 1]), radius=0.002)

    # Register input point cloud (in meters for visualization consistency with robot frame?)
    # The gui_viewer uses / 1000.0, so it visualizes in meters.
    pcl = ps.register_point_cloud(name = "centroid_points", points=input_point / 1000.0)
    pcl.add_color_quantity("cluster", color_per_point, enabled = True)

    mesh_path = os.path.join(ROOT_DIR, 'models', 'T.obj')
    if os.path.exists(mesh_path):
        teris_mesh = trimesh.load(mesh_path)
        group = ps.create_group("teris")

        for cluster_id in range(n_clusters):
            rot_mat = cluster_mat_pred[cluster_id]
            center = cluster_center_pred[cluster_id]

            T = np.eye(4)
            T[:3, :3] = rot_mat
            T[:3, 3] = center / 1000.0 # Convert back to meters for visualization

            copy_teris = teris_mesh.copy()
            copy_teris.apply_transform(T)

            # Register mesh
            obj = ps.register_surface_mesh(f"teris {cluster_id}", copy_teris.vertices, copy_teris.faces, color = color_cluster[cluster_id] / 255.0)
            obj.add_to_group(group)

            # Visualize frame
            visualize_frame(f"frame_{cluster_id}", center / 1000.0, rot_mat)

        group.set_hide_descendants_from_structure_lists(True)
        group.set_show_child_details(False)
    else:
        print(f"Mesh not found at {mesh_path}")

    ps.show()



