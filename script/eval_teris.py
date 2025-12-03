import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import warp as wp
import numpy as np

from pprnet import ROOT_DIR
from pprnet.pprnet import PPRNet, load_checkpoint
from pprnet.object_type import ObjectType
from pprnet.data.IPA_pose_dataset import IPAPoseDataset
from pprnet.data.pointcloud_transforms import PointCloudShuffle, ToTensor

def evaluate():
    # ----------------------- SETTINGS -----------------------
    BATCH_SIZE = 1 # Process one by one for clearer inspection
    NUM_POINT = 2**14

    # Path to your checkpoint
    CHECKPOINT_PATH = os.path.join(ROOT_DIR, 'logs', "ppr_teris", "log_teris_test", "checkpoint.tar")

    # Dataset settings
    DATASET_DIR = r'/workspace/PPR-Net-plus/dataset/teris/h5'
    TEST_CYCLE_RANGE = [121, 150] # Using the test range defined in training
    TEST_SCENE_RANGE = [1, 8]
    # --------------------------------------------------------

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define Object Type (Same as training)
    type_teris = ObjectType(type_name='teris', class_idx=0, symmetry_type='finite',
                            lambda_p=[[0.039965, 0.0, 0.0], [0.0, 0.028565, 0.0], [0.0, 0.0, 0.018634]],
                            G=[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])

    backbone_config = {
        'npoint_per_layer': [4096, 1024, 256, 64],
        'radius_per_layer': [[10, 20, 30], [30, 45, 60], [60, 80, 120], [120, 160, 240]]
    }

    # Initialize Model
    wp.init()
    loss_weights = {
        'trans_head': 1.0,
        'rot_head': 1.0,
        'vis_head': 1.0
    }
    net = PPRNet(type_teris, backbone_config, True, loss_weights, True)
    net.to(device)

    # Load Checkpoint
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        net.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint['epoch']}")
    else:
        print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}")
        return

    net.eval()

    # Load Dataset
    transforms_ = transforms.Compose([
        PointCloudShuffle(NUM_POINT),
        ToTensor()
    ])

    print('Loading test dataset...')
    test_dataset = IPAPoseDataset(DATASET_DIR, TEST_CYCLE_RANGE, TEST_SCENE_RANGE, transforms=transforms_)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f'Test dataset loaded with {len(test_dataset)} samples.')

    # Evaluation Loop
    total_dist_error = 0
    count = 0

    print("\nStarting Evaluation...")
    print(f"{'Sample':<10} | {'Dist Error':<15} | {'Pred Trans (m)':<30} | {'GT Trans (m)':<30}")
    print("-" * 90)

    with torch.no_grad():
        for i, batch_samples in enumerate(test_loader):
            labels = {
                'rot_label': batch_samples['rot_label'].to(device),
                'trans_label': batch_samples['trans_label'].to(device),
                'vis_label': batch_samples['vis_label'].to(device)
            }
            inputs = {
                'point_clouds': batch_samples['point_clouds'].to(device),
                'labels': labels
            }

            # Forward pass
            pred_results, _ = net(inputs)

            # pred_results[0] is translation
            pred_trans = pred_results[0].view(-1, 3)
            gt_trans = labels['trans_label'].view(-1, 3)

            # Calculate distance error for this batch
            # Assuming units are consistent (e.g. mm or m)
            dist_error = torch.mean(torch.norm(pred_trans - gt_trans, dim=1)).item()

            total_dist_error += dist_error
            count += 1

            # Print first few samples or samples with high error
            if i < 10 or dist_error > 5.0:
                # Take the first point's prediction for display
                p_t = pred_trans[0].cpu().numpy()
                g_t = gt_trans[0].cpu().numpy()
                print(f"{i:<10} | {dist_error:<15.4f} | {str(np.round(p_t, 3)):<30} | {str(np.round(g_t, 3)):<30}")

    avg_dist_error = total_dist_error / count
    print("-" * 90)
    print(f"Average Distance Error: {avg_dist_error:.4f}")

if __name__ == "__main__":
    evaluate()
