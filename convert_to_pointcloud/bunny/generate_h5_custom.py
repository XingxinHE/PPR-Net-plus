import os
import cv2
import numpy as np
from H5DataGenerator_custom import H5DataGenerator

# Configuration
DATA_ROOT = 'dataset'
ITEM_NAME = 'teris'
TRAIN_FOLDER = 'training'
OUT_H5_FOLDER = 'h5'

# Paths
IN_ROOT_DIR = os.path.join(DATA_ROOT, ITEM_NAME, TRAIN_FOLDER)
DEPTH_DIR = os.path.join(IN_ROOT_DIR, 'p_depth')
SEGMENT_DIR = os.path.join(IN_ROOT_DIR, 'p_segmentation')
GT_DIR = os.path.join(IN_ROOT_DIR, 'gt')
OUT_ROOT_DIR = os.path.join(DATA_ROOT, ITEM_NAME, OUT_H5_FOLDER)

if not os.path.exists(OUT_ROOT_DIR):
    os.makedirs(OUT_ROOT_DIR)

if __name__ == "__main__":
    # Initialize Generator
    g = H5DataGenerator('./convert_to_pointcloud/bunny/parameter_custom.json')

    # Iterate through cycles
    if not os.path.exists(DEPTH_DIR):
        print(f"Error: {DEPTH_DIR} does not exist.")
        exit(1)

    cycles = sorted([d for d in os.listdir(DEPTH_DIR) if d.startswith('cycle_')])

    for cycle_name in cycles:
        print(f"Processing {cycle_name}...")

        cycle_depth_dir = os.path.join(DEPTH_DIR, cycle_name)
        cycle_seg_dir = os.path.join(SEGMENT_DIR, cycle_name)
        cycle_gt_dir = os.path.join(GT_DIR, cycle_name)

        # Output cycle dir
        out_cycle_dir = os.path.join(OUT_ROOT_DIR, cycle_name)
        if not os.path.exists(out_cycle_dir):
            os.makedirs(out_cycle_dir)

        # Iterate through scenes (files)
        files = sorted([f for f in os.listdir(cycle_depth_dir) if f.endswith('_depth.png')])

        for f in files:
            # Filename format: 001_depth.png
            scene_id_str = f.split('_')[0]

            depth_path = os.path.join(cycle_depth_dir, f)
            seg_path = os.path.join(cycle_seg_dir, f"{scene_id_str}_segmentation.png")
            gt_path = os.path.join(cycle_gt_dir, f"{scene_id_str}.csv")
            out_path = os.path.join(out_cycle_dir, f"{scene_id_str}.h5")

            if not os.path.exists(seg_path):
                print(f"Warning: Segmentation file missing for {f}")
                continue
            if not os.path.exists(gt_path):
                print(f"Warning: GT file missing for {f}")
                continue

            # Load images
            # cv2.IMREAD_UNCHANGED is crucial for 16-bit images
            depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            segment_img = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)

            if depth_img is None:
                print(f"Error reading depth: {depth_path}")
                continue
            if segment_img is None:
                print(f"Error reading seg: {seg_path}")
                continue

            # Process
            # Pass None for bg_depth_img
            g.process_train_set(depth_img, None, segment_img, gt_path, out_path)
