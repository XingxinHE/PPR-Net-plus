import h5py
import numpy as np
import os

file_path = 'dataset/teris/training/h5/cycle_0001/001.h5'

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
else:
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"Inspecting: {file_path}")
            print("-" * 30)

            def print_attrs(name, obj):
                print(name)
                if isinstance(obj, h5py.Dataset):
                    print(f"  Shape: {obj.shape}")
                    print(f"  Dtype: {obj.dtype}")
                elif isinstance(obj, h5py.Group):
                    print("  Group")

            f.visititems(print_attrs)

            print("-" * 30)
            # Check for expected keys for PPR-Net
            expected_keys = ['data', 'label'] # Based on typical pointnet/pprnet datasets, but let's see what's there.
            # Actually, based on H5DataGenerator.py (which I should probably check if I haven't),
            # let's see what keys are usually generated.

    except Exception as e:
        print(f"Error reading file: {e}")