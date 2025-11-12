import pprnet.utils.dataset_util as dataset_util
import numpy as np
import torch.utils.data as data

class IPAPoseDataset(data.Dataset):
    def __init__(self, data_dir, cycle_range, scene_range, transforms=None, scale=1000.0, train = True):
        self.num_point = 16384
        self.transforms = transforms
        self.dataset = dataset_util.load_dataset_by_cycle(data_dir, range(cycle_range[0], cycle_range[1]), range(scene_range[0], scene_range[1]), train = train)
        # convert to mm
        self.dataset['data'] *= scale
        self.dataset['trans_label'] *= scale

    def __len__(self):
        return self.dataset['data'].shape[0]

    def __getitem__(self, idx):
        sample = {
            'point_clouds': self.dataset['data'][idx].copy().astype(np.float32),
            'rot_label': self.dataset['rot_label'][idx].copy().astype(np.float32),
            'trans_label':self.dataset['trans_label'][idx].copy().astype(np.float32),
            'cls_label':self.dataset['cls_label'][idx].copy().astype(np.int64),
            'vis_label':self.dataset['vs_label'][idx].copy().astype(np.float32)
        }
        if self.transforms is not None:
            sample = self.transforms(sample)
        
        return sample
