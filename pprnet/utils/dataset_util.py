import os
import numpy as np
import h5py
from pprnet import ROOT_DIR


def load_dataset_by_cycle(data_path, cycle_idx_list, scene_idx_list, num_point_in_h5 = 16384, train = True):

    data_list, trans_label_list, rot_label_list, vs_label_list, cls_label_list= [], [], [], [],[]
    
    for cycle_id in cycle_idx_list:
        # print('Loading cycle: %d'%cycle_id)
        for scene_id in scene_idx_list:
            try:
                h5_file_name = os.path.join(data_path, 'train', 'cycle_{:0>4}'.format(cycle_id), '{:0>3}.h5'.format(scene_id))

                f = h5py.File(h5_file_name, "r")
                data_list.append(f['data'][:].reshape(1, num_point_in_h5, 3))

                label = f['labels'][:]
                trans_label_list.append(label[:, :3].reshape(1, num_point_in_h5, 3))
                rot_mat = label[:, 3:12].reshape(1, num_point_in_h5, 3, 3)
                rot_label_list.append(rot_mat)
                vs = label[:, 12].reshape(1, num_point_in_h5)
                vs_label_list.append(vs)
                cls = label[:, -1].reshape(1, num_point_in_h5)
                cls_label_list.append(cls)
            except:
                print('Cycle %d scene %d error, please check' % (cycle_id, scene_id))
                continue

    dataset = {'data': np.concatenate(data_list, axis=0),  # shape: #scene,#point,3
               'trans_label': np.concatenate(trans_label_list, axis=0),  # shape: #scene,#point,3
               'rot_label': np.concatenate(rot_label_list, axis=0),  # shape: #scene,#point,3�?
               'vs_label': np.concatenate(vs_label_list, axis=0),  # shape: #scene,#point
               'cls_label': np.concatenate(cls_label_list, axis=0),  # shape: #scene,#point
               }

    return dataset

if __name__ == "__main__":
    data_path = f'E:\\h5_dataset\\bunny\\train'
    dataset = load_dataset_by_cycle(data_path, range(1, 2), range(1, 46))
    print("Finished Loading Dataset")