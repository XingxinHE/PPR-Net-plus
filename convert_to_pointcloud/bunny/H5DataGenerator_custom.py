import json
import math
import numpy as np
import cv2
import os
import h5py
import time
import os

class H5DataGenerator(object):
    def __init__(self, params_file_name, target_num_point = 16384):
        '''
        Input:
            params_file_name: path of parameter file ("parameter.json")
            target_num_point: target number of sampled points, default is 16384
        '''
        self.params = self._load_parameters(params_file_name)
        self.target_num_point = target_num_point

    def process_train_set(self, depth_img, bg_depth_img, segment_img, gt_file_path, output_file_path, xyz_limit=None, verbose=True):
        '''
        Input:
            depth_img: np array of depth image, dtype is uint16
            bg_depth_img: np array of background depth image, dtype is uint16
            segment_img: np array of segment image, dtype is uint16
            gt_file_path: str
            output_file_path: str, output h5 path
            xyz_limit: None if no limit for xyz. Typical [ [xmin, xmax], [ymin, ymax], [zmin, zmax] ]
            verbose: whether to display logging info
        '''
        if verbose:
            start_time = time.time()
        # step 1: check and parse input
        assert depth_img.shape == (self.params['resolutionY'], self.params['resolutionX']) and depth_img.dtype == np.uint16
        if bg_depth_img is not None:
            assert bg_depth_img.shape == depth_img.shape and bg_depth_img.dtype == np.uint16
        # segment_img might be uint16 now
        assert segment_img.shape == depth_img.shape
        label_trans, label_rot, label_vs = self._read_label_csv(gt_file_path)

        # step 2: convet foregroud pixel to 3d points, and extract its object ids
        if bg_depth_img is not None:
            ys, xs = np.where(depth_img != bg_depth_img)
        else:
            # Use segmentation mask to find foreground (assuming 0 is background)
            ys, xs = np.where(segment_img > 0)

        zs = depth_img[ys, xs]
        obj_ids = segment_img[ys, xs]
        ys = ys + self.params['pixelOffset_Y_KoSyTopLeft']
        xs = xs + self.params['pixelOffset_X_KoSyTopLeft']
        points = self._depth_to_pointcloud_optimized(xs, ys, zs, to_mm=False, xyz_limit=xyz_limit)

        # step 3: sample or pad to target_num_point
        # if len(points) <= target_num_point, pad to target_num_point
        num_pnt = points.shape[0]
        if num_pnt == 0:
            print('No foreground points!!!!!')
            return
        if num_pnt <= self.target_num_point:
            t = int(1.0 * self.target_num_point / num_pnt) + 1
            points_tile = np.tile(points, [t, 1])
            points = points_tile[:self.target_num_point]
            obj_ids_tile = np.tile(obj_ids, [t])
            obj_ids = obj_ids_tile[:self.target_num_point]

        if num_pnt > self.target_num_point:
            # Use random sampling instead of fpsample
            sampled_idx = np.random.choice(num_pnt, self.target_num_point, replace=False)
            points = points[sampled_idx]
            obj_ids = obj_ids[sampled_idx]

        # step 4: collect labels
        # obj_ids are 1-based. label_trans is 0-based (0 is bg).
        # Ensure obj_ids are within range
        valid_mask = obj_ids < len(label_trans)
        obj_ids[~valid_mask] = 0 # Set invalid IDs to background

        label_trans = label_trans[obj_ids]
        label_rot = label_rot[obj_ids]
        label_vs = label_vs[obj_ids]
        # reset background points translation label
        bg_ids = np.where(obj_ids==0)[0]
        num_bg_pnt = len(bg_ids)
        label_trans[bg_ids] = points[bg_ids]
        labels = np.concatenate( [label_trans, label_rot, label_vs.reshape([-1, 1]), obj_ids.reshape([-1, 1])], axis=-1 )
        assert points.shape == (self.target_num_point, 3) and labels.shape == (self.target_num_point, 14)

        # step 5: save as h5 file
        if not os.path.exists(os.path.dirname(output_file_path)):
             os.makedirs(os.path.dirname(output_file_path))

        with h5py.File(output_file_path, "w") as f:
            f['data'] = points
            f['labels'] = labels
            if verbose:
                t = time.time() - start_time
                print('Successfully write to %s in %f seconds.' % (output_file_path, t))
                print('Foreground point number: %d\t Background point number: %d' % (num_pnt, num_bg_pnt))

    def _depth_to_pointcloud_optimized(self, us, vs, zs, to_mm = False, xyz_limit=None):
        assert len(us) == len(vs) == len(zs)
        camera_info = self.params
        fx = camera_info['fu']
        fy = camera_info['fv']
        cx = camera_info['cu']
        cy = camera_info['cv']
        clip_start = camera_info['clip_start']
        clip_end = camera_info['clip_end']

        Zcs = (clip_start + (zs/float(camera_info['max_val_in_depth'])) * (clip_end - clip_start))
        if to_mm:
            Zcs *= 1000
        Xcs = -(us - cx) * Zcs / fx
        Ycs = -(vs - cy) * Zcs / fy
        Xcs = np.reshape(Xcs, (-1, 1))
        Ycs = np.reshape(Ycs, (-1, 1))
        Zcs = np.reshape(Zcs, (-1, 1))
        points = np.concatenate([Xcs, Ycs, Zcs], axis=-1)

        if xyz_limit is not None:
            if xyz_limit[0] is not None:
                xmin, xmax = xyz_limit[0]
                if xmin is not None:
                    idx = np.where( points[:, 0]>xmin )
                    points = points[idx]
                if xmax is not None:
                    idx = np.where( points[:, 0]<xmax )
                    points = points[idx]
            if xyz_limit[1] is not None:
                ymin, ymax = xyz_limit[1]
                if ymin is not None:
                    idx = np.where( points[:, 1]>ymin )
                    points = points[idx]
                if ymax is not None:
                    idx = np.where( points[:, 1]<ymax )
                    points = points[idx]
            if xyz_limit[2] is not None:
                zmin, zmax = xyz_limit[2]
                if zmin is not None:
                    idx = np.where( points[:, 2]>zmin )
                    points = points[idx]
                if zmax is not None:
                    idx = np.where( points[:, 2]<zmax )
                    points = points[idx]

        return points

    def _load_parameters(self, params_file_name):
        params = {}
        with open(params_file_name,'r') as f:
            config = json.load(f)
            params = config
            # compute fu and fv
            angle = params['perspectiveAngle'] * math.pi / 180.0
            # FOV is vertical, so use resolutionY
            f = 1.0 /  ( 2*math.tan(angle/2.0) ) * params['resolutionY']
            params['fu'] = f
            params['fv'] = f
            params['cu'] = params['resolutionX'] / 2.0
            params['cv'] = params['resolutionY'] / 2.0
            params['max_val_in_depth'] = 65535.0
        return params

    def _read_label_csv(self, file_name):
        num_obj = int(os.path.basename(file_name).split('.')[0])
        label_trans, label_rot, label_vs = [], [], []

        # Initialize with BG
        label_trans.append([0, 0, 0])
        label_rot.append( np.eye(3).reshape(-1) )
        label_vs.append(0.0)

        with open(file_name, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                line = line.strip()
                if len(line) == 0:
                    continue
                words = line.split(',')
                id = int(words[0])

                label_trans.append( list(map(float, words[2:5])) )
                R = np.array(list(map(float, words[5:14]))).reshape((3,3)).T.reshape(-1)
                label_rot.append( R )
                label_vs.append( float(words[-1]) )

        label_trans = np.array(label_trans)
        label_rot = np.array(label_rot)
        label_vs = np.array(label_vs)
        return label_trans, label_rot, label_vs
