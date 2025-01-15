import random
import open3d as o3d
from pathlib import Path
import time
import numpy as np
import json
from tqdm import tqdm
import yaml
from easydict import EasyDict
import argparse
from utils.utils import rotation_mat, length_angle_to_kps

map_ind_to_part = {0:'background', 1:'boom', 2:'stick', 3:'bucket', 4:'cabin', 5:'chassis'}


def move_obj_to_ref_pose(obj_list, trans, ry, scale):
    for i, obj in enumerate(obj_list):
        rm = rotation_mat(beta=ry[i], scale=scale[i])
        obj.rotate(rm, np.array([0,0,0]))
        obj.translate(trans[i])
    return obj_list


def synthesize_pointcloud(geometries, camera_loc=[20, 0, 4], r_noise=0.03, limit_radius: float = None):
    scene = o3d.t.geometry.RaycastingScene()

    ids = []
    for i,geo in enumerate(geometries):
        # add geometry to the scene according to their sequence. 
        # The id is continuous integer starting from 0
        geo_id = scene.add_triangles(geo)
        assert i == geo_id
        ids.append(geo_id)

    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        fov_deg=120,
        center=[0, 0, 0],
        eye=camera_loc,
        up=[0, 0, 1],  # define the up direction of the camera
        width_px=720,
        height_px=200,
    )
    ans = scene.cast_rays(rays)

    rays = rays.numpy()
    range = ans['t_hit'].numpy() + np.random.normal(0, r_noise, ans['t_hit'].shape)  # Add principle noise
    pts = []
    seg_label = []
    # iterate to get the points and corresponding segmentation labels of each part
    for pid in ids:
        hit = ans['geometry_ids'].eq(pid).numpy()
        points = rays[hit][:,:3] + rays[hit][:,3:]*range[hit].reshape((-1,1))
        pts.append(points)
        # background is the last part, its index is set to 0, the index of other parts is increased by one
        seg_id = 0 if pid >= 5 else pid+1  
        seg = np.tile(seg_id,(len(points),1))
        seg_label.append(seg)
    
    pts = np.concatenate(pts)  # (n,3)
    seg_label = np.concatenate(seg_label).flatten() # (n,)

    pts += np.random.normal(0, 0.01, pts.shape)  # Addtional noise

    if limit_radius is not None:
        keep_mask = np.linalg.norm(pts, axis=1) < limit_radius
        pts = pts[keep_mask]
        seg_label = seg_label[keep_mask]

    return pts, seg_label


class DatasetGenerator:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.cfg = EasyDict(yaml.safe_load(f))
        self.file_path = Path(self.cfg.file_path)
        self.out_path = Path(self.cfg.out_path)
        self.kp_root = [0,0,0]
        self.load_obj_files()
        random.seed(self.cfg.random_seed)
        np.random.seed(self.cfg.random_seed)

    def load_obj_files(self):
        obj_dict = {}
        all_obj_list = ['arm1_1', 'arm1_2', 'arm1_1_nobrace', 'arm1_2_nobrace', 'arm2', 'arm2_nobrace', 'bucket', 'cabin', 'chasis']
        for obj_name in all_obj_list:
            file = self.file_path / f'{obj_name}.obj'
            assert file.exists(), f"File not found: {file}"
            obj = o3d.t.io.read_triangle_mesh(str(file), enable_post_processing=True)
            obj_dict[obj_name] = obj
        
        other_obj_list = []
        for v in self.cfg.add_object.values():
            other_obj_list.extend(v['obj'])
        other_obj_list = list(set(other_obj_list))
        other_obj_dict = {}
        for obj_name in other_obj_list:
            file = self.file_path / f'{obj_name}.obj'
            assert file.exists()
            obj = o3d.t.io.read_triangle_mesh(str(file), enable_post_processing=True)
            other_obj_dict[obj_name] = obj
    
        self.obj_dict = obj_dict
        self.other_dict = other_obj_dict

    def select_obj_files(self):
        arm1_names = ['arm1_1', 'arm1_2', 'arm1_1_nobrace', 'arm1_2_nobrace']
        arm2_names = ['arm2', 'arm2_nobrace']
        self.obj_file_names[0] = random.choice(arm1_names)
        self.obj_file_names[1] = random.choice(arm2_names)
        return self.obj_file_names

    def generate_dataset(self, num_sequences: int, samples_per_sequence: int):
        start_time = time.time()
        seq_tokens = []
        for seq in tqdm(range(num_sequences)):
            seq_token = f'{seq:04d}'
            seq_tokens.append(seq_token)
            self.generate_sequence(seq_token, samples_per_sequence)
        self.split_dataset(seq_tokens)

        end_time = time.time()
        print(f"Dataset generation completed in {end_time-start_time:.2f} seconds")

    def generate_sequence(self, seq_token: str, samples_per_sequence: int):
        self.obj_file_names = ['', '', 'bucket', 'cabin', 'chasis']

        obj_file_names = self.select_obj_files()
        size_params = self.generate_size_params()
        pose_params = self.generate_pose_params()
        
        objs_raw = [self.obj_dict[f] for f in obj_file_names]

        camera_angle = np.deg2rad(pose_params['camera']['angle'])
        camera_loc = [pose_params['camera']['dist'] * np.cos(camera_angle), 
                      pose_params['camera']['dist'] * np.sin(camera_angle), 
                      size_params['offset']['bottom_z'] + pose_params['camera']['height']]
        
        for i in range(samples_per_sequence):
            # Very important to clone the objs, otherwise the objs will be modified in the next iteration
            # the new t api of open3d does not support deepcopy, so we have to use the clone method
            objs = [o.clone() for o in objs_raw]
            pose = self.interpolate_pose(pose_params, i, samples_per_sequence)
            init_kps = self.compute_init_pose(pose, size_params)

            objs = self.add_clutters(self.other_dict, size_params, objs, init_kps)
            
            objs, posed_camera = self.apply_transformations(objs, camera_loc, pose, init_kps)
            
            # generate point cloud by casting ray
            pts, seg_label = synthesize_pointcloud(objs, camera_loc=posed_camera, r_noise=self.cfg.range_noise,  
                                                   limit_radius=self.cfg.pointcloud_crop_range)

            self.save_pointcloud(pts, seq_token, i)
            self.save_annotation(size_params, pose_params, pose, seg_label, seq_token, i)


    def generate_size_params(self):
        def ru(a=0.8, b=1.2):
            return random.uniform(a, b)
        
        scale_cabin = [ru(0.7,1.5),ru(),ru()]
        scale_chasis = [ru(),ru(),ru()*0.84]
        gs = ru()
        gs_delta = gs - 1
        scale_arm1 = [1+gs_delta*0.5, 1+gs_delta*0.5, gs]
        scale_arm2 = [1+gs_delta*0.5, 1+gs_delta*0.5, gs]
        scale_bucket = [ru(),ru(0.8, 1.5),ru(0.8, 1.5)]  # [x,y,z] z is parallel to its axis
        cabin_x_offset = ru(-0.4, 0.4)   # this adds on the init x offset
        root_z = ru(0.4,1.2)   # for z translation of boom root
        root_y = ru(-0.6,0)  

        size_cabin = [s*k for s,k in zip(self.cfg.size_init_cabin, scale_cabin)]
        size_chasis = [s*k for s,k in zip(self.cfg.size_init_chasis, scale_chasis)]
        size_bucket = [self.cfg.size_init_bucket[0] * scale_bucket[0], self.cfg.size_init_bucket[1] * scale_bucket[1]]  # dx, dy

        len_arm1 = self.cfg.len_arm1[self.obj_file_names[0]] * scale_arm1[2]
        len_arm2 = self.cfg.len_arm2 * scale_arm2[2]
        len_bucket = self.cfg.len_bucket * scale_bucket[2]

        z_bottom = - (size_chasis[2] + self.cfg.gap_chasis_cabin)
        z_cabin = 0   # the bottom of cabin locates on z=0
        # note cabin_x_offset is to move the obj, x_cabin is for annotation target variable
        x_cabin = cabin_x_offset + self.cfg.cabin_init_x * scale_cabin[0]
        self.kp_root = [0, root_y, root_z]

        return {
            'scale': {'cabin': scale_cabin, 'chasis': scale_chasis, 'arm1': scale_arm1, 'arm2': scale_arm2, 'bucket': scale_bucket},
            'size': {'cabin': size_cabin, 'chasis': size_chasis, 'bucket': size_bucket},
            'length': {'arm1': len_arm1, 'arm2': len_arm2, 'bucket': len_bucket},
            'offset': {'cabin_x_trans': cabin_x_offset, 'root_z':root_z, 'cabin_x': x_cabin, 'root_y': root_y, 'cabin_z': z_cabin, 'bottom_z': z_bottom}
        }

    def generate_pose_params(self):

        # initial pose
        angle1 = random.uniform(*self.cfg.limit_angle1)
        angle2 = random.uniform(*self.cfg.limit_angle2)
        angle3 = random.uniform(*self.cfg.limit_angle3)
        theta = random.uniform(*self.cfg.limit_theta)
        yaw = random.uniform(*self.cfg.limit_yaw)
        rx, ry = random.uniform(*self.cfg.limit_roll), random.uniform(*self.cfg.limit_roll)
        tx = random.uniform(*self.cfg.limit_tx)   # the excavator locates by the root of boom
        ty = random.uniform(*self.cfg.limit_ty)
        tz = random.uniform(*self.cfg.limit_tz)
        translation = [tx, ty, tz]

        # finall pose
        final_angle1 = random.uniform(*self.cfg.limit_angle1)
        final_angle2 = random.uniform(*self.cfg.limit_angle2)
        final_angle3 = random.uniform(*self.cfg.limit_angle3)
        final_theta = random.uniform(*self.cfg.limit_theta)

        camera_dist = random.uniform(*self.cfg.lidar_dist)
        camera_height = random.uniform(*self.cfg.lidar_height)
        camera_angle = random.uniform(*self.cfg.lidar_angle)

        return {
            'initial': {'angles': [angle1, angle2, angle3], 'theta': theta, 'yaw': yaw, 'rx': rx, 'ry': ry, 'translation': translation},
            'final': {'angles': [final_angle1, final_angle2, final_angle3], 'theta': final_theta},
            'camera': {'dist': camera_dist, 'height': camera_height, 'angle': camera_angle}
        }

    def interpolate_pose(self, pose_params, i: int, total_samples: int):
        if total_samples == 1:
            t = 0
        else:
            t = i / (total_samples - 1)
        initial = pose_params['initial']
        final = pose_params['final']

        angles = [np.interp(t, [0,1], [initial['angles'][j], final['angles'][j]]) for j in range(3)]
        theta = np.interp(t, [0,1], [initial['theta'], final['theta']])

        yaw, rx, ry = map(np.deg2rad, (initial['yaw'], initial['rx'], initial['ry']))
        angles = list(map(np.deg2rad, angles))
        theta = np.deg2rad(theta)

        rot_mat = rotation_mat(alpha=rx, beta=ry, gamma=yaw, rot_order='xyz')  # rot sequence: -theta -> rx -> ry -> rz

        return {
            'angles': angles,
            'theta': theta,
            'rotation_mat': rot_mat.tolist(),
            'translation': initial['translation'],
        }
    
    def compute_init_pose(self, pose, size_params):
        l1 = size_params['length']['arm1']
        l2 = size_params['length']['arm2'] 
        l3 = size_params['length']['bucket']

        init_kps, tilts = length_angle_to_kps(pose['angles'], (l1, l2, l3), self.kp_root)

        trans_list = [self.kp_root, init_kps[0], init_kps[1], 
                      [size_params['offset']['cabin_x_trans'],0,size_params['offset']['cabin_z']], 
                      [0,0,size_params['offset']['bottom_z']]]
        ry_list = [*[np.pi/2 - tilt for tilt in tilts], 0, 0]
        scale_list = [size_params['scale'][key] for key in ['arm1', 'arm2', 'bucket', 'cabin', 'chasis']]

        pose.update({'part_trans':trans_list, 'part_ry': ry_list, 'part_scale': scale_list})

        return init_kps


    def apply_transformations(self, objs, camera_loc, pose, init_kps):
        
        self.kps = ( np.array([[0,0,0]] + init_kps) 
                  @ np.array(pose['rotation_mat']).T 
                  + np.array(pose['translation']) )

        core_objs = objs[:5]
        # transform the obj to the initial pose
        core_objs = move_obj_to_ref_pose(core_objs, pose['part_trans'], pose['part_ry'], pose['part_scale'])
        core_objs[-1].rotate(rotation_mat(gamma=pose['theta']), [0,0,0]) # rotation of chasis relative to upper part

        # rotation of the whole model, from the ref pose to target pose
        rot_mat = np.array(pose['rotation_mat'])
        for obj in objs:
            obj.rotate(rot_mat, [0,0,0])
            obj.translate(pose['translation'])

        posed_camera = np.asarray(camera_loc) @ rot_mat.T + np.asarray(pose['translation'])

        return objs, posed_camera
    
    def add_clutters(self, other_obj_dict, size_param, objs, init_kps):
        # add and transform other objs, like background
        z_bottom = size_param['offset']['bottom_z']
        self.clutter = False
        for key, value in self.cfg.add_object.items():
            if random.random() < value['prob']:
                obj_name = random.choice(value['obj'])
                rot_mat = None
                if key == 'clutter':
                    rot_mat = rotation_mat(gamma=random.uniform(-3.14/2, 3.14/2), scale=np.random.uniform(1, 1.4, 3))
                    r, a = random.uniform(2,4), np.random.normal(loc=0, scale=3.14/3)
                    translation_ = [r*np.cos(a), r*np.sin(a), z_bottom]
                elif key == 'load':
                    self.clutter = True 
                    rot_mat = rotation_mat(alpha=np.pi, gamma=random.uniform(-3.14/2, 3.14/2), scale=np.random.uniform(1, 1.4, 3))
                    translation_ = np.asarray([random.uniform(-0.5,0.5), random.uniform(-0.5,0.5), -random.uniform(0,1)]) + init_kps[2]
                elif key == 'truck':
                    rot_mat = rotation_mat(gamma=random.uniform(-3.14/4, 3.14/4), scale=np.random.uniform(0.8, 1.2, 3))
                    translation_ = [random.uniform(3.5, 4.5), random.uniform(-0.5, 0.5), z_bottom]
                else:
                    translation_ = [random.uniform(-1,1),random.uniform(-1,1),z_bottom]
                obj = other_obj_dict[obj_name].clone()
                if rot_mat is not None:
                    obj.rotate(rot_mat, [0,0,0])
                objs.append(obj.translate(translation_))
        
        return objs

    def save_pointcloud(self, pts: np.ndarray, seq_token: str, sample_num: int):
        filename = self.out_path / 'pointclouds' / f'{seq_token}_{sample_num:02d}.bin'
        filename.parent.mkdir(parents=True, exist_ok=True)
        pts.astype(np.float32).tofile(filename)

    def save_annotation(self, size_params, pose_params, pose, seg_label: np.ndarray, seq_token: str, sample_num: int):
        filename = self.out_path / 'labels' / f'{seq_token}_{sample_num:02d}.json'
        filename.parent.mkdir(parents=True, exist_ok=True)

        anno_dict = {
            'meta': {
                        'point_number': len(seg_label),
                        'valid_point_number': int(np.sum(seg_label > 0)),
                        'lidar_dist': pose_params['camera']['dist'],
                        'lidar_height': pose_params['camera']['height'],
                        'lidar_angle': pose_params['camera']['angle'],
                        'arm1_name': self.obj_file_names[0],
                        'arm2_name': self.obj_file_names[1],
                        'clutter': self.clutter
                    },
            'size': {
                        'arm_length': [size_params['length']['arm1'], size_params['length']['arm2'], size_params['length']['bucket']],
                        'cabin_size': size_params['size']['cabin'],
                        'chasis_size': size_params['size']['chasis'],
                        'bucket_size': size_params['size']['bucket'],  # dx, dy
                        'cabin_x': size_params['offset']['cabin_x'],  # the x offset (not abs)
                        'root_z': size_params['offset']['root_z'],
                        'root_y': size_params['offset']['root_y'],
                        'bottom_z': size_params['offset']['bottom_z']              
                    },
            'pose': {   
                        'translation_xyz': pose['translation'],
                        'angles': pose['angles'],
                        'rotation_mat': pose['rotation_mat'],
                        'theta': pose['theta'],
                        'keypoints': self.kps.tolist()
                    },
            'point_info': {'segment': seg_label.tolist()}
        }
        with open(filename, 'w') as f:
            json.dump(anno_dict, f)

    def split_dataset(self, seq_tokens):
        r_train, r_val, r_test = 0.6, 0.2, 0.2
        sp1 = round(r_train * len(seq_tokens))
        sp2 = sp1 + round(r_val * len(seq_tokens))
        train = seq_tokens[:sp1]
        val = seq_tokens[sp1:sp2]
        test = seq_tokens[sp2:]
        for name, seq in zip(['train', 'val', 'test'], [train, val, test]):
            with open(self.out_path / f'{name}.txt', 'w') as f:
                for token in seq:
                    f.write(f'{token}\n')
            print(f'{name} set has {len(seq)} sequences')



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--num_sequence', type=int, default=16000)
    argparser.add_argument('--samples_per_sequence', type=int, default=1)
    args = argparser.parse_args()

    num_sequence = args.num_sequence
    samples_per_sequence = args.samples_per_sequence

    generator = DatasetGenerator('cfg/syn_data.yaml')
    generator.generate_dataset(num_sequences=num_sequence, samples_per_sequence=samples_per_sequence)