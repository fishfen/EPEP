
import numpy as np
import torch
import os
import logging
from datetime import datetime
import yaml

def rotation_mat(alpha=0, beta=0, gamma=0, rot_order='xyz', scale=[1,1,1], is_degree=False):
      '''
      Compute the matrix to scale and then rotate the object, relative to the origin.
      The rotation is performed in the order of *rot_order*, default is 'xyz', i.e., first rotate along x, then y, and finally z.
      Use the right-hand coordinate system, i.e., the positive direction of the rotation angle is counterclockwise.
      
      Input:
            alpha, beta, gamma: rotation angles along x, y, z axis, in radian or degree
            rot_order: 'xyz'(default) or 'zyx
            scale: [sx, sy, sz]
            is_degree: if True, the input angles are in degree
      '''
      if is_degree:
            alpha, beta, gamma = np.deg2rad(alpha), np.deg2rad(beta), np.deg2rad(gamma)
      Rx = [[1, 0            ,  0            ],
            [0, np.cos(alpha), -np.sin(alpha)],
            [0, np.sin(alpha),  np.cos(alpha)],]
      Ry = [[ np.cos(beta), 0, np.sin(beta)],
            [ 0           , 1, 0           ],
            [-np.sin(beta), 0, np.cos(beta)],]
      Rz = [[np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma),  np.cos(gamma), 0],
            [0            ,  0            , 1],]
      
      Rx = np.array(Rx); Ry = np.array(Ry); Rz = np.array(Rz)
      if rot_order == 'xyz':
            R = Rz @ Ry @ Rx
      elif rot_order == 'zyx':
            R = Rx @ Ry @ Rz
        
      S = np.diag(scale)

      return  R @ S  # note the order of the transformation, from right to left


def length_angle_to_kps(angles, lengths, root=[0,0,0]):
    ''' 
    calculate the keypoints (1, 2, 3) and tilt angles of each arm component in a chain from root
    return:
        keypoints: [kp1, kp2, kp3]
        tilt_angles: [tilt1, tilt2, tilt3]
    '''
    assert len(angles) == 3 and len(lengths) == 3, 'Input should has length 3'
    angle1, angle2, angle3 = angles[0], angles[1], angles[2]
    l1, l2, l3 = lengths[0], lengths[1], lengths[2],
    kp_root = np.array(root)
    tilt1 = angle1
    kp1 = np.array([l1 * np.cos(tilt1), 0, l1 * np.sin(tilt1)]) + kp_root
    tilt2 = angle2 + tilt1 - np.pi
    kp2 = np.array([l2 * np.cos(tilt2), 0, l2 * np.sin(tilt2)]) + kp1
    tilt3 = angle3 + tilt2 - np.pi
    kp3 = np.array([l3 * np.cos(tilt3), 0, l3 * np.sin(tilt3)]) + kp2
    return [kp1, kp2, kp3], [tilt1, tilt2, tilt3]


def truncated_normal(loc, d, scale=0.4):
    '''
    d: the boundary to limit the value
    scale: the sigma relative to d
    '''
    while True:
        std_dev = d * scale
        random_number = np.random.normal(loc, scale=std_dev)        
        # If the number is within [-d, d], return it
        if loc-d <= random_number <= loc+d:
            return random_number


def filtered_mean(data, scale=1.0):
    '''
    if data is 2D, each column is processed separately
    return the mean of the data, and the lower and upper bounds of the data'''
    if len(data.shape) == 2:
        Q1 = np.percentile(data, 25, axis=0)
        Q3 = np.percentile(data, 75, axis=0)
        IQR = Q3 - Q1
        lower_bounds = Q1 - scale * IQR
        upper_bounds = Q3 + scale * IQR
        
        # Create a mask for valid data points
        mask = (data >= lower_bounds) & (data <= upper_bounds)
        
        # Calculate means for each column, ignoring masked values
        means = np.ma.array(data, mask=~mask).mean(axis=0)
        
        return np.array(means), lower_bounds, upper_bounds
    else:
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - scale * IQR
        upper_bound = Q3 + scale * IQR
        
        filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
        return np.mean(filtered_data), lower_bound, upper_bound
    

def is_SO3(matrix):
    if isinstance(matrix, torch.Tensor):
        if matrix.is_cuda:
            matrix = matrix.cpu()
        matrix = matrix.numpy()
    # Check if the matrix is 3x3
    if matrix.shape != (3, 3):
        return False
    
    # Check if the matrix is orthogonal
    # A.T @ A should be very close to the identity matrix
    identity = np.eye(3)
    if not np.allclose(matrix.T @ matrix, identity, atol=1e-6):
        return False
    
    # Check if the determinant is 1
    if not np.isclose(np.linalg.det(matrix), 1, atol=1e-6):
        return False
    
    # If all checks pass, the matrix is SO(3)
    return True


@torch.inference_mode()
def offset2bincount(offset):
    return torch.diff(
        offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
    )


@torch.inference_mode()
def offset2batch(offset):
    bincount = offset2bincount(offset)
    return torch.arange(
        len(bincount), device=offset.device, dtype=torch.long
    ).repeat_interleave(bincount)


@torch.inference_mode()
def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()


def setup_logger(log_dir=None, name=None):
    name = __name__ if name is None else name
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s  %(levelname)s  %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"training_log_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class Iter_counter:
    def __init__(self, start=0) -> None:
        self.iter = start

    def step(self):
        self.iter += 1
    
    def set_start(self, num):
        self.iter = num


def serialize_data(data):
    '''
    Serialize data to a format that can be saved to JSON'''
    if isinstance(data, dict):
        return {key: serialize_data(value) for key, value in data.items()}
    elif isinstance(data, (list, tuple)):
        return [serialize_data(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.integer, np.floating)):
        return float(data)
    elif isinstance(data, (str, int, float, bool, type(None))):
        return data
    else:
        return str(data)  # Convert any other types to string
    

def format_result_dict(raw_dict, batch_offset):
    '''
    Format the batch result dict, return a list of dict containing results of each sample.
    The contents in the dict are expected to be numpy array
    '''
    out_list = []
    segment = raw_dict['segment']
    if not isinstance(batch_offset, list):
        batch_offset = batch_offset.tolist()
    batch_offset = [0] + batch_offset

    for i in range(len(batch_offset)-1):
        sep_dict = {key: value[i] for key, value in raw_dict.items()}

        out_dict = {
            'size': {
                        'cabin_size': sep_dict['cabin_size'].tolist(),
                        'chasis_size':sep_dict['chasis_size'].tolist(),
                        'bucket_size':sep_dict['bucket_size'].tolist(),
                        'cabin_x':sep_dict['other_offset'][2],
                        'root_z':sep_dict['other_offset'][0],
                        'root_y':sep_dict['other_offset'][1]
                    },
            'pose': {   
                        'translation_xyz': sep_dict['keypoints'][0].tolist(),
                        'rotation_mat':  sep_dict['rotation'].tolist(),
                        'theta':  sep_dict['theta'],
                        'keypoints': sep_dict['keypoints'].tolist()   # 4x3
                    },
            'point_info': {'segment': segment[batch_offset[i]:batch_offset[i+1]].tolist()}
        }
        out_list.append(out_dict)
    
    return out_list


def deep_update(base_dict, update_dict):
    """Recursively update base_dict with update_dict"""
    for key, value in update_dict.items():
        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
            deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


def load_config(cfg_path):
    '''Load config from yaml file, and merge with base config if specified'''
    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load base config if specified
    if 'base_config' in config:
        base_cfg_path = config['base_config']
        with open(base_cfg_path, 'r') as f:
            base_config = yaml.safe_load(f)
        
        # Merge configs (train_config takes precedence)
        final_config = deep_update(base_config.copy(), config)
    else:
        final_config = config
    
    return final_config
