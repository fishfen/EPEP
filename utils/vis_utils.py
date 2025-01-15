import open3d
import numpy as np
import matplotlib.pyplot as plt
import time
import yaml
from easydict import EasyDict
from pathlib import Path
from matplotlib.colors import Normalize
from utils.utils import rotation_mat, length_angle_to_kps

color_table = [
    (1.0, 0.0, 0.0),  # Red
    (0.0, 0.8, 0.0),  # Green
    (0.0, 0.0, 1.0),  # Blue
    (1.0, 0.7, 0.0),  # Yellow
    (0.8, 0.0, 0.8),  # Purple
    (0.0, 0.8, 0.8),  # Cyan
    (0.7, 0.7, 0.7)   # Gray
]

color_table_light = [
    (0.9, 0.3, 0.9),  # Light Purple
    (0.3, 0.65, 1.0), # Light Blue
    (0.3, 0.9, 0.3),  # Light Green
    (1.0, 0.75, 0.2), # Light Yellow
    (1.0, 0.5, 0.5),  # Light Red
    (0.5, 0.9, 0.9),  # Light Cyan
    (0.7, 0.7, 0.7)   # Light Gray
]


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    lines = np.asarray(line_set.lines)
    points = np.asarray(line_set.points)
    vector = ((points[1]-points[0])/lwh[0]).reshape(1,-1)
    mid = ((points[4]+points[6]+points[7]+points[1])/4).reshape(1,-1)
    points = np.concatenate([points, mid, mid-vector*0.5], axis=0)
    lines = np.concatenate([lines, np.array([[9, 8]])], axis=0)

    line_set.points = open3d.utility.Vector3dVector(points)
    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def line_to_cylinder(pt0, pt1, radius=0.1):
    pt0, pt1 = np.asarray(pt0), np.asarray(pt1)
    v = pt1 - pt0
    # print('Line %d %d '%(line[1], line[0]), v)

    xy = np.sqrt(v[0]**2 + v[1]**2)
    rot_y = np.arctan2(xy, v[2])
    rot_z = np.arctan2(v[1], v[0]) # use arctan2 rather than arctan, easy to convert (x,y) to angle

    length = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    cyl = open3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length,resolution=8,split=1)
    rot_mat = rotation_mat(beta=rot_y, gamma=rot_z)
    cyl.rotate(rot_mat, np.array([0,0,0]).reshape(3,1))
    cyl.translate((pt1 + pt0)/2)
    return cyl


def geo_align_to_line(geo, pt0, pt1, l0):
    '''TODO has issue, only use roty and rotz is not enough'''
    pt0, pt1 = np.asarray(pt0), np.asarray(pt1)
    v = pt1 - pt0
    scale = np.linalg.norm(v) / l0
    xy = np.sqrt(v[0]**2 + v[1]**2)
    rot_y = np.arctan2(xy, v[2])
    rot_z = np.arctan2(v[1], v[0]) # use arctan2 rather than arctan, easy to convert (x,y) to angle
    rot_mat = rotation_mat(beta=rot_y, gamma=rot_z)
    geo.scale(scale, center=(0,0,0))
    geo.rotate(rot_mat, np.array([0,0,0]).reshape(3,1))
    geo.translate(pt0)
    return geo


def draw_box_cylinder(gt_boxes, radius=0.2, color=(0,1,0)):
    '''visualize the 3d bounding box with cylinder'''
    geo = []
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        lines = np.asarray(line_set.lines)
        points = np.asarray(line_set.points)
        # print('box', i)
        for line in lines:
            cyl = line_to_cylinder(points[line[1]], points[line[0]], radius)
            cyl.paint_uniform_color(np.array(color).reshape(3,1))
            cyl.compute_vertex_normals()
            geo.append(cyl)

    return geo


def render_3d(points=None, point_color=None, objects: list=None, frame_size=1, axis=True):
    '''
    The base function to render 3d points and objects
    points: (n, 3) array
    point_color: (n, 3) or (n,) or None. if None, will use gray scale based on z value
    '''
    render_list = []

    if points is not None:
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(points)

        if point_color is None:
            norm = Normalize(vmin=points[:, 2].min(), vmax=points[:, 2].max())
            gray_values = 0.7 - 0.7 * norm(points[:, 2])
            point_color = plt.cm.gray(gray_values)[:, :3]
            # point_color = np.ones((points.shape[0], 3)) * 0.5
        elif len(point_color.shape) == 1:
            point_color = array_to_color(point_color)
        elif len(point_color.shape) == 2:
            assert point_color.shape[1] == 3
            point_color = np.clip(point_color, 0, 1)
        else:
            raise ValueError('The shape of point_color is not correct')
        
        pcd.colors = open3d.utility.Vector3dVector(point_color)
        render_list.append(pcd)

    if axis:
        cf0 = open3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size, origin = [0,0,0]) # x, y, z  as red, green, and blue
        render_list.append(cf0)

    if objects:
        render_list.extend(objects)
    open3d.visualization.draw_geometries(render_list)


def render_pose_with_points(anno_dict, points, show_camera=False, key_width=0.12, other_width=0.06):  
    '''
    The main function to render pose in geometry primitives (e.g., lines, cuboids) with 3d points
    
    Input:
        anno_dict: dict, the annotation or predicted dict that contains the pose and size information
        points: (n, 3) array, the 3d points
        show_camera: bool, if show the camera location
        key_width: float, the width of the key links
        other_width: float, the width of the other lines
    '''
    size = anno_dict['size']
    pose = anno_dict['pose']  
    R = np.array(pose['rotation_mat'])

    # insert boom key points by root z
    boom_root = pose['keypoints'][0] + R[:,2] * size['root_z'] + R[:,1] * size['root_y']
    keypoints = pose['keypoints'][:]
    keypoints.insert(1, boom_root)

    geo = []
    # render arms
    for start, end in zip(keypoints[:4], keypoints[1:]):
        cyl = line_to_cylinder(start, end, key_width/2)
        cyl.paint_uniform_color([1,0,0])
        geo.append(cyl)
    
    # render bucket 
    bucket_size = size.get('bucket_size') if size.get('bucket_size') else [0.9, 1.2]
    plane_norm = R[:,1]
    kp3, kp4 = np.array(pose['keypoints'][2]), np.array(pose['keypoints'][3])
    kp_bucket_mid = (kp3 + kp4) / 2
    k1 = kp_bucket_mid + plane_norm * bucket_size[1] / 2
    k2 = kp_bucket_mid - plane_norm * bucket_size[1] / 2
    v_ = np.cross(kp4 - kp3, plane_norm)
    v_ = v_ / np.linalg.norm(v_) * bucket_size[0]
    k3 = v_ + k1
    k4 = v_ + k2
    for start, end in [(k1, k2), (k3, k4), (k1, k3), (k2, k4)]:
        cyl = line_to_cylinder(start, end, other_width/2)
        cyl.paint_uniform_color([0,0,1])
        geo.append(cyl)

    # render key points
    for kp in keypoints:
        sp = open3d.geometry.TriangleMesh.create_sphere(radius=0.2, resolution=10)
        sp.translate(kp)
        sp.paint_uniform_color([1,0,0])
        geo.append(sp)

    # render bbox
    size['cabin_size'][2] += 0.2
    box_chasis = [0, 0, -size['chasis_size'][2] / 2 -0.1] + size['chasis_size'] + [pose['theta']]
    box_cabin = [size['cabin_x'], 0, size['cabin_size'][2] / 2] + size['cabin_size'] + [0.0]
    boxes = np.row_stack([box_cabin, box_chasis])
    cyls = draw_box_cylinder(boxes, radius=other_width/2, color=(0,0,1))
    for cyl in cyls:
        cyl.rotate(R, [0,0,0])
        cyl.translate(pose['translation_xyz'])
    geo.extend(cyls)

    if show_camera and anno_dict['meta'].get('lidar_loc'):
        sp = open3d.geometry.TriangleMesh.create_sphere(radius=0.4, resolution=10)
        sp.translate(anno_dict['meta'].get('lidar_loc'))
        sp.paint_uniform_color([0,0,1])
        geo.append(sp)

    for g in geo:
        g.compute_vertex_normals()
    
    if anno_dict.get('point_info'):
        seg = anno_dict['point_info']['segment']
        # background that has id 0 will be -1 to index last color, ie, grey
        color = np.array([color_table_light[i-1] for i in seg])   
    else:
        norm = Normalize(vmin=points[:, 2].min(), vmax=points[:, 2].max())
        gray_values = 0.7 - 0.7 * norm(points[:, 2])
        color = plt.cm.gray(gray_values)[:, :3]
    render_3d(points, color, geo, axis=False)


def render_pose_with_geo(anno_dict, geo_path=Path('data/obj'), config_path=Path('cfg/syn_data.yaml')):
    '''use geometries to render the excavator pose'''
    with open(config_path, 'r') as f:
        cfg = EasyDict(yaml.safe_load(f))
    obj_dict = {}
    obj_names = {'arm1':'arm1_2', 'arm2':'arm2', 'bucket':'bucket', 'cabin':'cabin', 'chasis':'chasis'}
    for it, name in obj_names.items():
        file = geo_path / f'{name}.obj'
        assert file.exists(), f"File not found: {file}"
        obj = open3d.io.read_triangle_mesh(str(file), enable_post_processing=True)
        obj_dict[it] = obj
    
    size = anno_dict['size']
    pose = anno_dict['pose']  
    R = np.array(pose['rotation_mat'])
    translation = np.array(pose['translation_xyz'])
    kps = np.array(pose['keypoints'])
    # insert boom key points by root z
    boom_root = kps[0] + R[:,2] * size['root_z'] + R[:,1] * size['root_y']
    kps[0] = boom_root

    kps_ = (kps - translation) @ R
    vectors = [kps_[i+1] - kps_[i] for i in range(len(kps_)-1)]
    tilts = [np.arctan2(v[2], v[0]) for v in vectors]
    if size.get('arm_length'):
        lengths = size['arm_length']
    else:
        lengths = [np.linalg.norm(v) for v in vectors]
    
    kps = [kps_[0]]
    for i in range(len(tilts)):
        kp = np.array([lengths[i] * np.cos(tilts[i]), 0, lengths[i] * np.sin(tilts[i])]) + kps[i]
        kps.append(kp)
        
    arm_names = ['arm1', 'arm2', 'bucket']
    arm_ref_len = [cfg.len_arm1[obj_names['arm1']], cfg.len_arm2, cfg.len_bucket]
    for i in range(len(tilts)):
        scale = np.full(3, lengths[i] / arm_ref_len[i])
        rm = rotation_mat(beta=np.pi/2 - tilts[i], scale=scale)
        obj_dict[arm_names[i]].rotate(rm, [0,0,0])
        obj_dict[arm_names[i]].translate(kps[i])

    scale_cabin = np.array(size['cabin_size']) / np.array(cfg.size_init_cabin)
    tm = rotation_mat(scale=scale_cabin)
    obj_dict['cabin'].rotate(tm, [0,0,0])  # no rotation, only scale
    obj_dict['cabin'].translate([size['cabin_x'] - cfg.cabin_init_x, 0, 0])

    scale_chasis = np.array(size['chasis_size']) / np.array(cfg.size_init_chasis)
    tm = rotation_mat(gamma=pose['theta'], scale=scale_chasis)
    obj_dict['chasis'].rotate(tm, [0,0,0])
    obj_dict['chasis'].translate([0, 0, -size['chasis_size'][2] -0.1])

    for obj in obj_dict.values():
        obj.rotate(R, [0,0,0])
        obj.translate(translation)

    geometries = list(obj_dict.values())
    for comp in geometries:
        comp.paint_uniform_color(np.full(3,0.7))
        comp.compute_vertex_normals()
    
    render_3d(objects = geometries, axis=False)


def render_labels(points, label_dict):
    geo = []
    if 'keypoints' in label_dict:
        for kp in label_dict['keypoints']:
            sp = open3d.geometry.TriangleMesh.create_sphere(radius=0.2, resolution=10)
            sp.translate(kp)
            sp.paint_uniform_color([1,0,0])
            geo.append(sp)
    
    seg = label_dict['segment']
    color = np.array([color_table[i-1] for i in seg])  
    render_3d(points, color, geo)


def render_pointoffset(points, offset, mask=None):
    '''
    Input:
        point: (n, 3)  array
        offset: (n, 3)  array
        mask: (n,) [0,1]
    '''
    start = points
    end = start + offset
    points = np.concatenate([start, end])
    point_color = np.zeros((len(points),3))
    point_color[:len(start)] = [0,0,1]
    point_color[len(start):] = [0,1,0]

    corres = [(i,i) for i in range(len(start))]
    if mask is not None:
        corres = [corres[i] for i in range(len(corres)) if mask[i]]
    pcd_start = open3d.geometry.PointCloud()
    pcd_start.points = open3d.utility.Vector3dVector(start)
    pcd_end = open3d.geometry.PointCloud()
    pcd_end.points = open3d.utility.Vector3dVector(end)
    vectors = open3d.geometry.LineSet.create_from_point_cloud_correspondences(pcd_start, pcd_end, corres)
    render_3d(points, point_color, [vectors], axis=False)


def render_while_debug(points, offset, heatmap, kps, kp_ind=0, clean=True, mask_offset=False):
    '''
    clean: if remove the outliers in the heatmap
    mask_offset: if only show the offset with prominent heat value '''
    geo = []
    for kp in kps:
        sp = open3d.geometry.TriangleMesh.create_sphere(radius=0.2, resolution=10)
        sp.translate(kp)
        sp.paint_uniform_color([1,0,0])
        geo.append(sp)

    hm = heatmap[:,kp_ind]
    if clean:
        hm = remove_outliers(hm)  # for easy visualize the distribution

    print(f'Heatmap values max: {hm.max()}, min: {hm.min()}')
    time.sleep(1)
    if mask_offset:
        sorted_hm = np.sort(hm)
        differences = np.diff(sorted_hm)
        max_diff_index = np.argmax(differences)
        mask = hm > sorted_hm[max_diff_index]
        print('heatmap values over mask:')
        print(hm[mask])
    else:
        mask = None

    render_3d(points, hm, geo)
    render_pointoffset(points, offset[:, kp_ind], mask)


def array_to_color(arr, cmap_name='viridis'):
    """
    Map a 1D array to a specified colormap and return an RGB color array.
    
    Parameters:
    arr : (n,) array-like
    cmap_name : str, optional, The name of the colormap to use, default is 'viridis'
    
    Returns:
    colors: (n, 3) RGB color array
    """
    arr = np.asarray(arr).flatten()
    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=arr.min(), vmax=arr.max())
    scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = scalar_map.to_rgba(arr)[:, :3]
    return colors


def remove_outliers(arr, factor=1.0):
    """
    arr : numpy.ndarray
    factor : float
    """
    arr = np.asarray(arr).flatten()

    Q1 = np.percentile(arr, 10)
    Q3 = np.percentile(arr, 95)
    IQR = Q3 - Q1
    lower_bound = max(Q1 - factor * IQR, np.partition(arr, 2)[2])
    upper_bound = Q3 + factor * IQR

    outliers = arr[(arr < lower_bound) | (arr > upper_bound)]
    print(f"{len(outliers)} outliers detected:")
    print(np.sort(outliers))

    normal_max = np.max(arr[(arr >= lower_bound) & (arr <= upper_bound)])
    arr_cleaned = np.where((arr < lower_bound) | (arr > upper_bound), normal_max, arr)

    return arr_cleaned

