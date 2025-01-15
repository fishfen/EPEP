import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from utils.utils import rotation_mat


def pts_inside_box(pts, bbox):
    # pts: N x 3
    # bbox: 8 x 3 (-1, 1, 1), (1, 1, 1), (1, -1, 1), (-1, -1, 1), (-1, 1, -1), (1, 1, -1), (1, -1, -1), (-1, -1, -1)
    u1 = bbox[5, :] - bbox[4, :]
    u2 = bbox[7, :] - bbox[4, :]
    u3 = bbox[0, :] - bbox[4, :]

    up = pts - np.reshape(bbox[4, :], (1, 3))
    p1 = np.matmul(up, u1.reshape((3, 1)))
    p2 = np.matmul(up, u2.reshape((3, 1)))
    p3 = np.matmul(up, u3.reshape((3, 1)))
    p1 = np.logical_and(p1 > 0, p1 < np.dot(u1, u1))
    p2 = np.logical_and(p2 > 0, p2 < np.dot(u2, u2))
    p3 = np.logical_and(p3 > 0, p3 < np.dot(u3, u3))
    return np.logical_and(np.logical_and(p1, p2), p3)


def iou_3d(bbox1, bbox2, nres=50):
    '''bbox: (8, 3) eight vetices of a cuboid'''
    bmin = np.min(np.concatenate((bbox1, bbox2), 0), 0)
    bmax = np.max(np.concatenate((bbox1, bbox2), 0), 0)
    xs = np.tile(np.linspace(bmin[0], bmax[0], nres).reshape(-1, 1, 1), (1, nres, nres))
    ys = np.tile(np.linspace(bmin[1], bmax[1], nres).reshape(1, -1, 1), (nres, 1, nres))
    zs = np.tile(np.linspace(bmin[2], bmax[2], nres).reshape(1, 1, -1), (nres, nres, 1))
    pts = np.stack([xs, ys, zs], axis=-1)
    flag1 = pts_inside_box(pts, bbox1)
    flag2 = pts_inside_box(pts, bbox2)
    intersect = np.sum(np.logical_and(flag1, flag2))
    union = np.sum(np.logical_or(flag1, flag2))
    if union == 0:
        return 1
    else:
        return intersect / float(union)
    

def get_cuboid_vertices(box_codes, translation=None, rot_mats=None, rotation_order='xyz'):
    '''
    Parameters:
        box_codes: shape (n, 9) or (n, 7) array, each row is [x, y, z, dx, dy, dz, a_x, a_y, a_z]
        translation: shape (n, 3), optional
        rot_mats: shape (n, 3, 3) array of rotation matrices, optional
        rotation_order: string, default 'xyz'
    Returns:
        vertices: shape (n, 8, 3) array of vertex coordinates
    '''
    n = box_codes.shape[0]
    centers = box_codes[:, :3]
    dimensions = box_codes[:, 3:6]
    if box_codes.shape[1] == 7:
        angles = np.zeros((n, 3))
        angles[:, 2] = box_codes[:, 6]  # Only az is provided
    elif box_codes.shape[1] == 9:
        angles = box_codes[:, 6:]
    else:
        raise ValueError("box_codes must have 7 or 9 columns")

    base_vertices = np.array([
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5]
    ])

    # Scale vertices
    vertices = base_vertices[None, :, :] * dimensions[:, None, :]
     # transform the vertices to the initial pose
    rot_mats_from_angles = np.array([rotation_mat(*angle, rot_order=rotation_order) for angle in angles])
    transformed_vertices = np.einsum('nij,nkj->nki', rot_mats_from_angles, vertices) + centers[:, None, :]

    # Apply additional rotation and tranlation if provided
    if rot_mats is not None:
        transformed_vertices = np.einsum('nij,nkj->nki', rot_mats, transformed_vertices)
    if translation is not None:
        transformed_vertices = transformed_vertices + translation[:, None, :]

    return transformed_vertices


def compute_mpjpe(pred: np.ndarray, gt: np.ndarray, mean=True):
    '''
    calculate the Mean Per Joint Position Error

    Input:
        pred: (n, j, 3),
        gt: (n, j, 3)'''
    error = np.linalg.norm((pred - gt), axis=-1)
    if mean:
        return np.mean(error)
    else:
        return error


def compute_jpa(pred: np.ndarray, gt: np.ndarray, threshold=0.3):
    '''
    calculate the Joint Position Accuracy

    Input:
        pred: (n, j, 3),
        gt: (n, j, 3)'''
    error = np.linalg.norm((pred - gt), axis=-1)
    return np.mean(error < threshold)


def compute_seg_metrics(predictions, labels, num_classes):
    '''
    prediction: (n,) list of n samples segmentation
    labels: (n,) list of n samples
    '''
    if isinstance(predictions, list):
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)

    cm = confusion_matrix(labels, predictions, labels=range(num_classes))

    oa = np.sum(np.diag(cm)) / np.sum(cm)

    class_acc = np.diag(cm) / (np.sum(cm, axis=1) + 1e-10)
    macc = np.mean(class_acc)

    intersection = np.diag(cm)
    union = np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm)
    iou = intersection / (union + 1e-10)
    miou = np.mean(iou)

    precision = intersection / (np.sum(cm, axis=0) + 1e-10)
    recall = intersection / (np.sum(cm, axis=1) + 1e-10)

    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    mf1 = np.mean(f1)

    return {
        "Overall Accuracy": oa,
        "Mean Accuracy": macc,
        "Mean IoU": miou,
        "Class-wise IoU": iou.tolist(),
        "Class-wise Precision": precision.tolist(),
        "Class-wise Recall": recall.tolist(),
        "Class-wise F1": f1.tolist(),
        "Mean F1": mf1
    }


def compute_box_iou(pred_box, pred_trans, pred_rot, gt_box, gt_trans, gt_rot):
    pred_boxes = get_cuboid_vertices(pred_box, pred_trans, pred_rot)  # (n, 8, 3)
    gt_boxes = get_cuboid_vertices(gt_box, gt_trans, gt_rot)
    print('Computing box iou, may take some time...')
    iou_list = [iou_3d(pred_box, gt_box) for (pred_box, gt_box) in tqdm(zip(pred_boxes, gt_boxes))]
    return sum(iou_list) / len(iou_list)


def compute_orien_dif(vectors1, vectors2, mean=True):
    """
    Compute the orientation difference between two vectors in rad
    vectors: (n, 3)
    """
    assert vectors1.shape == vectors2.shape, "the shapes should be same"
    
    v1_normalized = vectors1 / np.linalg.norm(vectors1, axis=1)[:, np.newaxis]
    v2_normalized = vectors2 / np.linalg.norm(vectors2, axis=1)[:, np.newaxis]
    
    dot_products = np.sum(v1_normalized * v2_normalized, axis=1)
    dot_products = np.clip(dot_products, -1.0, 1.0)
    
    angles = np.arccos(dot_products)
    if mean:
        return angles.mean()
    else:
        return angles


def compute_angle_dif(pred, gt, mean=True):
    '''
    output angle difference within pi/2, unit: rad
    pred: (n,)
    gt: (n,)
    '''
    pred_mult_theta = [pred, pred + np.pi, pred - np.pi]
    theta_dif = np.stack([pred_theta - gt for pred_theta in pred_mult_theta], axis=1)
    theta_dif = np.min(np.abs(theta_dif), axis=1)
    if mean:
        return np.mean(theta_dif)
    else:
        return theta_dif