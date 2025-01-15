import numpy as np
import torch
from tqdm import tqdm
from utils.utils import serialize_data, filtered_mean, offset2batch
from utils.train_utils import load_dict_to_cuda
from utils.metrics import compute_orien_dif, compute_box_iou, compute_mpjpe, compute_jpa, compute_seg_metrics, compute_angle_dif


def evaluate(model, test_loader, device, cfg, logger):
    model.eval()
    pred = {'keypoints':[], 'segment':[], 'rotation':[], 'cabin_box':[], 'chasis_box':[], 'theta':[]}  
    gt = {'keypoints':[], 'rotation':[], 'segment':[], 'cabin_box':[], 'chasis_box':[], 'theta':[]}
    mt_dict = {}
    has_segment = False
    tokens = []

    with torch.no_grad():
        for batch_idx, (input, anno) in enumerate(test_loader):
            input = load_dict_to_cuda(input, device)

            result_dict = model(input)

            tokens.extend(anno['token'])
            pred['keypoints'].append(result_dict['keypoints'])  # (b, 4, 3)
            pred['rotation'].append(result_dict['rotation'])  # numpy array  (b, 3, 3)
            pred['theta'].append(result_dict['theta'])  # (b, 1)

            for key, value in anno.items():
                if isinstance(value, torch.Tensor):
                    anno[key] = value.numpy()

            gt['keypoints'].append(anno['keypoints'])  # tensor
            gt['rotation'].append(anno['rotation']) 
            gt['theta'].append(anno['theta']) 

            chasis_box_batch, cabin_box_batch = collate_box(result_dict)
            pred['cabin_box'].append(cabin_box_batch)  # (b, 7)
            pred['chasis_box'].append(chasis_box_batch)

            chasis_box_batch, cabin_box_batch = collate_box(anno)
            gt['cabin_box'].append(cabin_box_batch)  # (b, 7)
            gt['chasis_box'].append(chasis_box_batch)

            if 'segment' in anno:
                pred['segment'].append(result_dict['segment']) # (n,) segmentation index
                gt['segment'].append(anno['segment'])
                has_segment = True

    for this_dict in [pred, gt]:
        for key, value in this_dict.items():
            if len(value) > 0:
                this_dict[key] = np.concatenate(value)

    mt_dict['mpjpe'] = {'translation': compute_mpjpe(pred['keypoints'][:,0,:], gt['keypoints'][:,0,:]),
                        'boom_end': compute_mpjpe(pred['keypoints'][:,1,:], gt['keypoints'][:,1,:]),
                        'arm_end': compute_mpjpe(pred['keypoints'][:,2,:], gt['keypoints'][:,2,:]),
                        'bucket_end': compute_mpjpe(pred['keypoints'][:,3,:], gt['keypoints'][:,3,:]),
                        'arm_average': compute_mpjpe(pred['keypoints'][:,1:,:], gt['keypoints'][:,1:,:]),
                        'average': compute_mpjpe(pred['keypoints'], gt['keypoints'])}
    
    thre = cfg.jpa_thre
    mt_dict['jpa'] = {'translation': compute_jpa(pred['keypoints'][:,0,:], gt['keypoints'][:,0,:], thre),
                        'boom_end': compute_jpa(pred['keypoints'][:,1,:], gt['keypoints'][:,1,:], thre),
                        'arm_end': compute_jpa(pred['keypoints'][:,2,:], gt['keypoints'][:,2,:], thre),
                        'bucket_end': compute_jpa(pred['keypoints'][:,3,:], gt['keypoints'][:,3,:], thre),
                        'arm_average': compute_jpa(pred['keypoints'][:,1:,:], gt['keypoints'][:,1:,:], thre),
                        'average': compute_jpa(pred['keypoints'], gt['keypoints'], thre)}
    
    mt_dict['box_iou'] = {'cabin': compute_box_iou(pred['cabin_box'], pred['keypoints'][:,0,:], pred['rotation'], 
                                                   gt['cabin_box'], gt['keypoints'][:,0,:], gt['rotation']),
                            'chasis': compute_box_iou(pred['chasis_box'], pred['keypoints'][:,0,:], pred['rotation'], 
                                                      gt['chasis_box'], gt['keypoints'][:,0,:], gt['rotation'])}
    
    angle_dif = np.degrees(compute_angle_dif(pred['theta'], gt['theta'], mean=False))
    mt_dict['rotation_error'] = {   'R_x':np.degrees(compute_orien_dif(pred['rotation'][:,:,0], gt['rotation'][:,:,0])),
                                    'R_y':np.degrees(compute_orien_dif(pred['rotation'][:,:,1], gt['rotation'][:,:,1])),
                                    'R_z':np.degrees(compute_orien_dif(pred['rotation'][:,:,2], gt['rotation'][:,:,2])),
                                    'theta':np.mean(angle_dif)}
    if has_segment:
        mt_dict['segment'] = compute_seg_metrics(pred['segment'], gt['segment'], num_classes=model.cfg.point_wise_predict['segment']['out_channels'])
    
    mt_dict = serialize_data(mt_dict)
    return mt_dict, (tokens, angle_dif)


def estimate_size(model, test_loader, device, logger):
    '''
    estimate size for all samples, and optimize the estimated size, make sure the same excavator has the same size output
    the size code has length 3+3+3+2+3, indicating ['arm_length', 'cabin_size','chasis_size','bucket_size','other_offset'] 
    '''
    model.eval()
    size_keys = ['cabin_size','chasis_size','bucket_size','other_offset']
    pred = {'keypoints':[], 'size':[], 'rotation':[], 'other_offset':[]}  
    tokens = []

    with torch.no_grad():
        for batch_idx, (input, anno) in enumerate(test_loader):
            input = load_dict_to_cuda(input, device)

            result_dict = model(input)

            pred['keypoints'].append(result_dict['keypoints'])  # (b, 4, 3)
            pred['rotation'].append(result_dict['rotation'])  # numpy array  (b, 3, 3)
            pred['size'].append(np.concatenate([result_dict[key] for key in size_keys], axis=1))   #(b, 3+3+2+3)
            pred['other_offset'].append(result_dict['other_offset'])  # (b, 3)
            tokens.extend(anno['token'])
    
    for key, value in pred.items():
        pred[key] = np.concatenate(value)
    
    # calculate boom root from translation and size
    kps = pred['keypoints']  # (n, 4, 3)
    rotation = pred['rotation']  # (n, 3, 3)
    offset = rotation[:, :, 1] * pred['other_offset'][:, 1][:, np.newaxis] + rotation[:, :, 2] * pred['other_offset'][:, 0][:, np.newaxis]
    kps[:,0] += offset

    length = []
    # calculate arm length
    for i in range(3):
        length.append(np.linalg.norm(kps[:,i] - kps[:,i+1], axis=1, keepdims=True))
    length = np.concatenate(length, axis=1)  # (n, 3)
    size = np.concatenate([length, pred['size']], axis=1)  # (n, 3+3+3+2+3)

    # get unique excavator tokens
    id_list = [t[:22] for t in tokens]
    id_array = np.array(id_list)
    size_dict = {}
    all_size = {}
    for uni_token in list(set(id_list)):
        mask = id_array == uni_token
        this_size = size[mask]
        # estimate size for the same excavator by emsembling
        all_size[uni_token] = this_size
        size_dict[uni_token] = filtered_mean(this_size)[0]
    
    for key, value in size_dict.items():
        size_dict[key] = {'arm_length': value[:3],
                        'cabin_size': value[3:6],
                        'chasis_size': value[6:9],
                        'bucket_size': value[9:11],
                        'other_offset': value[11:]}
    
    return size_dict, all_size


def collate_box(batch_dict):
    '''
    Compute box code (x,y,z,dx,dy,dz,theta) in normalized coordinate of excavator
    
    The batch_dict should contain the following keys: cabin_size, chasis_size, other_offset, theta
    '''
    chasis_box_batch = np.zeros((len(batch_dict['cabin_size']), 7))
    cabin_box_batch = np.zeros_like(chasis_box_batch)
    chasis_box_batch[:,2] = - batch_dict['chasis_size'][:,2] / 2
    chasis_box_batch[:,3:6] = batch_dict['chasis_size']
    chasis_box_batch[:,6] = batch_dict['theta']
    cabin_box_batch[:,0] = batch_dict['other_offset'][:,2]  # cabin_x
    cabin_box_batch[:,2] = batch_dict['cabin_size'][:,2] / 2
    cabin_box_batch[:,3:6] = batch_dict['cabin_size']
    return chasis_box_batch, cabin_box_batch


def estimate_labels(model, data_loader, device, logger):
    '''for proxy ground truth generation for the self-supervised training'''
    model.eval()
    pred = {'keypoints':[], 'rotation':[], 'theta':[]}  
    tokens = []
    segment = []
    label_dict = {}

    with torch.no_grad():
        for batch_idx, (input, anno) in enumerate(data_loader):
            input = load_dict_to_cuda(input, device)

            result_dict = model(input)

            for key in pred.keys():
                pred[key].append(result_dict[key])
            
            batch = offset2batch(input['offset']).cpu().numpy()
            for i in range(len(anno['token'])):
                this_segment = result_dict['segment'][batch == i]
                segment.append(this_segment)

            tokens.extend(anno['token'])

    for key, value in pred.items():
        if len(value) > 0:
            pred[key] = np.concatenate(value)
    
    assert len(tokens) == len(pred['keypoints']), 'Token number does not match'
    for i in range(len(tokens)):
        label_dict[tokens[i]] = {'keypoints':pred['keypoints'][i], 'rotation':pred['rotation'][i], 'theta':pred['theta'][i], 'segment':segment[i]}
    
    return label_dict
        
    