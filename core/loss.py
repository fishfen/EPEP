import torch.nn as nn
import torch
import torch.nn.functional as F
import einops
from torch_scatter import scatter
from core.model_utils import vector6d_to_rotation_matrix


class SegmentationLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=-100) -> None:
        super().__init__()
        self.weight = weight
        self.ignore_index = ignore_index
    
    def forward(self, pred, gt):
        '''
        n is the total number of points in a batch.
        pred: (n, s) tensor, predicted segmentation logits, s is the seg class number
        gt: (n) int tensor, seg class of each point
        '''
        # Reshape pred to (n, s) if it's not already in that shape
        if len(pred.shape) > 2:
            n, s = pred.shape[0], pred.shape[-1]
            pred = pred.view(-1, s)   
        gt = gt.view(-1)
        
        # Calculate cross-entropy loss
        loss = F.cross_entropy(
            pred,
            gt,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction='mean'
        )
        
        return loss


class RegressLoss(nn.Module):
    def __init__(self, loss_type, reduction='mean') -> None:
        super().__init__()
        self.reduction = reduction

        if loss_type == 'l1':
            self.criterion = nn.L1Loss(reduction=self.reduction)
        elif loss_type in ['mse', 'l2']:
            self.criterion = nn.MSELoss(reduction=self.reduction)
        elif loss_type == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss(reduction=self.reduction)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def forward(self, pred, gt, mask=None):
        '''
        pred: can be any number of dimensions.
        gt: same shape as the input
        mask: whose value is between 0 and 1'''
        if mask is not None:
            pred = pred * mask
            gt = gt * mask
        loss = self.criterion(pred, gt)

        return loss


class PlanarityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-6 

    def compute_planarity(self, p1, p2, p3, p4):
        # p1, p2, p3, p4 shape: (batch_size, 3)
        v1 = p2 - p1  # shape: (batch_size, 3)
        v2 = p3 - p1  # shape: (batch_size, 3)
        normal = torch.cross(v1, v2)  # shape: (batch_size, 3)
        normal = normal / (torch.norm(normal, dim=1, keepdim=True) + self.epsilon)  # shape: (batch_size, 3)
        
        v4 = p4 - p1  # shape: (batch_size, 3)
        dot_product = torch.sum(normal * v4, dim=1)  # shape: (batch_size,)
        
        return dot_product ** 2  # shape: (batch_size,)

    def forward(self, points):
        # points shape: (batch_size, 4, 3)
        # Consider all possible combinations
        loss = (
            self.compute_planarity(points[:, 0], points[:, 1], points[:, 2], points[:, 3]) +
            self.compute_planarity(points[:, 1], points[:, 2], points[:, 3], points[:, 0]) +
            self.compute_planarity(points[:, 2], points[:, 3], points[:, 0], points[:, 1]) +
            self.compute_planarity(points[:, 3], points[:, 0], points[:, 1], points[:, 2])
        ) / 4.0  # shape: (batch_size,)
        
        return torch.mean(loss)  # scalar


class Points2PlaneLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-6 

    def forward(self, points, rot_mat):
        """
        encourage the last 3 key points to locate on the mid plane defined by translation and rotation

        points: (b, 4, 3) 
        rot_mat: (b, 3, 3)
        """
        # the initial normal of the plane is [0,1,0], heading y, because the excavator heads x
        normal = torch.tensor([0.0, 1.0, 0.0]).unsqueeze(0).to(rot_mat.device)  # (1, 3)
        normal = torch.matmul(rot_mat, normal.t()).transpose(1, 2)   # (b, 1, 3)
        plane_normal = normal / (torch.norm(normal, dim=2, keepdim=True) + self.epsilon)

        vectors1 = points[:,1:] - points[:,:-1]  # (b, 3, 3)
        vectors2 = points[:,2:] - points[:,0].unsqueeze(1)  # (b, 4, 3) -> (b, 2, 3)
        vectors = torch.cat([vectors1, vectors2], dim=1)  # (b, 5, 3)

        # dot product between vectors and plane_normal
        squared_distances = torch.sum(vectors * plane_normal, dim=2) ** 2  # (b, 5)
        loss = torch.mean(squared_distances)  # scalar

        return loss


def get_rotmat_from_z(theta):
    '''compute rotation matrix based on z-axis rotation (theta)'''
    cos_theta = theta[:,0]  # (n,)
    sin_theta = theta[:,1]  # (n,)
    # normalize theta, make sure the output is a rotation matrix
    length = torch.clamp(torch.norm(theta, dim=1), min=1e-6)  # (n,)
    cos_theta = cos_theta / length
    sin_theta = sin_theta / length
    
    Rz = torch.zeros((theta.shape[0], 3, 3), device=theta.device)
    Rz[:,0,0] = cos_theta
    Rz[:,0,1] = -sin_theta
    Rz[:,1,0] = sin_theta
    Rz[:,1,1] = cos_theta
    Rz[:,2,2] = 1
    return Rz


def positional_weight(pos, type, end_range=0.2):
    '''
    pos: (n,) tensor, 1 denotes end, 0 denotes start
    weight: (n,) tensor
    '''
    if type == 'uniform':
        weight = (pos >= 0) & (pos <= 1)
        weight.to(torch.float32)
    elif type == 'two ends':
        weight = torch.zeros_like(pos)

        mask1 = (pos >= 0) & (pos <= end_range)
        b = 1/(end_range**2)  # Makes weight=1 at x=0 and weight=0 at x=0.3
        weight[mask1] = b*(end_range-pos[mask1])**2
        
        mask2 = (pos >= 1-end_range) & (pos <= 1)
        weight[mask2] = b*(pos[mask2]-(1-end_range))**2
    else: 
        raise NotImplementedError
    return weight


def point2line_loss(points, line_starts, line_ends, batch_inds, membership, type, epsilon=1e-6):
    """
    Calculate the distance between points and lines defined by start and end points, this is used 
    by boom and stick self-supervised loss. Assume that N is the total number of points in a batch.
    
    points: torch.Tensor in shape (N, d), N = n1 + n2 + ... + nb, d=2 or 3
    lines_starts: torch.Tensor in shape (B, d), each row is [start_x, start_y, start_z]
    batch_inds: torch.Tensor in shape (N,), indicating batch index for each point
    membership: (N,), indicateing if each point is a member of this part
    type: 'uniform' or 'two ends'
    """
    point_line_starts = line_starts[batch_inds]  # (N, 3)
    point_line_ends = line_ends[batch_inds]      # (N, 3)
    
    line_vecs = point_line_ends - point_line_starts  # (N, 3)
    point_vecs = points - point_line_starts  # (N, 3)
    
    if point_vecs.shape[1] == 2:
        cross_products = point_vecs[:, 0] * line_vecs[:, 1] - point_vecs[:, 1] * line_vecs[:, 0]  # (N,)
        cross_norm = torch.abs(cross_products)
    else:
        cross_products = torch.cross(point_vecs, line_vecs, dim=1)  # (N, 3)
        cross_norm = torch.norm(cross_products, dim=1)

    line_lengths = torch.norm(line_vecs, dim=1).clamp(min=epsilon)  # (N,)
    distances = (cross_norm / line_lengths)  # (N,)
    
    projection = torch.sum(point_vecs * line_vecs, dim=1) / line_lengths  # (N,) dot product
    pos_weight = positional_weight(projection/line_lengths, type)
    mask = pos_weight * membership

    loss = torch.sum(distances*mask) / torch.clamp(torch.sum(mask), min=1)
    
    return loss


def bucket_loss(proj_points, proj_kps_, bucket_size, batch, membership, device, 
                bucket_lambda=(0.4, 0.2)):
    '''Calculate the bucket loss, based on the distance between the centroids'''

    lambda1, lambda2 = bucket_lambda  # the ratio of (mid-point/end-point) from start point
    bucket_mid = (1-lambda1) * proj_kps_[:,2] + lambda1 * proj_kps_[:,3]
    y_vector = torch.tensor([0.0, 1.0, 0.0]).to(device)
    v_ = torch.cross((proj_kps_[:,3] - proj_kps_[:,2]), y_vector.unsqueeze(0), dim=1)   # (b, 3)
    v_ = v_ / (torch.norm(v_, dim=1, keepdim=True) + 1e-6) * bucket_size[:,0].unsqueeze(-1)   # (b, 3) * (b,1)
    # This is the approximated centroid of the bucket
    bucket_anchor = bucket_mid + lambda2 * v_  # (b,3)

    # threshold the membership less than 0.3 to be 0
    membership = membership * (membership > 0.3).to(torch.float32)
    batch_membership_sum = scatter(membership, batch, dim=0, reduce='sum')  # (b,)
    
    # identify the valid batch (membership sum > 1)
    valid_mask = batch_membership_sum > 1.0  # (b,)
    
    if not torch.any(valid_mask):
        # if no valid batch, return 0 loss
        return torch.tensor(0.0, device=device)
    
    # only calculate centroid for valid batch
    bucket_centroid = scatter(proj_points[:,[0,2]] * membership.unsqueeze(-1), 
                              batch, dim=0, reduce='sum')  # (b,2)
    bucket_centroid = bucket_centroid / batch_membership_sum.unsqueeze(-1).clamp(min=1e-6)  # (b,2)
    
    # only calculate loss for valid batch
    valid_centroids = bucket_centroid[valid_mask]  # (valid_b,2)
    valid_anchors = bucket_anchor[valid_mask][:,[0,2]]  # (valid_b,2)
    
    loss = torch.mean(torch.norm(valid_centroids - valid_anchors, dim=1))

    return loss


def point_to_cuboid_loss(points, sizes, membership, ignore_z=False, q=2):
    """
    Calculate distances between batched points and axis-aligned cuboids
    
    points: torch.Tensor of shape (N, 3), should be transformed to normalized frame
    cuboids: torch.Tensor of shape (N, 3) 3d dimensions 
    membership: torch.Tensor of shape (N,) indicating if each point is a member of this part
    ignore_z: bool, whether to ignore the distance to z face
    q: float, weight for outside distance
    """
    # Convert dimensions to half-sizes
    half_sizes = sizes / 2  # (N, 3)
    abs_points = torch.abs(points)  # (N, 3)
    normed_sizes = torch.ones_like(half_sizes)  # (N, 3)
    scaled_points = abs_points / half_sizes  # (N, 3)
    
    # Distance when point is inside cuboid
    if ignore_z:
        inside_dists = torch.min(normed_sizes[:,:2] - scaled_points[:,:2], dim=1)[0]
    else:
        inside_dists = torch.min(normed_sizes - scaled_points, dim=1)[0]
    inside_term = torch.clamp(inside_dists, min=0)  # (n,) use absolute distance
    
    # Distance when point is outside cuboid
    outside_dists = torch.clamp(abs_points - half_sizes, min=0)
    outside_term = torch.norm(outside_dists, dim=1)  # (n,)
    # outside_term = torch.sum(outside_dists**2, dim=1)  # mse loss, deprecated
    
    distance = inside_term + q * outside_term   # (n,)
    loss = torch.sum(distance * membership) / torch.clamp(torch.sum(membership), min=1)

    return loss


def compute_ms_cuboid(points, sizes, ignore_z=False, condition_inside=False, in_thres=0.2, ext_thres=0.1):
    '''
    compute the membership of points, determined by the distance to the cuboid.

    parameters:
        points: (n, 3) tensor
        sizes: (n, 3) tensor
        ignore_z: bool, whether to ignore the distance to z face
        condition_inside: bool, if true, the inside points has to be conditioned to be a member
    return:
        membership: (n,)
    '''
    half_sizes = sizes / 2  # (N, 3)
    abs_points = torch.abs(points)  # (N, 3)
    normed_sizes = torch.ones_like(half_sizes)  # (N, 3)
    scaled_points = abs_points / half_sizes  # (N, 3)
    
    
    # Distance when point is outside cuboid
    outside_dists = torch.clamp(abs_points - half_sizes, min=0)
    outside_term = torch.norm(outside_dists, dim=1)  # (n,)
    
    if condition_inside:
        # Distance when point is inside cuboid
        if ignore_z:
            inside_dists = torch.min(normed_sizes[:,:2] - scaled_points[:,:2], dim=1)[0]
        else:
            inside_dists = torch.min(normed_sizes - scaled_points, dim=1)[0]
        inside_term = torch.clamp(inside_dists, min=0)  # (n,) use absolute distance
        membership = (inside_term < in_thres) & (outside_term < ext_thres)
    else:
        membership = outside_term < ext_thres
    membership = membership.to(torch.float32)
    return membership


def compute_ms_link(points, start, end, size3d, y_vector, extra_depth=0.3):
    '''
    compute the membership of points inside a cuboid defined by start, end and size3d.
    local x-axis of the cuboid is parallel to the line defined by start and end.
    local y-axis is parallel to the y_vector.

    parameters:
        points: (n, 3) tensor
        start: (n, 3) tensor
        end: (n, 3) tensor
        size3d: (n, 3) tensor, (x, y, z) x is size along the line, y is size along y_vector, 
        z is size along z
    returns:
        membership: (n,) tensor
    '''
    # compute the local coordinate system
    x_axis = end - start  
    x_axis = x_axis / torch.norm(x_axis, dim=1, keepdim=True)
    
    # make sure y_vector is orthogonal to x_axis
    y_axis = y_vector - torch.sum(y_vector * x_axis, dim=1, keepdim=True) * x_axis
    y_axis = y_axis / torch.norm(y_axis, dim=1, keepdim=True)  # (n, 3)
    
    z_axis = torch.cross(x_axis, y_axis)
    
    relative_points = points - start
    
    # calculate the projection length on each axis
    proj_x = torch.sum(relative_points * x_axis, dim=1)
    proj_y = torch.sum(relative_points * y_axis, dim=1)
    proj_z = torch.sum(relative_points * z_axis, dim=1)
    
    inside_x = (proj_x >= 0) & (proj_x <= size3d[..., 0])
    inside_y = (proj_y >= -size3d[..., 1]/2) & (proj_y <= size3d[..., 1]/2)
    inside_z = (proj_z >= -extra_depth) & (proj_z <= size3d[..., 2])
    
    membership = inside_x & inside_y & inside_z
    
    return membership.to(torch.float32)


def compute_ms_arm(proj_points, start, end, batch, name='boom', extra_depth=0.3, extra_length=0.3):
    bs = len(start)
    length = torch.norm(end - start, dim=1, keepdim=True) # (bs, 1)
    ed = extra_depth
    el = extra_length

    if name == 'boom':
        size_yz = torch.tensor([0.8, 1.2])
    elif name == 'stick':
        size_yz = torch.tensor([0.5, 1.0])
    elif name == 'bucket':
        size_yz = torch.tensor([1.5, 1.5])
        length = length + extra_length
        ed = 0.4
    else:
        raise NotImplementedError
    size_yz = size_yz.repeat(bs,1).to(proj_points.device)
    size3d = torch.cat([length, size_yz], dim=1)  # (bs, 3)

    y_vector = torch.tensor([0.0, 1.0, 0.0]).to(proj_points.device).unsqueeze(0)  # (1, 3)

    return compute_ms_link(proj_points, start[batch], end[batch], size3d[batch], 
                    y_vector, extra_depth=ed)
   

class SupervisedLoss(nn.Module):

    def __init__(self, loss_info_dict, config, device) -> None:
        super().__init__()
        self.loss_info = loss_info_dict
        self.cfg = config
        self.device = device   # model.device
        self.regress_names = ['rotation', 'theta', 
                              'cabin_size', 'chasis_size', 'bucket_size', 'other_offset']
        self.seg_loss = SegmentationLoss()
        self.planar_loss = PlanarityLoss()
        self.p2plane_loss = Points2PlaneLoss()
    
    def add_loss(self, this_loss, loss_name):
        self.loss_record[loss_name] = this_loss.item()
        self.loss += this_loss * self.loss_info[loss_name]['weight']
    
    def forward(self, pred_dict, target_dict, *args, **kwargs):
        self.loss  = 0.0
        self.loss_record = {}

        # direct regression loss
        for key in self.regress_names:
            this_loss_info = self.loss_info[key]
            criterion = RegressLoss(this_loss_info['loss_type'], this_loss_info.get('reduction', 'mean'))
            this_loss = criterion(pred_dict[key], target_dict[key])
            self.add_loss(this_loss, key)
        
        # key points loss
        this_loss_info = self.loss_info['keypoints']
        criterion = RegressLoss(this_loss_info['loss_type'], this_loss_info.get('reduction', 'mean'))
        kps_loss = criterion(pred_dict['keypoints'], target_dict['keypoints'])
        self.add_loss(kps_loss, 'keypoints')

        kps = einops.rearrange(pred_dict['keypoints'], 'b (k d) -> b k d', d=3) # (b, 4, 3)
        # replace the first kp with the root of boom
        rot_mat = vector6d_to_rotation_matrix(pred_dict['rotation'])
        v_z = rot_mat[:,:,2] * pred_dict['other_offset'][:,0].unsqueeze(-1)  
        v_y = rot_mat[:,:,1] * pred_dict['other_offset'][:,1].unsqueeze(-1)
        kps_ = kps.clone()
        kps_[:,0] = kps[:,0] + v_z + v_y  

        # point-to-plane loss
        self.add_loss(self.p2plane_loss(kps_, rot_mat), 'point2plane')

        # 4 points planarity loss
        self.add_loss(self.planar_loss(kps_), 'planarity')

        # segmentation loss
        self.add_loss(self.seg_loss(pred_dict['segment'], target_dict['segment']), 'segment')

        # point-wise loss
        for key in ['heatmap', 'vector']:
            this_loss_info = self.loss_info[key]
            criterion = RegressLoss(this_loss_info['loss_type'], this_loss_info.get('reduction', 'mean'))
            if key == 'vector':
                mask = target_dict['heatmap'].repeat_interleave(3, dim=1)  # (n,k) -> (n,kx3)
                mask = (mask > self.cfg.hm_thre).float()
                this_loss = criterion(pred_dict[key], target_dict[key], mask)
            else:
                this_loss = criterion(pred_dict[key], target_dict[key])
            self.add_loss(this_loss, key)
        
        self.loss_record['total'] = self.loss.item()

        return self.loss, self.loss_record


class SelfSuperLoss(nn.Module):

    def __init__(self, loss_info_dict, config, device) -> None:
        super().__init__()
        self.loss_info = loss_info_dict
        self.cfg = config
        self.device = device   # model.device
        self.seg_loss = SegmentationLoss()
        self.planar_loss = PlanarityLoss()
        self.p2plane_loss = Points2PlaneLoss()
        self.epsilon = 1e-6
    
    def add_loss(self, this_loss, loss_name):
        self.loss_record[loss_name] = this_loss.item()
        self.loss += this_loss * self.loss_info[loss_name]['weight']
        
    
    def forward(self, pred_dict, target_dict, points, batch):
        '''
        points: (N, 3)
        batch: (N,)
        '''
        self.loss  = 0.0
        self.loss_record = {}

        # process before calculating loss
        kps = einops.rearrange(pred_dict['keypoints'], 'b (k d) -> b k d', d=3)
        rot_mat = vector6d_to_rotation_matrix(target_dict['rotation'])
        translation = kps[:,0]  # (b, 3)

        # replace the first kp with the root of boom
        v_z = rot_mat[:,:,2] * pred_dict['other_offset'][:,0].unsqueeze(-1)  # carefully think, use pred or target
        v_y = rot_mat[:,:,1] * pred_dict['other_offset'][:,1].unsqueeze(-1)
        kps_ = kps.clone()
        kps_[:,0] = kps[:,0] + v_y + v_z  

        diff = kps_[:, 1:] - kps_[:, :-1]  # (n, 3, 3)
        pred_dict['arm_length'] = torch.norm(diff, dim=2)  # (n, 3)
        ms = F.softmax(pred_dict['segment'], dim=-1)  # (n,6) membership matrix
        ms = ms.detach()  # detach the membership matrix

        # size loss trying to fix size
        regress_names = ['arm_length', 'cabin_size', 'chasis_size', 
                         'bucket_size', 'other_offset', 'rotation']
        for key in regress_names:
            this_loss_info = self.loss_info[key]
            criterion = RegressLoss(this_loss_info['loss_type'], this_loss_info.get('reduction', 'mean'))
            this_loss = criterion(pred_dict[key], target_dict[key])
            self.add_loss(this_loss, key)

        # point-to-plane loss
        self.add_loss(self.p2plane_loss(kps_, rot_mat), 'point2plane')

        # 4 points planarity loss
        self.add_loss(self.planar_loss(kps_), 'planarity')

        # apply reverse rotation, (R^T)^-1=R, note points in row require right mul with R^T rather than R
        proj_points = torch.einsum('ni,nij->nj', points - translation[batch], rot_mat[batch])  # (n,3) * (n,3,3) -> (n,3)
        proj_kps_ = torch.einsum('bni,bij->bnj', kps_ - translation.unsqueeze(1), rot_mat)  # (b, 4, 3) * (b, 3, 3)

        cache_dict = self.add_p2p_loss(proj_points, proj_kps_, pred_dict, target_dict, ms, batch)

        new_ms = self.update_ms(proj_points, proj_kps_, cache_dict, target_dict, batch)
        # segment loss
        self.add_loss(self.seg_loss(pred_dict['segment'], new_ms), 'segment')

        return self.loss, self.loss_record
    

    def add_p2p_loss(self, proj_points, proj_kps_, pred_dict, target_dict, ms, batch):
        '''  point-to-primitive loss '''

        # boom loss, only need the projected x and z
        boom_loss = point2line_loss(proj_points[:,[0,2]], proj_kps_[:,0,[0,2]], 
                                    proj_kps_[:,1,[0,2]], batch, ms[:,1], 'two ends')  
        self.add_loss(boom_loss, 'p2p_boom')
        
        # stick loss
        stick_loss = point2line_loss(proj_points[:,[0,2]], proj_kps_[:,1,[0,2]], 
                                     proj_kps_[:,2,[0,2]], batch, ms[:,2], 'uniform')  
        self.add_loss(stick_loss, 'p2p_stick')

        # bucket loss
        bk_loss = bucket_loss(proj_points, proj_kps_, target_dict['bucket_size'], 
                              batch, ms[:,3], self.device, self.cfg.bucket_lambda)
        self.add_loss(bk_loss, 'p2p_bucket')

        # cabin loss
        proj_points_cab = proj_points.clone()
        # here pred_dict is used because we want network adjusted it, the target may not be accurate
        proj_points_cab[:,2] -= target_dict['cabin_size'][:,2][batch] / 2
        proj_points_cab[:,0] -= pred_dict['other_offset'][:,2][batch]  # cabin_x
        cb_loss = point_to_cuboid_loss(proj_points_cab, target_dict['cabin_size'][batch], ms[:,4], 
                                       ignore_z=self.cfg.cabin_ignore_z, q=self.cfg.cabin_q)
        self.add_loss(cb_loss, 'p2p_cabin')

        # chassis loss
        rotmat_z = get_rotmat_from_z(pred_dict['theta'])  # (b,3,3) 
        # chassis has z offset of -dz/2 relative to the overall translation
        proj_points_chassis = torch.einsum('ni,nij->nj', proj_points, rotmat_z[batch])  # (n,3) * (n,3,3) -> (n,3)
        proj_points_chassis[:,2] += target_dict['chasis_size'][:,2][batch] / 2
        chas_loss = point_to_cuboid_loss(proj_points_chassis, target_dict['chasis_size'][batch], ms[:,5], 
                                         ignore_z=self.cfg.chasis_ignore_z, q=self.cfg.chasis_q)
        self.add_loss(chas_loss, 'p2p_chassis')

        return {'proj_points_cab': proj_points_cab, 'proj_points_chassis': proj_points_chassis}

    @torch.no_grad()
    def update_ms(self, proj_points, proj_kps_, cache_dict, target_dict, batch):

        new_ms_cabin = compute_ms_cuboid(cache_dict['proj_points_cab'], 
                                         target_dict['cabin_size'][batch], 
                                         ignore_z=self.cfg.cabin_ignore_z)
        new_ms_chassis = compute_ms_cuboid(cache_dict['proj_points_chassis'], 
                                           target_dict['chasis_size'][batch], 
                                           ignore_z=self.cfg.chasis_ignore_z)
        new_ms_boom = compute_ms_arm(proj_points, proj_kps_[:,0], proj_kps_[:,1], batch, name='boom')
        new_ms_stick = compute_ms_arm(proj_points, proj_kps_[:,1], proj_kps_[:,2], batch, name='stick')
        new_ms_bucket = compute_ms_arm(proj_points, proj_kps_[:,2], proj_kps_[:,3], batch, name='bucket')
        new_ms_bg = torch.zeros_like(new_ms_cabin)
        new_ms = torch.stack([new_ms_bg, new_ms_boom, new_ms_stick, new_ms_bucket, 
                              new_ms_cabin, new_ms_chassis], dim=1)  # (n, 6)
        # argmax can 1) resolve duplicated membership 2) determine the background of index 0
        new_ms = torch.argmax(new_ms, dim=1)  # (n,)

        old_ms = target_dict['segment']
        updated_ms = old_ms.clone()
        # updated_ms[(old_ms == 1) & (new_ms == 0)] = 0
        # updated_ms[(old_ms == 2) & (new_ms == 0)] = 0
        updated_ms[(old_ms == 3) & (new_ms == 0)] = 0  # correct the bucket membership
        # updated_ms[(old_ms == 4) & (new_ms == 0)] = 0
        updated_ms[(old_ms == 5) & (new_ms == 0)] = 0  # correct the chassis membership

        return updated_ms
