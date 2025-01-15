import torch.nn as nn
import torch
import einops
from torch_scatter.composite import scatter_softmax
from torch_scatter import scatter
from core.pt_network import PointTransformerV2, PointBatchNorm
from core.loss import SupervisedLoss, SelfSuperLoss
from core.model_utils import generate_heatmap_and_vector_batch, vector6d_to_rotation_matrix
from utils.utils import offset2batch


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(2, 1)
    
    def forward(self, x):
        avg_out = x.mean(dim=-1, keepdim=True)  # (n, 1)
        max_out = x.max(dim=-1, keepdim=True).values  # (n, 1)

        f_cat = torch.cat([avg_out, max_out], dim=-1)  # (n, 2)
        attn = self.sigmoid(self.linear(f_cat))
        return x * attn


class SingleHead(nn.Module):

    def __init__(self, num_layers, head_channels, out_channels, is_spatial_attn=False):
        super().__init__()
        layer_list = []
        for k in range(num_layers - 1):
            layer_list.append(nn.Sequential(
                nn.Linear(head_channels, head_channels),
                PointBatchNorm(head_channels),
                nn.ReLU(inplace=True),
            ))
        self.point_wise_mlp = nn.Sequential(*layer_list)
        self.out_mlp = nn.Linear(2*head_channels, out_channels)
        self.is_spatial_attn = is_spatial_attn
        self.spatial_attn = SpatialAttention()
    
    def forward(self, feat, offset):
        out = self.point_wise_mlp(feat)
        if self.is_spatial_attn:
            out = self.spatial_attn(out)
        batch = offset2batch(offset)
        out_ave = scatter(out, batch.clone(), dim=-2, reduce='max')
        out_max = scatter(out, batch.clone(), dim=-2, reduce='mean')
        out = torch.concat([out_ave, out_max], dim=-1)
        out = self.out_mlp(out)
        return out


class PoseEstimator(nn.Module):
    def __init__(self, in_channels, cfg, device, train_mode=None) -> None:
        super().__init__()
        self.cfg = cfg
        cfg1 = cfg.backbone
        self.backbone = PointTransformerV2(in_channels=in_channels,
                                           patch_embed_depth=cfg1.patch_embed_depth,
                                           patch_embed_channels=cfg1.patch_embed_channels,
                                           patch_embed_groups=cfg1.patch_embed_groups,
                                           patch_embed_neighbours=cfg1.patch_embed_neighbours,
                                           enc_depths=cfg1.enc_depths,
                                           enc_channels=cfg1.enc_channels,
                                           enc_groups=cfg1.enc_groups,
                                           enc_neighbours=cfg1.enc_neighbours,
                                           dec_depths=cfg1.dec_depths,
                                           dec_channels=cfg1.dec_channels,
                                           dec_groups=cfg1.dec_groups,
                                           dec_neighbours=cfg1.dec_neighbours,
                                           grid_sizes=cfg1.grid_sizes,
                                           attn_drop_rate=cfg1.attn_drop_rate,
                                           drop_path_rate=cfg1.drop_path_rate,
                                           global_pool_method=cfg1.global_pool_method)
        
        self.direct_head_dict = cfg.direct_regress
        self.point_head_dict = cfg.point_wise_predict

        head_channels = cfg.backbone.enc_channels[-1]
        # this head directly regresses pose variables, following encoder
        self.make_direct_head(head_channels)
        # this head yields point-wise prediction, following decoder
        self.make_point_head(cfg.backbone.dec_channels[0])
        self.shared_mlp = nn.Sequential(nn.Linear(head_channels, head_channels),
                                        PointBatchNorm(head_channels),
                                        nn.ReLU(inplace=True))

        self.device = device
        self.train_mode = train_mode
        if train_mode is not None:
            if train_mode == 'supervised':
                self.loss_func = SupervisedLoss(self.cfg.loss_info, cfg, self.device)
            elif train_mode == 'self-supervised':
                self.loss_func = SelfSuperLoss(self.cfg.loss_info, cfg.loss_cfg, self.device)
            else:
                raise NotImplementedError

        self.predict_dict = {}
    
    def make_direct_head(self, head_channels):
        for cur_name in self.direct_head_dict:
            out_channels = self.direct_head_dict[cur_name]['out_channels']
            num_layers = self.direct_head_dict[cur_name]['num_layer']
            headnet = SingleHead(num_layers, head_channels, out_channels, self.cfg.spatial_attention)

            self.__setattr__(cur_name, headnet)
    
    def make_point_head(self, head_channels):
        for cur_name in self.point_head_dict:
            out_channels = self.point_head_dict[cur_name]['out_channels']
            num_layers = self.point_head_dict[cur_name]['num_layer']

            layer_list = []
            for k in range(num_layers - 1):
                layer_list.append(nn.Sequential(
                    nn.Linear(head_channels, head_channels),
                    PointBatchNorm(head_channels),
                    nn.ReLU(inplace=True),
                ))
            layer_list.append(nn.Linear(head_channels, out_channels))
            headnet = nn.Sequential(*layer_list)

            # register the headnet in a class attribute
            self.__setattr__(cur_name, headnet)
    
    
    def estimate_kps_from_heatmap(self, coord, heatmap, vector, batch, mask=None):
        '''
        coord: (n, 3)
        heatmap: (n, k)
        vector: (n, kx3)  unit vector field
        batch: (n,)  n=n_1 + n_2 + .. + n_B
        mask: (n,) 1 is body, 0 is background
        TODO whether use top-k 
        '''
        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(heatmap)  # (n,) -> (n, k)
            large_negative = -1e4  # You can adjust this value
            heatmap = heatmap + (1 - mask) * large_negative

        kp_offset = einops.rearrange(vector, 'n (k d) -> n k d', d=3)  # (n, k, 3)
        kp_offset = coord.unsqueeze(1) + kp_offset  # (n, 3) + (n, k, 3) -> (n, k, 3)

        # apply sigmoid to heatmap
        heatmap = torch.sigmoid(heatmap)  # (n, k)
        temp = einops.einsum(kp_offset, heatmap, 'n k d, n k -> n k d')
        kps = scatter(temp, batch, dim=0, reduce='sum')   # (n, k, 3) -> (B, k, 3)
        weight = scatter(heatmap, batch, dim=0, reduce='sum')  # (n, k) -> (B, k)
        weight = weight.clamp(min=1e-6)
        kps = kps / weight.unsqueeze(-1)  # (B, k, 3)
        kps = einops.rearrange(kps, 'b k d -> b (k d)')
        return kps  # (B, kx3)
    
    def pointwise_postprocessing(self, pointwise_dict, anno, coord, batch):

        if False:
            if self.training:
                body_mask = (anno['segment'] > 0).float()
            else:
                body_mask = (pointwise_dict['segment'].argmax(dim=1) > 0).float()
        else:
            body_mask = None
        # esitmate key points based on the predicted heatmap and offset field
        kps = self.estimate_kps_from_heatmap(coord, pointwise_dict['heatmap'], pointwise_dict['vector'], batch, body_mask)
        result = {'keypoints': kps}
        return result

    
    def assign_target(self, input, anno):
        
        batch = offset2batch(input['offset'])
        
        target_dict = {}
        for key in anno:
            gt = anno[key]
            if key == 'keypoints':
                gt = einops.rearrange(gt, 'b k d -> b (k d)')
            elif key == 'rotation':
                # use the first 2 columns of rotation matrix for 6d representation
                gt = einops.rearrange(gt[:,:,:2], 'b d1 d2 -> b (d2 d1)')  
            elif key == 'theta':
                if gt.dim() == 1:
                    gt = gt.view(-1,1)
                gt = torch.hstack([torch.cos(gt), torch.sin(gt)])  # transform theta into cos and sin
            target_dict[key] = gt
        
        if self.train_mode == 'supervised':
            hm_target, vc_target = generate_heatmap_and_vector_batch(input['coord'], anno['keypoints'], 
                                            batch, self.cfg.hm_sigma)
            target_dict['heatmap'] = hm_target
            target_dict['vector'] = einops.rearrange(vc_target, 'n k d -> n (k d)')
        self.target_dict = target_dict
        return target_dict
    
    def forward(self, input, anno=None):
        # input: {'coord', 'feat', 'offset'}
        out1_dict, out2 = self.backbone(input)   # (B, c), (n, c)
        direct_dict = {}
        pointwise_dict = {}
        pred_dict = {}

        _, out1, offset1 = out1_dict
        out1 = self.shared_mlp(out1)
        for cur_name in self.direct_head_dict:
            direct_dict[cur_name] = self.__getattr__(cur_name)(out1, offset1)   

        for cur_name in self.point_head_dict:
            pointwise_dict[cur_name] = self.__getattr__(cur_name)(out2)
        
        pred_dict.update(direct_dict)
        pred_dict.update(pointwise_dict)

        coord = input['coord']
        batch = offset2batch(input['offset']).clone()

        post_pw_dict = self.pointwise_postprocessing(pointwise_dict, anno, coord, batch)
        pred_dict.update(post_pw_dict)
        self.predict_dict = pred_dict
        self.input_dict = input
        self.input_dict['batch'] = batch

        if self.training:
            self.assign_target(input, anno)
            loss, loss_dict = self.get_loss()
            return loss, loss_dict, pred_dict
        else:
            result_dict = self.post_processing(to_numpy=True)
            return result_dict
    
    def get_loss(self):
        loss, loss_dict = self.loss_func(self.predict_dict, self.target_dict,
                                         points=self.input_dict['coord'], 
                                         batch=self.input_dict['batch'])            
        return loss, loss_dict

    def post_processing(self, result_dict=None, to_numpy=False):
        if result_dict is None:
            result_dict = self.predict_dict
        
        out_dict = {}
        for key in result_dict:
            pred = result_dict[key]
            if key == 'keypoints':
                pred = einops.rearrange(pred, 'b (k d) -> b k d', d=3)
            elif key == 'theta':
                pred = torch.atan2(pred[:,1], pred[:,0])
            elif key =='segment':
                pred = pred.argmax(dim=1)   # logits to index
            elif key == 'rotation':
                pred = vector6d_to_rotation_matrix(pred)  

            if to_numpy:
                if isinstance(pred, torch.Tensor):
                    if pred.is_cuda:
                        pred = pred.cpu()
                    pred = pred.numpy()
            out_dict[key] = pred

        return out_dict



