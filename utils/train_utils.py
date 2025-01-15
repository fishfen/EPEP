import os
import numpy as np
import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from utils.metrics import compute_mpjpe, compute_seg_metrics


def train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch, writer, logger, counter):
    model.train()
    scheduler_after_batch = True if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR) else False
    for batch_idx, (input, anno) in enumerate(train_loader):
        input = load_dict_to_cuda(input, device)
        anno = load_dict_to_cuda(anno, device)
        optimizer.zero_grad()

        loss, loss_dict, pred = model(input, anno)

        loss.backward()
        optimizer.step()
        counter.step()
        if scheduler_after_batch:
            scheduler.step()

        iter_count = counter.iter
        if iter_count % 5 == 0:
            for key, value in loss_dict.items():
                writer.add_scalar(f'Loss/{key}', value, iter_count)

        if batch_idx % 100 == 0:
            logger.info(f'Train Epoch: {epoch+1} [{batch_idx}/{len(train_loader)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        if iter_count % 1000 == 0:
            gpu_info = os.popen('gpustat').read()  # please install gpustat through "pip install gpustat"
            logger.info(gpu_info)

    if not scheduler_after_batch:
        scheduler.step()
    writer.add_scalar(f'Learning rate', scheduler.get_last_lr()[0], iter_count)


def validate(model, val_loader, device, writer, logger, counter):
    model.eval()
    pred = {'rotation':[], 'keypoints':[], 'theta':[]}  
    seg_pred = []
    gt = {'keypoints':[], 'rotation':[], 'theta':[]}
    seg_gt = []
    loss_dict_sum = None
    has_segment = False
    train_mode = model.train_mode

    with torch.no_grad():
        for batch_idx, (input, anno) in enumerate(val_loader):
            input = load_dict_to_cuda(input, device)
            anno_cuda = load_dict_to_cuda(anno, device)

            result_dict = model(input)   # evalution mode

            if train_mode == 'supervised':
                model.assign_target(input, anno_cuda)
                _, loss_dict = model.get_loss()
                if loss_dict_sum is None:
                    loss_dict_sum = {key: 0.0 for key in loss_dict.keys()}
                for key, value in loss_dict.items():
                        loss_dict_sum[key] += value

            pred['rotation'].append(result_dict['rotation'])  # numpy array  (b, 3, 3)
            pred['keypoints'].append(result_dict['keypoints'])
            pred['theta'].append(result_dict['theta'])

            gt['keypoints'].append(anno['keypoints'].numpy())  # tensor
            gt['rotation'].append(anno['rotation'].numpy()) 
            gt['theta'].append(anno['theta'].numpy())
            if 'segment' in anno:
                seg_pred.append(result_dict['segment'])   # (n,) segmentation index
                seg_gt.append(anno['segment'].numpy())
                has_segment = True

    for key, value in pred.items():
        pred[key] = np.concatenate(value)
    
    for key, value in gt.items():
        gt[key] = np.concatenate(value)

    error_trans2 = compute_mpjpe(pred['keypoints'][:,0,:], gt['keypoints'][:,0,:])
    error_kps2 = compute_mpjpe(pred['keypoints'][:,1:,:], gt['keypoints'][:,1:,:])
    angle_error = np.degrees(np.mean(np.abs(pred['theta'] - gt['theta'])))

    iter_count = counter.iter
    if loss_dict_sum is not None:
        for key, value in loss_dict_sum.items():
            writer.add_scalar(f'Val loss/{key}', value/len(val_loader), iter_count)
    writer.add_scalar(f'Validation/keypoints pointwise', error_kps2, iter_count)
    writer.add_scalar(f'Validation/translation pointwise', error_trans2, iter_count)
    writer.add_scalar(f'Validation/Angle error', angle_error, iter_count)
    if has_segment:
        seg_metrics = compute_seg_metrics(seg_pred, seg_gt, num_classes=model.cfg.point_wise_predict['segment']['out_channels'])
        writer.add_scalar(f'Validation/Segmentation mIOU', seg_metrics['Mean IoU'], iter_count)

    logger.info(f'Validation Set - key point error: {error_kps2:.4f}, translation error: {error_trans2:.4f}')
    return error_kps2 + error_trans2


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    print(f"Checkpoint saved: {filename}")


def load_checkpoint(filename):
    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        return checkpoint
    else:
        print(f"No checkpoint found at '{filename}'")
        return None


def load_dict_to_cuda(dict, device):
    output_dict = {}
    for key, val in dict.items():
        # if isinstance(val, np.ndarray):
        #     output_dict[key] = torch.from_numpy(val).float().to(device)
        if isinstance(val, torch.Tensor):
            output_dict[key] = val.to(device)
        else:
            output_dict[key] = val
    return output_dict


def build_optimizer(model, optim_cfg):
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    if optim_cfg.optimizer == 'adam':
        optimizer = optim.Adam(trainable_params, lr=optim_cfg.learning_rate, weight_decay=optim_cfg.weight_decay)
    elif optim_cfg.optimizer == 'sgd':
        optimizer = optim.SGD(
            trainable_params, lr=optim_cfg.learning_rate, weight_decay=optim_cfg.weight_decay,
            momentum=optim_cfg.momentum
        )
    else:
        raise NotImplementedError

    return optimizer


def build_scheduler(optimizer, cfg, epoch, steps_per_epoch):
    scheduler_type = cfg.scheduler.lower()
    
    if scheduler_type == 'lambdalr':
        def lr_lambda(epoch):
            return cfg.lambda_func(epoch)
        return lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    elif scheduler_type == 'steplr':
        return lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.step_size,
            gamma=cfg.gamma
        )
    
    elif scheduler_type == 'multisteplr':
        return lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.milestones,
            gamma=cfg.gamma
        )
    
    elif scheduler_type == 'exponentiallr':
        return lr_scheduler.ExponentialLR(
            optimizer,
            gamma=cfg.gamma
        )
    
    elif scheduler_type == 'cosineannealinglr':
        return lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.T_max,
            eta_min=cfg.eta_min if hasattr(cfg, 'eta_min') else 0
        )
    
    elif scheduler_type == 'onecyclelr':
        return lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.max_lr,
            total_steps=epoch * steps_per_epoch,
            pct_start=cfg.pct_start if hasattr(cfg, 'pct_start') else 0.3,
            anneal_strategy=cfg.anneal_strategy if hasattr(cfg, 'anneal_strategy') else 'cos',
            cycle_momentum=cfg.cycle_momentum if hasattr(cfg, 'cycle_momentum') else True,
            base_momentum=cfg.base_momentum if hasattr(cfg, 'base_momentum') else 0.85,
            max_momentum=cfg.max_momentum if hasattr(cfg, 'max_momentum') else 0.95,
            div_factor=cfg.div_factor if hasattr(cfg, 'div_factor') else 25.0,
            final_div_factor=cfg.final_div_factor if hasattr(cfg, 'final_div_factor') else 1e4,
        )
    
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")