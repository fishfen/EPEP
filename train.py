import os
from easydict import EasyDict
from pathlib import Path
import glob
import shutil
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from core.model import PoseEstimator
from core.dataset import build_dataloader
from utils.utils import setup_logger, Iter_counter, load_config
from utils.train_utils import train_one_epoch, validate
from utils.train_utils import load_checkpoint, save_checkpoint, build_optimizer, build_scheduler



def main(cfg_path, out_path, pretrained_path=None, load_ckpt=False):

    yaml_config = load_config(cfg_path)
    config = EasyDict(yaml_config)
    assert config.train_mode in ['supervised', 'self-supervised']
    train_mode = config.train_mode

    result_path = out_path / config.experiment_name
    if not os.path.exists(result_path):
        os.makedirs(result_path, exist_ok=True)
    ckpt_path = result_path / 'ckpt'
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path, exist_ok=True)

    shutil.copy2(cfg_path, result_path)
    shutil.copy2(config.base_config, result_path)

    writer = SummaryWriter(result_path / 'tb')
    logger = setup_logger(result_path, config.model_name)

    start_epoch = 0
    best_val = 1e3

    data_path = Path(config.dataset.data_path)
    train_loader = build_dataloader(config=config, mode='train', data_path=data_path, 
                                    logger=logger, batchsize=config.optimization.batchsize)
    val_loader = build_dataloader(config=config, mode='val', data_path=data_path, 
                                  logger=logger, batchsize=config.optimization.batchsize)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PoseEstimator(in_channels=3, cfg=config.model, 
                          device=device, train_mode=train_mode).to(device)

    if load_ckpt:
        # Resume training/fine-tuning from checkpoint
        ckpt_list = glob.glob(str(ckpt_path / '*.pth*'))
        assert len(ckpt_list) > 0, 'checkpoint does not exist'
        ckpt_list.sort(key=os.path.getmtime)
        checkpoint = load_checkpoint(ckpt_list[-1])
        model.load_state_dict(checkpoint['state_dict'])
        logger.info(f"Loaded checkpoint from {ckpt_list[-1]}")
    elif train_mode == 'self-supervised':
        # Start new fine-tuning from pretrained model
        pretrained_path = config.pretrain if pretrained_path is None else pretrained_path
        assert pretrained_path is not None, "Pretrained model path must be provided for new fine-tuning"
        logger.info(f"Loading pretrained model from {pretrained_path}")
        pretrained_ckpt = load_checkpoint(pretrained_path)
        model.load_state_dict(pretrained_ckpt['state_dict'], strict=False)
        logger.info("Pretrained weights loaded successfully")

    # Apply fine-tuning configurations (regardless of whether resuming or starting new)
    if train_mode == 'self-supervised':
        if hasattr(config, 'fine_tune') and hasattr(config.fine_tune, 'freeze_layers'):
            for name, param in model.named_parameters():
                if any(layer in name for layer in config.fine_tune.freeze_layers):
                    param.requires_grad = False
                    logger.info(f"Freezing layer: {name}")

    optimizer = build_optimizer(model, config.optimization)
    scheduler = build_scheduler(optimizer, config.optimization, 
                                config.optimization.num_epochs, len(train_loader))
    counter = Iter_counter()

    # Load optimizer and scheduler states if resuming
    if load_ckpt:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        best_val = checkpoint['best_performance']
        counter.set_start(checkpoint['iter'])
        logger.info(f"Resumed training from epoch {start_epoch}")
    else:
        if train_mode == 'supervised':
            logger.info('Training from scratch')
        else:
            logger.info('Starting new self-supervised fine-tuning')

    logger.info(f'Start {train_mode} model: {config.model_name}')
    epochs_num = config.optimization.num_epochs
    for epoch in range(start_epoch, epochs_num):
        logger.info(f'Epoch {epoch+1}/{epochs_num}')
        train_one_epoch(model, train_loader, optimizer, scheduler, 
                        device, epoch, writer, logger, counter)
        error = validate(model, val_loader, device, writer, logger, counter)
        
        is_best = error < best_val
        best_val = min(error, best_val)
        if epoch % 5 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'iter': counter.iter,
                'state_dict': model.state_dict(),
                'best_performance': best_val,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, filename=ckpt_path / f"checkpoint_epoch_{epoch+1}.pth.tar")
        
        if is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'iter': counter.iter,
                'state_dict': model.state_dict(),
                'best_performance': best_val,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, filename= ckpt_path / "best_model.pth.tar")

    writer.close()
    logger.info(f'{train_mode.capitalize()} finished for {config.model_name}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, default='cfg/pretrain.yaml')
    parser.add_argument('--pretrained_path', type=str, default=None)
    args = parser.parse_args()

    cfg_path = args.cfg_path
    pretrained_path = args.pretrained_path
    out_path = Path('output')
    main(cfg_path, out_path, pretrained_path=pretrained_path, load_ckpt=False)