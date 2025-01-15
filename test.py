import os
from easydict import EasyDict
from pathlib import Path
import json
import argparse
import torch
from core.model import PoseEstimator
from core.dataset import build_dataloader
from utils.utils import setup_logger, load_config
from utils.train_utils import load_checkpoint
from utils.eval_utils import evaluate


# evaluate model
def main(cfg_path, out_path, data_path=None, ckpt=None):

    yaml_config = load_config(cfg_path)
    config = EasyDict(yaml_config)

    result_path = out_path / config.experiment_name
    if not os.path.exists(result_path):
        os.makedirs(result_path, exist_ok=True)

    logger = setup_logger()
    data_path = Path(config.dataset.data_path) if data_path is None else Path(data_path)
    test_loader = build_dataloader(config=config, mode='test', data_path=data_path, 
                                  logger=logger, batchsize=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PoseEstimator(in_channels=3, cfg=config.model, device=device).to(device)
    # print('Model architecture:')
    # print(model)

    ckpt = ckpt if ckpt is not None else config.ckpt
    checkpoint = load_checkpoint(ckpt)
    model.load_state_dict(checkpoint['state_dict'])

    logger.info(f'Start evaluate model: {config.model_name}')
    mt_dict, _ = evaluate(model, test_loader, device, config, logger)
    print(mt_dict)
    if config.dataset.get('filter', None) is not None:
        filter = config.dataset.filter
        eval_name = f'{filter["name"]}-{filter["range"][0]}'
    else:
        eval_name = ''
    with open(result_path / f'evaluation-{config.dataset.type}-{eval_name}.json', 'w') as f:
        json.dump(mt_dict, f)
        
    logger.info(f'Evaluation finished {config.model_name}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, default='cfg/eval.yaml')
    parser.add_argument('--out_path', type=str, default='output')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--ckpt', type=str, default=None)
    args = parser.parse_args()

    out_path = Path(args.out_path)
    data_path = args.data_path
    ckpt = args.ckpt
    cfg_path = args.cfg_path

    main(cfg_path=cfg_path, out_path=out_path, data_path=data_path, ckpt=ckpt)