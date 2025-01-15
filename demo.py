from easydict import EasyDict
from pathlib import Path
import torch
import argparse
from core.model import PoseEstimator
from core.dataset import DemoDataset
from utils.train_utils import load_checkpoint, load_dict_to_cuda
from utils.utils import setup_logger, format_result_dict, load_config
from utils.vis_utils import render_pose_with_points


def main(cfg_path, ckpt_path, data_path):

    yaml_config = load_config(cfg_path)
    config = EasyDict(yaml_config)
    logger = setup_logger()

    demo_dataset = DemoDataset(Path(data_path))
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PoseEstimator(in_channels=3, cfg=config.model, device=device).to(device)

    checkpoint = load_checkpoint(ckpt_path)
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    with torch.no_grad():
        while True:
            ind = input('please input sample index in 0-%d: ' % len(demo_dataset))
            if ind == '':
                break
            elif not ind.isdigit():
                print('Try to load a sample by token...')
                found = False
                for i, data_anno in enumerate(demo_dataset):
                    if data_anno[1]['token'] == ind:
                        found = True
                        break
                if not found:
                    print('Token not found.')
                    continue
                else:
                    input_dict, anno = demo_dataset.collate_batch([data_anno])
                    logger.info(f'Load {i}th sample with token {anno["token"]}')
                    input_dict = load_dict_to_cuda(input_dict, device)
            elif int(ind) >= len(demo_dataset):
                print('Index out of range.')
                continue
            else:
                data_anno = demo_dataset[int(ind)]
                input_dict, anno = demo_dataset.collate_batch([data_anno])
                logger.info(f'Load sample with token {anno["token"]}')
                input_dict = load_dict_to_cuda(input_dict, device)

            result_dict = model(input_dict)

            batch_offset = input_dict['offset'].cpu().numpy().tolist()

            result_list_batch = format_result_dict(result_dict, batch_offset)

            points = input_dict['coord'].cpu().numpy()
            offset_ = [0] + batch_offset
            points_list = []
            for i in range(len(offset_) - 1):
                points_list.append(points[offset_[i]:offset_[i+1]])

            render_pose_with_points(result_list_batch[0], points_list[0])

    logger.info('Demo done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='None', required=True)
    parser.add_argument('--cfg_path', type=str, default='cfg/model.yaml')
    parser.add_argument('--data_path', type=str, default='data/demo_data')

    args = parser.parse_args()
    cfg_path = args.cfg_path
    ckpt = args.ckpt
    data_path = args.data_path
    main(cfg_path, ckpt, data_path)