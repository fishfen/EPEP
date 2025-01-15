from pathlib import Path
import numpy as np
import json
import torch
import torch.utils.data as torch_data
from torch.utils.data import DataLoader

class DatasetBase(torch_data.Dataset):
    def __init__(self, mode, cfg, logger, root_path=None) -> None:
        super().__init__()

        assert mode in ['train', 'val', 'test']
        self.split = mode
        self.cfg = cfg
        root_path = Path('data/syn_data') if root_path is None else root_path
        file_path = root_path / f'{self.split}.txt'
        assert file_path.exists()
        self.seq_list = [x.strip() for x in open(file_path).readlines()]

        self.sample_path = root_path / 'pointclouds'
        self.label_path = root_path / 'labels'

        self.logger = logger

    def __len__(self):
        return NotImplementedError
    
    def load_label(self, token):
        label_path = self.label_path / f'{token}.json'
        assert label_path.exists()
        with open(label_path, 'r') as f:
            anno = json.load(f)

        anno_used = {}
        anno_used['token'] = token
        anno_used.update(anno['size'])
        anno_used.update(anno['pose'])
        anno_used.update(anno['point_info'])
        anno_used['other_offset'] = [anno_used.pop(k) for k in ['root_z', 'root_y', 'cabin_x']]  # arrange them in one key for regression
        anno_used['rotation'] = anno_used.pop('rotation_mat')  # change keyname in line with prediction
        length = anno['meta']['point_number']

        return anno_used, length
    
    def __getitem__(self, index):
        return NotImplementedError
    

class SeqDataset(DatasetBase):
    '''Load one sequence as a batch, so the batchsize is fixed'''
    def __init__(self, mode, cfg, logger, root_path=None) -> None:
        super().__init__(mode, cfg, logger, root_path)
        self.logger.info(f'Loading {self.split} dataset with {self.__len__()} sequence, {self.cfg.num_per_seq} samples in each seq')

    def __len__(self):
        return len(self.seq_list)
    
    def __getitem__(self, index):

        # load a number of samples in a sequence, which can be regarded as a mini batch
        # the point number of each sample may vary
        seq = [f'{self.seq_list[index]}_{i:02d}' for i in range(self.cfg.num_per_seq)]
        points_batch = []
        anno_batch = []
        batch_offset = [0]
        for s in seq:
            lidar_path = self.sample_path / f'{s}.bin'
            assert lidar_path.exists()
            points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 3])
            points_batch.append(points)

            anno, length = self.load_label(s)
            assert len(points) == length
            anno_batch.append(anno)
            batch_offset.append(length + batch_offset[-1])
        
        batch_offset = np.array(batch_offset[1:])
        points_batch = np.concatenate(points_batch, axis=0)
        input_dict = {'coord':points_batch,
                      'feat':points_batch,
                      'offset':batch_offset}
        
        return input_dict, anno_batch
    
    @staticmethod
    def collate_batch(batch_list):
        # used for collate_fn of DataLoader, one item from Dataset is already a batch
        assert len(batch_list) == 1, 'the batch size for dataloader should be 1'

        # note that, when used by torch Dataloader, the batch_list is a list. In our case, it is a list of length 1.
        # the following items are the same as the return of __getitem__()
        input_dict, anno_dict_list = batch_list[0][0], batch_list[0][1]

        # for points data, just convert to tensor
        for key, value in input_dict.items():
            if key in ['offset']:
                input_dict[key] = torch.from_numpy(value).int()
            else:
                input_dict[key] = torch.from_numpy(value).float()

        # for target variables, stack them in a batch
        anno_dict = {}
        for key in anno_dict_list[0].keys():
            values = [d[key] for d in anno_dict_list]
            if key in ['keypoints','rotation']:
                values = np.array(values)
                values = torch.from_numpy(values).float()
            # for point-wise groundtruth, concatenate all of them as a longer 1d array.
            elif key in ['segment']:
                values = np.concatenate(values)
                values = torch.from_numpy(values).long()
            else:
                values = np.array(values)
                values = torch.from_numpy(values).float()
            anno_dict[key] = values

        return input_dict, anno_dict


class FrameDataset(DatasetBase):
    '''Load one frame with customed batchsize'''
    def __init__(self, mode, cfg, logger, root_path=None) -> None:
        super().__init__(mode, cfg, logger, root_path)
        self.token_list = [f'{index}_{i:02d}' for i in range(self.cfg.num_per_seq) for index in self.seq_list]
        self.logger.info(f'Loading {self.split} dataset with {self.__len__()} samples')

    def __len__(self):
        return len(self.token_list)
    
    def __getitem__(self, index):

        # the point number of each sample may vary
        token = self.token_list[index]

        lidar_path = self.sample_path / f'{token}.bin'
        assert lidar_path.exists()
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 3])
        anno, length = self.load_label(token)
        assert len(points) == length
        
        input_dict = {'coord':points,
                      'feat':points,
                      'offset':length}
        
        return input_dict, anno
    
    @staticmethod
    def collate_batch(batch_list):
        # note that, when used by torch Dataloader, the batch_list is a list.
        # the following items are the same as the return of __getitem__()
        input_dict_list = [tup[0] for tup in batch_list]
        anno_dict_list = [tup[1] for tup in batch_list]

        # for points data, concatenate them in 1D
        input_dict = {}
        for key in input_dict_list[0].keys():
            values = [d[key] for d in input_dict_list]
            if key in ['offset']:
                value = np.cumsum(values)
                input_dict[key] = torch.from_numpy(value).int()
            elif key in ['coord', 'feat']:
                value = np.concatenate(values)
                input_dict[key] = torch.from_numpy(value).float()
            else:
                raise NotImplementedError
            
        # for target variables, stack them in a batch
        anno_dict = {}
        for key in anno_dict_list[0].keys():
            values = [d[key] for d in anno_dict_list]
            if key in ['keypoints','rotation']:
                values = np.array(values)
                values = torch.from_numpy(values).float()
            # for point-wise groundtruth, concatenate all of them as a longer 1d array.
            elif key in ['segment']:
                values = np.concatenate(values)
                values = torch.from_numpy(values).long()
            elif key not in ['token']:
                values = np.array(values)
                values = torch.from_numpy(values).float()
            anno_dict[key] = values

        return input_dict, anno_dict


class RealDataset(torch_data.Dataset):
    def __init__(self, mode, root_path, logger, need_label=True, sort=True, filter=None) -> None:
        super().__init__()

        assert mode in ['val', 'test', 'train']
        mode = 'test' if mode == 'val' else mode
        self.split = mode
        self.logger = logger
        self.root_path = root_path

        with open(root_path / 'split.json', 'r') as f:
            scene_names = json.load(f)[self.split]
        token_list = list((root_path / 'pointclouds').glob('*.bin'))
        self.token_list = [t for t in token_list if t.stem[:16] in scene_names]
        if sort:
            self.token_list.sort()
        self.logger.info(f'Loading REAL {self.split} dataset with {self.__len__()} samples')

        if mode == 'test' and filter is not None:
            self.filter_token(filter)  # filter the token list based on the given filter pattern

        self.need_label = need_label
        if need_label:
            with open(self.root_path / 'estimated_size_train.json', 'r') as f:
                self.size_info = json.load(f)
    
    def filter_token(self, filter):
        with open(self.root_path / 'dataset_info.json', 'r') as f:
            data_info = json.load(f)
        filter_key = filter['name']
        filter_range = filter['range']

        if filter_key == 'token':
            self.token_list = [t for t in self.token_list if any(t.stem.startswith(f) for f in filter_range)]
        else:
            token_filter = [d['token'] for d in data_info if d[filter_key] > filter_range[0] and d[filter_key] <= filter_range[1]]
            self.token_list = [t for t in self.token_list if t.stem in token_filter]
        print(f'Filter {filter_key} in range {filter_range}, {len(self.token_list)} samples left')
    
    def __len__(self):
        return len(self.token_list)
    
    def __getitem__(self, index):
        lidar_path = self.token_list[index]
        token = lidar_path.stem
        assert lidar_path.exists()
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 3])

        input_dict = {'coord':points,
                      'feat':points,
                      'offset':len(points)}
        anno = {'token': token}
        if self.need_label:
            anno_ = self.__getattribute__(f'load_{self.split}_label')(token)
            anno.update(anno_)
        return input_dict, anno
    
    def load_train_label(self, token):
        size_dict = self.size_info[token[:22]] # (14,)

        label_path = self.root_path / 'train_labels' / f'{token}.json'
        assert label_path.exists()
        with open(label_path, 'r') as f:
            anno = json.load(f)
        
        anno.update(size_dict)
        return anno 
    
    def load_test_label(self, token):
        label_path = self.root_path / 'test_labels' / f'{token}.json'
        assert label_path.exists()
        with open(label_path, 'r') as f:
            anno = json.load(f)

        anno_used = {}
        anno_used['token'] = token
        anno_used.update(anno['size'])
        anno_used.update(anno['pose'])
        anno_used['other_offset'] = [anno_used.pop(k) for k in ['root_z', 'root_y', 'cabin_x']]  # arrange them in one key for regression
        anno_used['rotation'] = anno_used.pop('rotation_mat')  # change keyname in line with prediction

        return anno_used

    @staticmethod
    def collate_batch(batch_list):
        input_dict_list = [tup[0] for tup in batch_list]
        anno_dict_list = [tup[1] for tup in batch_list]

        # for points data, concatenate them in 1D
        input_dict = {}
        for key in input_dict_list[0].keys():
            values = [d[key] for d in input_dict_list]
            if key in ['offset']:
                value = np.cumsum(values)
                input_dict[key] = torch.from_numpy(value).int()
            elif key in ['coord', 'feat']:
                value = np.concatenate(values)
                input_dict[key] = torch.from_numpy(value).float()
            else:
                raise NotImplementedError
            
        # for target variables, stack them in a batch
        anno_dict = {}
        for key in anno_dict_list[0].keys():
            values = [d[key] for d in anno_dict_list]
            if key in ['segment']:
                values = np.concatenate(values)
                values = torch.from_numpy(values).long()
            elif key not in ['token']:
                values = np.array(values)
                values = torch.from_numpy(values).float()
            
            anno_dict[key] = values

        return input_dict, anno_dict


class DemoDataset(torch_data.Dataset):
    def __init__(self, data_path=None, sort=True) -> None:
        super().__init__()
        self.token_list = list(data_path.glob('*.bin'))
        if sort:
            self.token_list.sort()
    
    def __len__(self):
        return len(self.token_list)
    
    def __getitem__(self, index):
        lidar_path = self.token_list[index]
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 3])

        points = torch.from_numpy(points).float()
        input_dict = {'coord':points,
                      'feat':points,
                      'offset':torch.tensor([len(points)], dtype=int)}
        anno = {'token': lidar_path.stem}
        return input_dict, anno
    
    @staticmethod
    def collate_batch(data):
        return data[0]


def build_dataloader(config, mode, data_path, logger, batchsize=1):
    
    dataset_type = config.dataset.type
    shuffle = True if mode == 'train' else False
    if dataset_type == 'seq':
        dataset = SeqDataset(mode=mode, cfg=config.dataset, logger=logger, root_path=data_path)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=2, collate_fn=dataset.collate_batch)
    elif dataset_type == 'frame':
        dataset = FrameDataset(mode=mode, cfg=config.dataset, logger=logger, root_path=data_path)
        dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=shuffle, num_workers=2, collate_fn=dataset.collate_batch)
    elif dataset_type == 'real':
        filter = config.dataset.get('filter', None)
        dataset = RealDataset(mode=mode, root_path=data_path, logger=logger, filter=filter)
        dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=shuffle, num_workers=2, collate_fn=dataset.collate_batch)
    else:
        raise NotImplementedError
    
    return dataloader


