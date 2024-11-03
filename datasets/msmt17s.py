# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import glob
import os.path as osp
import re
from collections import defaultdict

import pytorch_lightning as pl
from torch.utils.data import (DataLoader, Dataset, DistributedSampler,
                              SequentialSampler)

from .bases import (BaseDatasetLabelled, BaseDatasetLabelledPerPid,
                    ReidBaseDataModule, collate_fn_alternative, pil_loader)
from .samplers import get_sampler
from .transforms import ReidTransforms
import time

##### Log #####
# 22.01.2019
# - add v2
# - v1 and v2 differ in dir names
# - note that faces in v2 are blurred
TRAIN_DIR_KEY = 'train_dir'
TEST_DIR_KEY = 'test_dir'
VERSION_DICT = {
    'MSMT17_V1s': {
        TRAIN_DIR_KEY: 'train',
        TEST_DIR_KEY: 'test',
    },
    'MSMT17_V2s': {
        TRAIN_DIR_KEY: 'mask_train_v2',
        TEST_DIR_KEY: 'mask_test_v2',
    }
}


class MSMT17s(ReidBaseDataModule):
    """MSMT17.
    Reference:
        Wei et al. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. CVPR 2018.
    URL: `<http://www.pkuvmc.com/publications/msmt17.html>`_

    Dataset statistics:
        - identities: 4101.
        - images: 32621 (train) + 11659 (query) + 82161 (gallery).
        - cameras: 15.
    """
    # dataset_dir = 'MSMT17_V2'
    dataset_url = None
    # dataset_dir = 'msmt17'
    
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.dataset_dir = cfg.DATASETS.ROOT_DIR
        print("self.dataset_dir---: ", self.dataset_dir)
        has_main_dir = False
        for main_dir in VERSION_DICT:
            print("aaaaa: ", osp.join(self.dataset_dir, main_dir))
            if osp.exists(osp.join(self.dataset_dir, main_dir)):
                train_dir = VERSION_DICT[main_dir][TRAIN_DIR_KEY]
                test_dir = VERSION_DICT[main_dir][TEST_DIR_KEY]
                has_main_dir = True
                break
        assert has_main_dir, 'Dataset folder not found'
        self.train_dir = osp.join(self.dataset_dir, main_dir, train_dir)
        self.test_dir = osp.join(self.dataset_dir, main_dir, test_dir)
        self.list_train_path = osp.join(self.dataset_dir, main_dir, 'list_train.txt')
        self.list_val_path = osp.join(self.dataset_dir, main_dir, 'list_val.txt')
        self.list_query_path = osp.join(self.dataset_dir, main_dir, 'list_query.txt')
        self.list_gallery_path = osp.join(self.dataset_dir, main_dir, 'list_gallery.txt')
        
    def setup(self):
        # self._check_before_run()
        transforms_base = ReidTransforms(self.cfg)
        import time
        # time.sleep(5)
        train, train_dict = self._process_dir(self.train_dir, self.list_train_path)
        val, val_dict = self._process_dir(self.train_dir, self.list_val_path, True, len(train))
        
        self.train_list = train + val
        self.train_dict = self.concatenate_dicts(train_dict, val_dict)
        # self.train_dict = train_dict
        self.train = BaseDatasetLabelledPerPid(self.train_dict, transforms_base.build_transforms(is_train=True), self.num_instances, self.cfg.DATALOADER.USE_RESAMPLING)
        
        query, query_dict = self._process_dir(self.test_dir, self.list_query_path, is_train=False)
        gallery, gallery_dict  = self._process_dir(self.test_dir, self.list_gallery_path, is_train=False)
        
        # self.val = val
        
        self.query_list = query
        self.gallery_list = gallery
        self.val = BaseDatasetLabelled(query+gallery, transforms_base.build_transforms(is_train=False))
        
        
        self._print_dataset_statistics(self.train_list, query, gallery)
        # For reid_metic to evaluate properly
        num_query_pids, num_query_imgs, num_query_cams = self._get_imagedata_info(query)
        num_train_pids, num_train_imgs, num_train_cams = self._get_imagedata_info(train)
        self.num_query = len(query)
        self.num_classes = num_train_pids
        
    def concatenate_dicts(self, d1, d2):
        result = defaultdict(list, d1)  # Start with the contents of the first dict
        for key, value in d2.items():
            result[key].extend(value)  # Extend the list for each key
        return result

    def _process_dir(self, dir_path, list_path, is_train=True, index_start=0):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()
            
        dataset_dict = defaultdict(list)
        data = []
       
        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            pid = int(pid)  # no need to relabel
            camid = int(img_path.split('_')[2]) - 1  # index starts from 0
            img_path = osp.join(dir_path, img_path)
            # if is_train:
            #     pid = str(pid)
            #     camid = str(camid)
            data.append((img_path, pid, camid, img_idx + index_start))
            dataset_dict[pid].append((img_path, pid, camid, img_idx + index_start))

        return data, dataset_dict

    def process_merge(self, dir_path, list_path, is_train=True):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()
        data = []

        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            pid = int(pid)  # no need to relabel
            camid = int(img_path.split('_')[2]) - 1  # index starts from 0
            img_path = osp.join(dir_path, img_path)
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid))

        return data
        
    def parse_data(self, data):
        """Parses data list and returns the number of person IDs
        and the number of camera views.
        Args:
            data (list): contains tuples of (img_path(s), pid, camid)
        """
        pids = set()
        cams = set()
        
        for info in data:
            pids.add(info[1])
            cams.add(info[2])
        return len(pids), len(cams)