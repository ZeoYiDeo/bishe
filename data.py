import torch
import random
import pandas as pd
from copy import deepcopy
from torch.utils.data import Dataset
import os

random.seed(0)


class UserItemRatingDataset(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""

    def __init__(self, user_tensor, item_tensor, target_tensor, features_dir='../data/Bili_Food/'):
        """
        args:
            user_tensor: torch.Tensor, user IDs
            item_tensor: torch.Tensor, item IDs
            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
            features_dir: str, directory containing feature files
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor
        self.features_dir = features_dir

        # 使用内存映射加载特征文件
        self.cv = torch.load(os.path.join(features_dir, 'Bili_Food_vit.pt'), map_location='cpu')
        self.ct = torch.load(os.path.join(features_dir, 'Bili_Food_bert.pt'), map_location='cpu')

    def __getitem__(self, index):
        item_id = int(self.item_tensor[index])

        # 获取特征时才将数据移动到GPU
        cv_feature = self.cv[item_id]
        ct_feature = self.ct[item_id]

        return (self.user_tensor[index],
                self.item_tensor[index],
                self.target_tensor[index],
                cv_feature,
                ct_feature)

    def __len__(self):
        return self.user_tensor.size(0)