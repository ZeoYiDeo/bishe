import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import data_processor_v2
import copy
import numpy as np
import pandas as pd
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class UserItemRatingDataset_V2(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""

    def __init__(self, user_tensor, item_tensor, target_tensor):
        """
        args:

            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):

        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)


def negative_sampling_v2(train_data, num_negatives):
    """sample negative instances for training, refer to Heater."""
    # warm items in training set.
    item_warm = np.unique(train_data['iid'].values)
    # arrange the training data with form {user_1: [[user_1], [user_1_item], [user_1_rating]],...}.
    train_dict = {}
    single_user, user_item, user_rating = [], [], []
    grouped_train_data = train_data.groupby('uid')
    for userId, user_train_data in grouped_train_data:
        temp = copy.deepcopy(item_warm)
        for row in user_train_data.itertuples():
            single_user.append(int(row.uid))
            user_item.append(int(row.iid))
            user_rating.append(float(1))
            temp = np.delete(temp, np.where(temp == row.iid))
            for i in range(num_negatives):
                single_user.append(int(row.uid))
                negative_item = np.random.choice(temp)
                user_item.append(int(negative_item))
                user_rating.append(float(0))
                temp = np.delete(temp, np.where(temp == negative_item))
        train_dict[userId] = [single_user, user_item, user_rating]
        single_user = []
        user_item = []
        user_rating = []
    return train_dict


def instance_user_train_loader_v2(user_train_data):
        """instance a user's train loader."""
        dataset = UserItemRatingDataset_V2(user_tensor=torch.LongTensor(user_train_data[0]),
                                        item_tensor=torch.LongTensor(user_train_data[1]),
                                        target_tensor=torch.FloatTensor(user_train_data[2]))
        return DataLoader(dataset, batch_size=64, num_workers=2, shuffle=True)

# if __name__ == '__main__':
#     file = "../data/Bili_Food/Bili_Food_pair.csv"
#     data_dict = data_processor_v2.load_data(file)
#     all_data = negative_sampling_v2(data_dict['id_data'], 5)

    # for user in data_dict['user_ids']:
    #     user_train_data = all_data[user]
    #     train_loader = instance_user_train_loader_v2(user_train_data)
    #     for batch in train_loader:
    #         user_ids,item_ids,target = batch
    #         print(user_ids.shape, item_ids.shape,target.shape)
    # print(len(data_dict['item_ids']))
    # dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
    # for batch in dataloader:
    #
    #     img_feat = batch['item_img_features']
    #     text_feat = batch['item_text_features']
    #     user_ids = batch['user_ids']
    #     item_ids = batch['item_ids']
    #     id_data = batch['id_data']
    #     print(img_feat.shape, text_feat.shape, id.shape)
    #
    #     # for i in range(len(batch['id'])):
    #     #     print(f"id: {id[i]}")
    #     #     print(f"图像特征: {img_feat[i]}")
    #     #     print(f"文本特征: {text_feat[i]}")
    #     # break
    # print(len(dataset))
    # # print(f"图像特征数量: {len(dataset.img_feat)}")
    # # print(f"文本特征数量: {len(dataset.text_feat)}")
    # # print(f"共同特征数量: {len(dataset.ids)}")
