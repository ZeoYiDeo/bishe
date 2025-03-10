import torch
from utils import *
import numpy as np
import copy
from data import UserItemRatingDataset
from torch.utils.data import DataLoader


class Engine(object):
    """Meta Engine for training & evaluating NCF model

    Note: Subclass should implement self.client_model and self.server_model!
    """

    def __init__(self, config):
        self.config = config  # model configuration
        # self.server_opt = torch.optim.Adam(self.server_model.parameters(), lr=config['lr_server'],
        #                                    weight_decay=config['l2_regularization'])
        # self.server_model_param = {}
        self.client_model_params = {}
        self.client_crit = torch.nn.BCELoss()
        # self.server_crit = torch.nn.MSELoss()

    def instance_user_train_loader(self, user_train_data):
        """instance a user's train loader."""
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(user_train_data[0]),
                                        item_tensor=torch.LongTensor(user_train_data[1]),
                                        target_tensor=torch.FloatTensor(user_train_data[2]))
        return DataLoader(dataset, batch_size=64, shuffle=True)

    def fed_train_single_batch(self, model_client, batch_data, optimizers):
        """train a batch and return an updated model."""
        # load batch data.
        _, items, ratings, cv, ct = batch_data[0], batch_data[1], batch_data[2], batch_data[3], batch_data[4]
        ratings = ratings.float()

        if self.config['use_cuda'] is True:
            items, ratings = items.cuda(), ratings.cuda()

        optimizer, optimizer_u, optimizer_i = optimizers
        # update score function.
        optimizer.zero_grad()
        optimizer_u.zero_grad()
        optimizer_i.zero_grad()
        ratings_pred = model_client(cv, ct)
        loss = self.client_crit(ratings_pred.view(-1), ratings)
        print("\r loss is {:.4f}".format(loss.item()), end='')
        loss.backward()
        optimizer.step()
        optimizer_u.step()
        optimizer_i.step()
        # 手动删除不再使用的Tensor
        del ratings_pred
        del loss
        return model_client

    def aggregate_clients_params(self, round_user_params):  #在一轮训练中接收客户端模型的参数，对这些参数进行聚合操作，然后将聚合后的结果存储在服务器端。
        """receive client models' parameters in a round, aggregate them and store the aggregated result for server."""
        # aggregate item embedding and score function via averaged aggregation.
        t = 0
        for user in round_user_params.keys():   #遍历 round_user_params 字典的所有键，这些键代表不同的用户。
            # load a user's parameters.
            user_params = round_user_params[user]
            # print(user_params)
            if t == 0:
                self.server_model_param = copy.deepcopy(user_params)
            else:
                for key in user_params.keys():
                    self.server_model_param[key].data += user_params[key].data
            t += 1
        for key in self.server_model_param.keys():
            self.server_model_param[key].data = self.server_model_param[key].data / len(round_user_params)



    def fed_train_a_round(self, user_ids, all_train_data, round_id):
        """train a round."""
        # sample users participating in single round.
        if self.config['clients_sample_ratio'] <= 1:
            num_participants = int(self.config['num_users'] * self.config['clients_sample_ratio'])
            participants = np.random.choice(user_ids, num_participants, replace=False)
        else:
            participants = np.random.choice(user_ids, self.config['clients_sample_num'], replace=False)

        # store users' model parameters of current round.
        round_participant_params = {}
        # perform model update for each participated user.
        for user in participants:
            # copy the client model architecture from self.client_model
            model_client = copy.deepcopy(self.client_model)
            # for the first round, client models copy initialized parameters directly.
            # for other rounds, client models receive updated item embedding from server.
            if round_id != 0:
                user_param_dict = copy.deepcopy(self.client_model.state_dict())
                if user in self.client_model_params.keys():
                    for key in self.client_model_params[user].keys():
                        user_param_dict[key] = copy.deepcopy(self.client_model_params[user][key].data).cuda()
                user_param_dict['embedding_item.weight'] = copy.deepcopy(self.server_model_param['embedding_item.weight'].data).cuda()
                model_client.load_state_dict(user_param_dict)
            # Defining optimizers
            # optimizer is responsible for updating score function.
            optimizer = torch.optim.SGD(
                [{"params": model_client.fc_layers.parameters()},
                 {"params": model_client.affine_output.parameters()}],
                lr=self.config['lr_client'])  # MLP optimizer
            # optimizer_u is responsible for updating user embedding.
            optimizer_u = torch.optim.SGD(model_client.embedding_user.parameters(),
                                          lr=self.config['lr_client'] / self.config['clients_sample_ratio'] * self.config['lr_eta'] - self.config['lr_client'])  # User optimizer
            # optimizer_i is responsible for updating item embedding.
            optimizer_i = torch.optim.SGD(model_client.embedding_item.parameters(),
                                          lr=self.config['lr_client'] * self.config['num_items_train'] * self.config['lr_eta'] -self.config['lr_client'])  # Item optimizer
            optimizers = [optimizer, optimizer_u, optimizer_i]

            # load current user's training data and instance a train loader.
            user_train_data = all_train_data[user]
            user_dataloader = self.instance_user_train_loader(user_train_data)
            model_client.train()
            # update client model.
            for epoch in range(self.config['local_epoch']):
                for batch_id, batch in enumerate(user_dataloader):
                    assert isinstance(batch[0], torch.LongTensor)  #检查批次数据的类型是否为 torch.LongTensor，如果不是则会抛出 AssertionError。
                    model_client = self.fed_train_single_batch(model_client, batch, optimizers)
            # obtain client model parameters.
            client_param = model_client.state_dict()
            # store client models' local parameters for personalization.
            self.client_model_params[user] = {}
            for key in client_param.keys():
                if key != 'embedding_item.weight':
                    self.client_model_params[user][key] = copy.deepcopy(client_param[key].data).cpu()
            # store client models' local parameters for global update.
            round_participant_params[user] = {}
            round_participant_params[user]['embedding_item.weight'] = copy.deepcopy(
                client_param['embedding_item.weight']).data.cpu()
            del user_dataloader
            del model_client
            del client_param
        # aggregate client models in server side.
        self.aggregate_clients_params(round_participant_params)


    def fed_evaluate(self, evaluate_data, item_cv_features, item_text_features, item_ids_map,is_test=True):
        """evaluate all client models' performance using testing data."""
        """input: 
        evaluate_data: (uid, iid) dataframe.
        item_content: evaluated item raw feature.
        item_ids_map: {ori_id: reindex_id} dict.
           output:
        recall, precision, ndcg
        """

        print(f"\n开始{'测试集' if is_test else '验证集'}评估")
        print(f"评估数据大小: {len(evaluate_data)}")
        print(f"物品特征数量: {len(item_cv_features)}")
        print(f"物品映射大小: {len(item_ids_map)}")

        item_cv_content = torch.tensor(item_cv_features)
        item_text_content = torch.tensor(item_text_features)


        # obtain cola-start items' prediction for each user.
        user_ids = evaluate_data['uid'].unique()
        user_item_preds = {}
        for user in user_ids:
            user_model = copy.deepcopy(self.client_model)
            user_param_dict = copy.deepcopy(self.client_model.state_dict())
            if user in self.client_model_params.keys():
                for key in self.client_model_params[user].keys():
                    user_param_dict[key] = copy.deepcopy(self.client_model_params[user][key].data).cuda()
            user_model.load_state_dict(user_param_dict)
            user_model.eval()
            with torch.no_grad():
                cold_pred = user_model(item_cv_content, item_text_content)
                user_item_preds[user] = cold_pred.view(-1)

        # compute the evaluation metrics.
        recall, precision, ndcg =compute_metrics(
            evaluate_data,
            user_item_preds,
            item_ids_map,
            self.config['recall_k'],
            is_test = is_test
        )
        return recall, precision, ndcg
