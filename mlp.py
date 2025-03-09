import torch
from engine import Engine


class Client(torch.nn.Module):
    def __init__(self, config):
        super(Client, self).__init__()
        self.config = config
        self.num_items_train = config['num_items_train']
        self.num_items_test = config['num_items_test']  # 添加测试集物品数量
        self.num_items_vali = config['num_items_vali']  # 添加验证集物品数量
        self.latent_dim = config['latent_dim']
        self.relu = torch.nn.ReLU()

        # 保留embedding_item，但改变其用途
        self.embedding_user = torch.nn.Embedding(num_embeddings=1, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Linear(in_features=768 * 2, out_features=self.latent_dim)

        # 创建多层网络结构
        self.fc_layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        # 解析层配置
        if isinstance(config['client_model_layers'], str):
            layers = [int(x) for x in config['client_model_layers'].split(',')]
        else:
            layers = config['client_model_layers']

        # 构建网络层（第一层处理latent_dim维度的输入）
        input_dim = self.latent_dim
        for output_dim in layers:
            self.fc_layers.append(torch.nn.Linear(input_dim, output_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(output_dim))
            input_dim = output_dim

        # 最终输出层
        self.affine_output = torch.nn.Linear(in_features=layers[-1], out_features=1)
        self.dropout = torch.nn.Dropout(0.3)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, item_cv, item_txt):
        # 先通过embedding_item处理特征
        item_fet = self.relu(torch.cat([item_cv, item_txt], dim=-1)).cuda()
        item_embedding = self.embedding_item(item_fet)

        # 通过多层网络
        vector = item_embedding
        for idx in range(len(self.fc_layers)):
            vector = self.fc_layers[idx](vector)
            vector = self.batch_norms[idx](vector)
            vector = self.relu(vector)
            vector = self.dropout(vector)

        # 最终输出
        logits = self.affine_output(vector)
        rating = self.logistic(logits)

        if self.training:
            max_idx = self.num_items_train
        else:
            # 根据评估阶段使用不同的最大索引
            max_idx = self.num_items_test  # 或 self.num_items_vali

        rating = torch.clamp(rating, 0, max_idx - 1)
        return rating


    def init_weight(self):
        pass

    def load_pretrain_weights(self):
        pass




class MLPEngine(Engine):
    """Engine for training & evaluating GMF model"""
    def __init__(self, config):
        self.client_model = Client(config)
        # self.server_model = Server(config)
        if config['use_cuda'] is True:
            # use_cuda(True, config['device_id'])
            self.client_model.cuda()
            # self.server_model.cuda()
        super(MLPEngine, self).__init__(config)
