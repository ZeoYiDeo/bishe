import torch
from engine import Engine


class Client(torch.nn.Module):
    def __init__(self, config):
        super(Client, self).__init__()
        self.config = config
        self.num_items_train = config['num_items_train']
        self.num_items_test = config['num_items_test']
        self.num_items_vali = config['num_items_vali']
        self.latent_dim = config['latent_dim']

        # 减小 embedding 维度
        self.embedding_user = torch.nn.Embedding(num_embeddings=1, embedding_dim=self.latent_dim)
        # 使用更小的中间维度
        self.embedding_item = torch.nn.Sequential(
            torch.nn.Linear(768 * 2, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, self.latent_dim)
        )

        # 优化网络结构
        layers = [int(x) for x in config['client_model_layers'].split(',')] if isinstance(config['client_model_layers'],
                                                                                          str) else config[
            'client_model_layers']

        self.network = torch.nn.ModuleList()
        input_dim = self.latent_dim

        for output_dim in layers:
            layer = torch.nn.Sequential(
                torch.nn.Linear(input_dim, output_dim),
                torch.nn.BatchNorm1d(output_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2)
            )
            self.network.append(layer)
            input_dim = output_dim

        self.output = torch.nn.Sequential(
            torch.nn.Linear(layers[-1], 1),
            torch.nn.Sigmoid()
        )

    @torch.cuda.amp.autocast()  # 使用混合精度训练
    def forward(self, item_cv, item_txt):
        # 合并特征并移动到 GPU
        item_fet = torch.cat([item_cv, item_txt], dim=-1)
        if self.training and item_fet.device.type != 'cuda':
            item_fet = item_fet.cuda()

        # 特征提取
        item_embedding = self.embedding_item(item_fet)

        # 通过网络层
        vector = item_embedding
        for layer in self.network:
            vector = layer(vector)

        # 输出预测
        rating = self.output(vector)

        if self.training:
            max_idx = self.num_items_train
        else:
            max_idx = self.num_items_test if self.config.get('is_test', True) else self.num_items_vali

        return torch.clamp(rating, 0, max_idx - 1)

    def init_weight(self):
        pass

    def load_pretrain_weights(self):
        pass


class Server(torch.nn.Module):
    def __init__(self, config):
        super(Server, self).__init__()
        self.config = config
        self.content_dim = config['content_dim']
        self.latent_dim = config['latent_dim']

        # 创建多层网络结构
        self.fc_layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        # 解析层配置
        if isinstance(config['server_model_layers'], str):
            layers = [int(x) for x in config['server_model_layers'].split(',')]
        else:
            layers = config['server_model_layers']

        # 第一层处理拼接后的特征 (768*2 = 1536)
        input_dim = self.content_dim * 2

        # 构建网络层
        for output_dim in layers:
            self.fc_layers.append(torch.nn.Linear(input_dim, output_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(output_dim))
            input_dim = output_dim

        # 最终输出层
        self.affine_output = torch.nn.Linear(in_features=layers[-1], out_features=self.latent_dim)
        self.dropout = torch.nn.Dropout(0.2)
        self.logistic = torch.nn.Tanh()

    def forward(self, item_cv_feat, item_txt_feat):
        vector = torch.cat([item_cv_feat, item_txt_feat], dim=-1).cuda()

        for idx in range(len(self.fc_layers)):
            vector = self.fc_layers[idx](vector)
            vector = self.batch_norms[idx](vector)
            vector = torch.nn.ReLU()(vector)
            vector = self.dropout(vector)

        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass

    def load_pretrain_weights(self):
        pass


class MLPEngine(Engine):
    """Engine for training & evaluating GMF model"""
    def __init__(self, config):
        self.client_model = Client(config)
        self.server_model = Server(config)
        if config['use_cuda'] is True:
            # use_cuda(True, config['device_id'])
            self.client_model.cuda()
            self.server_model.cuda()
        super(MLPEngine, self).__init__(config)
