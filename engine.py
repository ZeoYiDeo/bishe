import torch
import torch.nn as nn
import numpy as np
import logging
from torch.utils.data import DataLoader, TensorDataset
import copy
from typing import Dict, List, Tuple
import gc
from tqdm import tqdm


class Engine:
    """Base Engine class for training and evaluating recommendation models"""

    def __init__(self, config):
        self.config = config
        self.client_model_params = {}
        self.server_model_param = None
        # 使用混合精度训练
        self.scaler = torch.cuda.amp.GradScaler()
        # 设置内存管理
        torch.cuda.empty_cache()
        if config['use_cuda']:
            torch.backends.cudnn.benchmark = True

    def instance_user_train_loader(self, user_train_data: np.ndarray) -> DataLoader:
        """创建用户训练数据加载器，优化内存使用"""
        user_ids = torch.LongTensor(user_train_data[:, 1])
        item_ids = torch.LongTensor(user_train_data[:, 0])

        # 使用 TensorDataset 提高效率
        dataset = TensorDataset(user_ids, item_ids)
        return DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            pin_memory=True,
            num_workers=4
        )

    def _move_to_device(self, data: torch.Tensor) -> torch.Tensor:
        """将数据移动到适当的设备（GPU/CPU）"""
        if self.config['use_cuda'] and not data.is_cuda:
            return data.cuda(non_blocking=True)
        return data

    def _release_memory(self):
        """释放不需要的内存"""
        gc.collect()
        torch.cuda.empty_cache()

    @torch.cuda.amp.autocast()
    def fed_train_single_batch(self, model: nn.Module, batch: Tuple[torch.Tensor], optimizers: List) -> nn.Module:
        """单批次联邦学习训练"""
        optimizer, optimizer_u, optimizer_i = optimizers

        # 清除梯度
        optimizer.zero_grad()
        optimizer_u.zero_grad()
        optimizer_i.zero_grad()

        # 将数据移到GPU
        batch = [self._move_to_device(b) for b in batch]

        # 前向传播
        loss = model(*batch)

        # 使用混合精度训练
        self.scaler.scale(loss).backward()

        # 更新参数
        self.scaler.step(optimizer)
        self.scaler.step(optimizer_u)
        self.scaler.step(optimizer_i)

        self.scaler.update()

        return model

    def aggregate_clients_params(self, round_user_params: Dict, item_cv_features: np.ndarray,
                                 item_text_features: np.ndarray):
        """聚合客户端参数"""
        if not round_user_params:
            return

        # 使用高效的参数聚合方法
        param_keys = next(iter(round_user_params.values())).keys()
        num_clients = len(round_user_params)

        # 初始化聚合参数
        self.server_model_param = {}
        for key in param_keys:
            # 使用第一个客户端的参数作为初始值
            first_client = next(iter(round_user_params.values()))
            aggregated_param = first_client[key].clone()

            # 累加其他客户端的参数
            for client_params in list(round_user_params.values())[1:]:
                aggregated_param.add_(client_params[key])

            # 计算平均值
            aggregated_param.div_(num_clients)
            self.server_model_param[key] = aggregated_param

    def fed_train_a_round(self, user_ids: List[int], all_train_data: Dict, round_id: int,
                          item_cv_features: np.ndarray, item_text_features: np.ndarray):
        """训练一轮联邦学习"""
        # 采样参与训练的用户
        num_participants = int(self.config['num_users'] * self.config['clients_sample_ratio']) \
            if self.config['clients_sample_ratio'] <= 1 else self.config['clients_sample_num']
        participants = np.random.choice(user_ids, num_participants, replace=False)

        # 将特征数据转换为张量并移到GPU
        item_cv_features = torch.tensor(item_cv_features, device='cuda' if self.config['use_cuda'] else 'cpu')
        item_text_features = torch.tensor(item_text_features, device='cuda' if self.config['use_cuda'] else 'cpu')

        # 存储参与者的模型参数
        round_participant_params = {}

        # 训练每个参与者的模型
        for user in tqdm(participants, desc=f"Training Round {round_id}"):
            # 避免完整的模型复制
            model_client = self.client_model

            if round_id > 0:
                # 只更新必要的参数
                if user in self.client_model_params:
                    for key, param in self.client_model_params[user].items():
                        getattr(model_client, key).data.copy_(param.data.cuda())

                # 更新服务器端参数
                if self.server_model_param is not None:
                    model_client.embedding_item.weight.data.copy_(
                        self.server_model_param['embedding_item.weight'].data.cuda()
                    )

            # 设置优化器
            optimizer = torch.optim.SGD([
                {"params": model_client.network.parameters()},
                {"params": model_client.output.parameters()}
            ], lr=self.config['lr_client'])

            optimizer_u = torch.optim.SGD(
                model_client.embedding_user.parameters(),
                lr=self.config['lr_client'] / self.config['clients_sample_ratio'] * self.config['lr_eta']
            )

            optimizer_i = torch.optim.SGD(
                model_client.embedding_item.parameters(),
                lr=self.config['lr_client'] * self.config['num_items_train'] * self.config['lr_eta']
            )

            optimizers = [optimizer, optimizer_u, optimizer_i]

            # 训练用户模型
            model_client.train()
            user_dataloader = self.instance_user_train_loader(all_train_data[user])

            for _ in range(self.config['local_epoch']):
                for batch in user_dataloader:
                    model_client = self.fed_train_single_batch(model_client, batch, optimizers)

            # 保存用户模型参数
            self.client_model_params[user] = {
                k: v.data.cpu() for k, v in model_client.state_dict().items()
                if k != 'embedding_item.weight'
            }

            round_participant_params[user] = {
                'embedding_item.weight': model_client.embedding_item.weight.data.cpu()
            }

            # 定期清理内存
            if len(round_participant_params) % 100 == 0:
                self._release_memory()

        # 聚合参数
        self.aggregate_clients_params(round_participant_params, item_cv_features, item_text_features)

        # 清理本轮内存
        self._release_memory()

    def fed_evaluate(self, evaluate_data, item_cv_features, item_text_features, item_ids_map, is_test=True):
        """评估联邦学习模型"""
        print(f"\n开始{'测试集' if is_test else '验证集'}评估")

        # 将特征数据转换为张量
        item_cv_content = self._move_to_device(torch.tensor(item_cv_features))
        item_text_content = self._move_to_device(torch.tensor(item_text_features))

        # 获取唯一用户ID
        user_ids = evaluate_data['uid'].unique()
        user_item_preds = {}

        # 批量处理预测
        batch_size = 128
        for i in range(0, len(user_ids), batch_size):
            batch_users = user_ids[i:i + batch_size]

            for user in batch_users:
                # 使用用户模型进行预测
                user_model = self.client_model
                if user in self.client_model_params:
                    user_param_dict = copy.deepcopy(self.client_model.state_dict())
                    for key, param in self.client_model_params[user].items():
                        user_param_dict[key] = param.cuda()
                    user_model.load_state_dict(user_param_dict)

                user_model.eval()
                with torch.no_grad(), torch.cuda.amp.autocast():
                    predictions = user_model(item_cv_content, item_text_content)
                    user_item_preds[user] = predictions.cpu()

            # 定期清理内存
            if i % (batch_size * 10) == 0:
                self._release_memory()

        # 计算评估指标
        recall, precision, ndcg = self._compute_metrics(
            evaluate_data,
            user_item_preds,
            item_ids_map,
            self.config['recall_k']
        )

        return recall, precision, ndcg

    def _compute_metrics(self, data, predictions, item_map, k_values):
        """计算评估指标"""
        recalls = []
        precisions = []
        ndcgs = []

        for k in k_values:
            # 计算每个k值的指标
            recall_k = self._compute_recall_at_k(data, predictions, item_map, k)
            precision_k = self._compute_precision_at_k(data, predictions, item_map, k)
            ndcg_k = self._compute_ndcg_at_k(data, predictions, item_map, k)

            recalls.append(recall_k)
            precisions.append(precision_k)
            ndcgs.append(ndcg_k)

        return recalls, precisions, ndcgs