"""
    Some handy functions for pytroch model training ...
"""
import torch
import logging
import numpy as np
import scipy.sparse as sp
import copy
from transformers import BertTokenizer, BertModel, CLIPModel, CLIPProcessor
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm



# Hyper params
def use_cuda(enabled, device_id=0):
    if enabled:
        assert torch.cuda.is_available(), 'CUDA is not available'
        torch.cuda.set_device(device_id)


def use_optimizer(network, params):
    if params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(),
                                    lr=params['sgd_lr'],
                                    momentum=params['sgd_momentum'],
                                    weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), 
                                     lr=params['lr'],
                                     weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(network.parameters(),
                                        lr=params['rmsprop_lr'],
                                        alpha=params['rmsprop_alpha'],
                                        momentum=params['rmsprop_momentum'])
    return optimizer


def initLogging(logFilename):
    """Init for logging
    """
    logging.basicConfig(
                    level    = logging.DEBUG,
                    format='%(asctime)s-%(levelname)s-%(message)s',
                    datefmt  = '%y-%m-%d %H-%M',
                    filename = logFilename,
                    filemode = 'w');
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def tfidf(R):
    row = R.shape[0]
    col = R.shape[1]
    Rbin = R.copy()
    Rbin[Rbin != 0] = 1.0
    R = R + Rbin
    tf = R.copy()
    tf.data = np.log(tf.data)
    idf = np.sum(Rbin, 0)
    idf = np.log(row / (1 + idf))
    idf = sp.spdiags(idf, 0, col, col)
    return tf * idf


def process_and_save_image_features_vit(image_folder, output_pt_path):
    # 加载 CLIP ViT 模型和预处理器
    vit_model = CLIPModel.from_pretrained("../CLIP-vit-large-patch14").to('cuda:0').eval()
    vit_processor = CLIPProcessor.from_pretrained("../CLIP-vit-large-patch14")

    features_dict = {}

    # 遍历文件夹中的所有图片
    for img_name in tqdm(os.listdir(image_folder), desc="Processing images"):
        img_path = os.path.join(image_folder, img_name)

        try:
            # 加载图片并转换为 RGB 格式
            image = Image.open(img_path).convert("RGB")

            # 预处理图片
            inputs = vit_processor(images=image, return_tensors="pt", padding=True).to('cuda:0')

            # 提取特征
            with torch.no_grad():
                outputs = vit_model.get_image_features(**inputs)
                features = outputs.cpu().squeeze(0)  # 转换到 CPU 并移除多余维度
            # 使用图片名称（去掉扩展名）作为 ID
            img_id = int(os.path.splitext(img_name)[0])
            # print(img_id)
            features_dict[img_id] = features

        except Exception as e:
            print(f"Error processing image {img_name}: {e}")
            continue

    # 保存为 .pt 文件
    torch.save(features_dict, output_pt_path)
    print(f"Features saved to {output_pt_path}")

def process_and_save_text_features_bert(text_file, output_pt_path):
    # 加载 BERT 模型和预处理器
    bert_model = BertModel.from_pretrained("../bert-base").to('cuda:0').eval()
    bert_tokenizer = BertTokenizer.from_pretrained("../bert-base")

    features_dict = {}
    # 读取文本数据
    text_data = pd.read_csv(text_file,header=None,usecols=[0,2])
    # 遍历文本数据
    for i in tqdm(range(len(text_data)), desc="Processing text"):
        text = text_data.iloc[i,1]
        # 预处理文本
        inputs = bert_tokenizer(text, return_tensors="pt", padding=True).to('cuda:0')
        # 提取特征
        with torch.no_grad():
            outputs = bert_model(**inputs)
            features = outputs.last_hidden_state[:,0,:].cpu().squeeze(0)  # 转换到 CPU 并移除多余维度
        # 使用文本 ID 作为 ID
        text_id = int(text_data.iloc[i,0])
        # print(text_id)
        features_dict[text_id] = features

    # 保存为 .pt 文件
    torch.save(features_dict, output_pt_path)
    print(f"Features saved to {output_pt_path}")

def load_data(data_file):

    id_data = pd.read_csv(data_file+'/Bili_Food_pair.csv',header=None,usecols=[0,1])
    id_data.columns = ['iid', 'uid']


    user_ids = list(set(id_data.iloc[:,1]))
    item_ids = list(set(id_data.iloc[:,0]))

    item_img_dict = torch.load("../data/Bili_Food/Bili_Food_vit.pt")
    item_text_dict = torch.load("../data/Bili_Food/Bili_Food_bert.pt")

    item_img_features = {int(keys): v  for keys,v in item_img_dict.items()}
    item_text_features = {int(keys): v  for keys,v in item_text_dict.items()}

    # 获取排序后的物品ID列表（确保顺序一致）
    sorted_item_ids = sorted(item_ids)

    # 转换为二维张量矩阵 [num_items, feature_dim]
    img_feature_matrix = torch.stack([item_img_features[iid] for iid in sorted_item_ids])
    text_feature_matrix = torch.stack([item_text_features[iid] for iid in sorted_item_ids])

    # 如果需要numpy格式
    img_feature_array = img_feature_matrix.numpy()
    text_feature_array = text_feature_matrix.numpy()

    # 读取训练数据
    train_id_data = pd.read_csv(data_file + '/train.csv', header=None)
    train_id_data.columns = ['iid', 'uid']



    # train_item_ids = list(set(train_id_data.iloc[:, 0]))

    train_img_features = img_feature_array[train_id_data['iid'].values]
    train_text_features = text_feature_array[train_id_data['iid'].values]
    # train_item_ids_map = {iid: i for i, iid in enumerate(train_item_ids)}

    # 读取测试数据
    test_id_data = pd.read_csv(data_file + '/test.csv', header=None)
    test_id_data.columns = ['iid', 'uid']

    # test_item_ids = list(set(test_id_data.iloc[:, 0]))

    test_img_features = img_feature_array[test_id_data['iid'].values]
    test_text_features = text_feature_array[test_id_data['iid'].values]
    # test_item_ids_map = {iid: i for i, iid in enumerate(test_item_ids)}

    # 读取验证数据
    valid_id_data = pd.read_csv(data_file + '/vali.csv', header=None)
    valid_id_data.columns = ['iid', 'uid']

    # valid_item_ids = list(set(valid_id_data.iloc[:, 0]))

    valid_img_features = img_feature_array[valid_id_data['iid'].values]
    valid_text_features = text_feature_array[valid_id_data['iid'].values]
    # vali_item_ids_map = {iid: i for i, iid in enumerate(valid_item_ids)}


    train_item_ids = sorted(list(set(train_id_data.iloc[:, 0])))
    test_item_ids = sorted(list(set(test_id_data.iloc[:, 0])))
    valid_item_ids = sorted(list(set(valid_id_data.iloc[:, 0])))

    # 创建连续的映射，范围与配置匹配
    train_item_ids_map = {iid: i for i, iid in enumerate(train_item_ids)}
    test_item_ids_map = {iid: i for i, iid in enumerate(test_item_ids)}
    vali_item_ids_map = {iid: i for i, iid in enumerate(valid_item_ids)}

    # 添加验证代码
    assert len(train_item_ids_map) <= 13584, f"训练集映射大小 ({len(train_item_ids_map)}) 超过配置值 (13584)"
    assert len(test_item_ids_map) <= 2378, f"测试集映射大小 ({len(test_item_ids_map)}) 超过配置值 (2378)"
    assert len(vali_item_ids_map) <= 2378, f"验证集映射大小 ({len(vali_item_ids_map)}) 超过配置值 (2378)"

    data_dict = {'train_data':train_id_data,'train_img_features':train_img_features,'train_text_features':train_text_features,'train_item_ids_map':train_item_ids_map,
                 'test_data':test_id_data,'test_img_features':test_img_features,'test_text_features':test_text_features,'test_item_ids_map':test_item_ids_map,
                 'valid_data':valid_id_data,'valid_img_features':valid_img_features,'valid_text_features':valid_text_features,'vali_item_ids_map':vali_item_ids_map,
                 'user_ids':user_ids,'item_ids':item_ids
                 }
    return data_dict


def negative_sampling(train_data, num_negatives):
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


def compute_metrics(evaluate_data, user_item_preds, item_ids_map, recall_k, is_test=True):
    """compute evaluation metrics for cold-start items."""
    if not isinstance(recall_k, list) or not recall_k:
        recall_k = [20, 50, 100]

    max_k = max(recall_k)
    print(f"\nRecall@K values: {recall_k}")
    print(f"最大K值: {max_k}")

    print(f"评估数据大小: {len(evaluate_data)}")
    print(f"用户预测数量: {len(user_item_preds)}")
    print(f"物品映射大小: {len(item_ids_map)}")

    pred = []
    target_rows = []
    target_columns = []
    temp = 0

    valid_users = set(user_item_preds.keys())

    for uid in valid_users:
        try:
            user_pred = user_item_preds[uid]
            k = min(max_k, len(user_pred))

            _, user_pred_all = user_pred.topk(k=k)
            # 直接转换为Python列表
            user_pred_all = user_pred_all.cpu().tolist()

            # 确保预测值在有效范围内
            valid_preds = [p for p in user_pred_all if p < len(item_ids_map)]

            if valid_preds:
                # 填充到最大长度
                while len(valid_preds) < max_k:
                    valid_preds.append(0)
                pred.append(valid_preds[:max_k])  # 截取到max_k长度

                # 获取用户的实际物品
                user_items = evaluate_data[evaluate_data['uid'] == uid]['iid'].unique()
                valid_items = [item for item in user_items if item in item_ids_map]

                if valid_items:
                    for item in valid_items:
                        target_rows.append(temp)
                        target_columns.append(item_ids_map[item])
                temp += 1

        except Exception as e:
            print(f"处理用户 {uid} 时出错: {str(e)}")
            continue

    if not pred:
        print("警告：没有有效的预测结果")
        return [0.0] * len(recall_k), [0.0] * len(recall_k), [0.0] * len(recall_k)

    try:
        # 转换为numpy数组
        pred = np.array(pred)

        print(f"预测矩阵形状: {pred.shape}")
        print(f"目标矩阵行数: {temp}")
        print(f"目标矩阵列数: {len(item_ids_map)}")

        # 创建目标矩阵
        target = sp.coo_matrix(
            (np.ones(len(target_columns)),
             (target_rows, target_columns)),
            shape=[temp, len(item_ids_map)]
        ).tocsr()  # 转换为CSR格式以提高效率

        recall, precision, ndcg = [], [], []

        for at_k in recall_k:
            actual_k = min(at_k, pred.shape[1])
            preds_k = pred[:, :actual_k]

            # 创建预测矩阵
            x = sp.lil_matrix((temp, len(item_ids_map)))
            for i, user_preds in enumerate(preds_k):
                valid_indices = [p for p in user_preds if 0 <= p < len(item_ids_map)]
                if valid_indices:
                    x.rows[i] = valid_indices
                    x.data[i] = [1.0] * len(valid_indices)

            # 转换为CSR格式
            x = x.tocsr()

            # 计算交集
            intersect = target.multiply(x)

            # 计算recall
            target_sums = np.asarray(target.sum(axis=1)).flatten()
            target_sums[target_sums == 0] = 1  # 避免除零
            recall_scores = np.asarray(intersect.sum(axis=1)).flatten() / target_sums
            recall.append(float(np.mean(recall_scores)))

            # 计算precision
            precision_scores = np.asarray(intersect.sum(axis=1)).flatten() / actual_k
            precision.append(float(np.mean(precision_scores)))

            # 计算NDCG
            dcg = np.zeros(temp)
            idcg = np.zeros(temp)

            for i in range(actual_k):
                dcg += np.asarray(intersect[:, i].todense()).flatten() / np.log2(i + 2)

            for i in range(temp):
                relevant_count = int(target[i].sum())
                ideal_dcg = sum(1.0 / np.log2(j + 2) for j in range(min(relevant_count, actual_k)))
                idcg[i] = ideal_dcg if ideal_dcg > 0 else 1.0

            ndcg_score = np.mean(dcg / idcg)
            ndcg.append(float(ndcg_score))

        # 打印详细结果
        print("\n评估结果:")
        for i, k in enumerate(recall_k):
            print(f"@{k}:")
            print(f"  Recall: {recall[i]:.4f}")
            print(f"  Precision: {precision[i]:.4f}")
            print(f"  NDCG: {ndcg[i]:.4f}")

        return recall, precision, ndcg

    except Exception as e:
        print(f"计算指标时出错: {str(e)}")
        print(f"Debug信息:")
        print(f"pred.shape: {pred.shape if isinstance(pred, np.ndarray) else 'not an array'}")
        print(f"target_rows数量: {len(target_rows)}")
        print(f"target_columns数量: {len(target_columns)}")
        return [0.0] * len(recall_k), [0.0] * len(recall_k), [0.0] * len(recall_k)


def compute_regularization(model, parameter_label):
    reg_fn = torch.nn.MSELoss(reduction='mean')
    for name, param in model.named_parameters():
        if name == 'embedding_item.weight':
            reg_loss = reg_fn(param, parameter_label)
            return reg_loss
#定义损失函数：使用 torch.nn.MSELoss(reduction='mean') 定义一个均方误差损失函数 reg_fn。
# 遍历模型参数：通过 model.named_parameters() 遍历模型的所有命名参数，每次迭代得到一个参数的名称 name 和参数本身 param。
# 检查参数名称：判断参数名称是否为 embedding_item.weight。
# 计算并返回损失：如果参数名称匹配，使用 reg_fn 计算该参数与 parameter_label 之间的均方误差损失，并将结果存储在 reg_loss 中，然后立即返回该损失值。


