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


# Checkpoints
def save_checkpoint(model, model_dir):
    torch.save(model.state_dict(), model_dir)


def resume_checkpoint(model, model_dir, device_id):
    state_dict = torch.load(model_dir,
                            map_location=lambda storage, loc: storage.cuda(device=device_id))  # ensure all storage are on gpu
    model.load_state_dict(state_dict)


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
    train_id_data = pd.read_csv(data_file + '/train_data.csv', header=None)
    train_id_data.columns = ['iid', 'uid']



    # train_item_ids = list(set(train_id_data.iloc[:, 0]))

    train_img_features = img_feature_array[train_id_data['iid'].values]
    train_text_features = text_feature_array[train_id_data['iid'].values]
    # train_item_ids_map = {iid: i for i, iid in enumerate(train_item_ids)}

    # 读取测试数据
    test_id_data = pd.read_csv(data_file + '/test_data.csv', header=None)
    test_id_data.columns = ['iid', 'uid']

    # test_item_ids = list(set(test_id_data.iloc[:, 0]))

    test_img_features = img_feature_array[test_id_data['iid'].values]
    test_text_features = text_feature_array[test_id_data['iid'].values]
    # test_item_ids_map = {iid: i for i, iid in enumerate(test_item_ids)}

    # 读取验证数据
    valid_id_data = pd.read_csv(data_file + '/val_data.csv', header=None)
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

    data_dict = {'train_data':train_id_data,'train_img_features':train_img_features,'train_text_features':train_text_features,'train_item_ids_map':train_item_ids,
                 'test_data':test_id_data,'test_img_features':test_img_features,'test_text_features':test_text_features,'test_item_ids_map':test_item_ids,
                 'valid_data':valid_id_data,'valid_img_features':valid_img_features,'valid_text_features':valid_text_features,'vali_item_ids_map':valid_item_ids,
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


def compute_metrics(evaluate_data, user_item_preds, item_ids_map, recall_k):
    """compute evaluation metrics for cold-start items."""
    """input:
    evaluate_data: (uid, iid) dataframe.
    user_item_preds: cold-start item prediction for each user.
    item_ids_map: {ori_id: reindex_id} dict.
    recall_k: top_k metrics.
       output:
    recall, precision, ndcg
    """
    pred = []
    target_rows, target_columns = [], []
    temp = 0
    for uid in user_item_preds.keys():
        # predicted location for each user.
        user_pred = user_item_preds[uid]
        _, user_pred_all = user_pred.topk(k=recall_k[-1])
        user_pred_all = user_pred_all.cpu()
        pred.append(user_pred_all.tolist())

        # cold-start items real location for each user.
        user_cs_items = list(evaluate_data[evaluate_data['uid']==uid]['iid'].unique())
        # record sparse target matrix indexes.
        for item in user_cs_items:
            target_rows.append(temp)
            target_columns.append(item_ids_map[item])
        temp += 1
    pred = np.array(pred)
    target = sp.coo_matrix(
        (np.ones(len(evaluate_data)),
         (target_rows, target_columns)),
        shape=[len(pred), len(item_ids_map)]
    )
    recall, precision, ndcg = [], [], []
    idcg_array = np.arange(recall_k[-1]) + 1
    idcg_array = 1 / np.log2(idcg_array + 1)
    idcg_table = np.zeros(recall_k[-1])
    for i in range(recall_k[-1]):
        idcg_table[i] = np.sum(idcg_array[:(i + 1)])
    for at_k in recall_k:
        preds_k = pred[:, :at_k]
        x = sp.lil_matrix(target.shape)
        x.rows = preds_k
        x.data = np.ones_like(preds_k)

        z = np.multiply(target.todense(), x.todense())
        recall.append(np.mean(np.divide((np.sum(z, 1)), np.sum(target, 1))))
        precision.append(np.mean(np.sum(z, 1) / at_k))

        x_coo = sp.coo_matrix(x.todense())
        rows = x_coo.row
        cols = x_coo.col
        target_csr = target.tocsr()
        dcg_array = target_csr[(rows, cols)].A1.reshape((preds_k.shape[0], -1))
        dcg = np.sum(dcg_array * idcg_array[:at_k].reshape((1, -1)), axis=1)
        idcg = np.sum(target, axis=1) - 1
        idcg[np.where(idcg >= at_k)] = at_k - 1
        idcg = idcg_table[idcg.astype(int)]
        ndcg.append(np.mean(dcg / idcg))

    return recall, precision, ndcg



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

if __name__ == '__main__':
    data_dict = load_data('../data/Bili_Food')
    print(len(data_dict['user_ids']))
    print(len(data_dict['item_ids']))
