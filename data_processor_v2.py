import torch
from transformers import BertTokenizer, BertModel, CLIPModel, CLIPProcessor
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm

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

    id_data = pd.read_csv(data_file,header=None,usecols=[0,1])
    id_data.columns = ['iid', 'uid']


    user_ids = id_data.iloc[:,1].unique().tolist()
    item_ids = id_data.iloc[:,0].unique().tolist()

    item_img_dict = torch.load("../data/Bili_Food/Bili_Food_vit.pt")
    item_text_dict = torch.load("../data/Bili_Food/Bili_Food_bert.pt")

    item_img_features = {int(keys): v  for keys,v in item_img_dict.items()}
    item_text_features = {int(keys): v  for keys,v in item_text_dict.items()}

    data_dict = {'id_data': id_data, 'user_ids': user_ids, 'item_ids': item_ids,
                 'item_img_features': item_img_features, 'item_text_features': item_text_features,
                 }
    return data_dict


if __name__ == '__main__':
    # 图片文件夹路径
    image_folder = "../data/Bili_Movie/Bili_Movie_cover"
    # 输出文件路径
    output_pt_path = "../data/Bili_Movie/Bili_Movie_vit.pt"
    # 调用函数处理图片特征并保存
    process_and_save_image_features_vit(image_folder, output_pt_path)

    # 文本文件路径
    text_file = "../data/Bili_Movie/Bili_Movie_item.csv"
    # 输出文件路径
    output_pt_path = "../data/Bili_Movie/Bili_Movie_bert.pt"
    # 调用函数处理文本特征并保存
    process_and_save_text_features_bert(text_file, output_pt_path)
    # # data_dict = torch.load("../data/Bili_Food/Bili_Food_vit.pt")
    # # print(data_dict['1'])