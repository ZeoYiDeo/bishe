import torch
from pandas.core.internals.concat import JoinUnit
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import BertTokenizer, BertModel, ViTFeatureExtractor, ViTModel,AutoTokenizer, AutoModelForMaskedLM
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm

tokenizer = BertTokenizer.from_pretrained('../bert-base')
bert_model = BertModel.from_pretrained('../bert-base')
vit_extractor = ViTFeatureExtractor.from_pretrained('../vit-base-patch16-224')
vit_model = ViTModel.from_pretrained('../vit-base-patch16-224')


def load_data(data_path):
    image_dir = os.path.join(data_path, 'Bili_Food_cover')

    df = pd.read_csv(os.path.join(data_path, 'Bili_Food_item.csv'))
    image_names = list(set(df[0].values))
    text = df[2].values.tolist()

    data_dict = {'image_name':image_names, 'text':text}

    return data_dict

if __name__ == '__main__':
    data_path = '../data/Bili_Food'
    data_dict = load_data(data_path)
    image_names = data_dict['image_name']
    text = data_dict['text']

    bert_inputs = tokenizer(text[0], padding=True, return_tensors='pt')
    bert_outputs = bert_model(**bert_inputs)
    bert_last_hidden_state = bert_outputs.last_hidden_state


    image = Image.open(os.path.join(data_path, 'Bili_Food_cover', image_names[0], '.jpg'))
    vit_inputs = vit_extractor(images=image, return_tensors='pt')
    vit_outputs = vit_model(**vit_inputs)
    vit_last_hidden_state = vit_outputs.last_hidden_state

    print(bert_outputs, vit_outputs)