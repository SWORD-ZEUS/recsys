import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset, DataLoader
import torch
from config import *

def load_movielens_data(size='100k'):
    # 读取合并后的数据
    merged_data_path = MERGED_DATA_PATH
    df = pd.read_csv(merged_data_path)

    # 返回 DataFrame
    return df

def preprocess_data(df):
    # 对 genre 进行多标签二值化
    mlb_genres = MultiLabelBinarizer()
    genres = mlb_genres.fit_transform(df[ITEM_FEAT_COL].apply(lambda x: x.split('|')))
    df[ITEM_FEAT_COL] = genres.tolist()

    # 对 tag 进行多标签二值化
    mlb_tags = MultiLabelBinarizer()
    tags = mlb_tags.fit_transform(df[TAG_COL].apply(lambda x: x.split('|')))
    df[TAG_COL] = tags.tolist()

    # 创建用户和物品的映射
    user_mapping = {user_id: idx for idx, user_id in enumerate(df[USER_COL].unique())}
    item_mapping = {item_id: idx for idx, item_id in enumerate(df[ITEM_COL].unique())}

    # 将原始 ID 替换为连续索引
    df[USER_COL] = df[USER_COL].map(user_mapping)
    df[ITEM_COL] = df[ITEM_COL].map(item_mapping)

    return df, user_mapping, item_mapping

def split_data(df, test_size=0.1):
    return train_test_split(df, test_size=test_size, random_state=RANDOM_SEED)

class MovieLensDataset(Dataset):
    def __init__(self, df):
        self.users = df[USER_COL].values
        self.items = df[ITEM_COL].values
        self.ratings = df[RATING_COL].values
        self.genres = df[ITEM_FEAT_COL].values
        self.tags = df[TAG_COL].values  # 添加对 tags 的处理

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return {
            'user': torch.tensor(self.users[idx], dtype=torch.long),
            'item': torch.tensor(self.items[idx], dtype=torch.long),
            'genre': torch.tensor(self.genres[idx], dtype=torch.float),
            'tag': torch.tensor(self.tags[idx], dtype=torch.float),  # 添加 tags
            'rating': torch.tensor(self.ratings[idx], dtype=torch.float)
        }

def get_dataloaders(train_df, test_df):
    train_dataset = MovieLensDataset(train_df)
    test_dataset = MovieLensDataset(test_df)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    return train_loader, test_loader
