from data_utils import load_movielens_data, preprocess_data, split_data, get_dataloaders
from model import WideAndDeepModel
from train import train_model
from config import *

def main():
    # 加载数据
    df = load_movielens_data(MOVIELENS_DATA_SIZE)
    df, user_mapping, item_mapping = preprocess_data(df)
    
    # 分割数据
    train_df, valid_df, test_df = split_data(df)
    
    # 准备数据加载器
    train_loader, valid_loader, test_loader = get_dataloaders(train_df, valid_df, test_df)
    
    # 初始化模型
    num_users = len(user_mapping)  # 使用映射后的用户数量
    num_items = len(item_mapping)   # 使用映射后的物品数量
    num_genres = len(df[ITEM_FEAT_COL][0])
    num_tags = len(df[TAG_COL][0])  # 获取 tag 的数量
    model = WideAndDeepModel(num_users, num_items, num_genres, num_tags).to(DEVICE)
    
    # 训练模型
    train_model(model, train_loader, valid_loader, test_loader, EPOCHS)
    
    # 保存模型
    torch.save(model.state_dict(), SAVE_PATH)

if __name__ == "__main__":
    main()
