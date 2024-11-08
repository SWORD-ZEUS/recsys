import torch
from model import WideAndDeepModel
from data_utils import load_movielens_data, preprocess_data, get_dataloaders, split_data
from config import *
import pandas as pd
import numpy as np
import os

def test_trained_model():
    # 加载数据
    df = load_movielens_data(MOVIELENS_DATA_SIZE)
    df, user_mapping, item_mapping = preprocess_data(df)
    
    # 只需要测试集
    _, _, test_df = split_data(df)
    _, _, test_loader = get_dataloaders(test_df, test_df, test_df)
    
    # 初始化模型
    num_users = len(user_mapping)
    num_items = len(item_mapping)
    num_genres = len(df[ITEM_FEAT_COL][0])
    num_tags = len(df[TAG_COL][0])
    model = WideAndDeepModel(num_users, num_items, num_genres, num_tags).to(DEVICE)
    
    # 加载训练好的权重
    model.load_state_dict(torch.load(LOAD_WEIGHT_PATH))
    model.eval()
    
    # 在测试集上进行预测
    predictions = []
    true_ratings = []
    users = []
    items = []
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            user = batch['user'].to(DEVICE)
            item = batch['item'].to(DEVICE)
            genre = batch['genre'].to(DEVICE)
            tag = batch['tag'].to(DEVICE)
            rating = batch['rating']
            
            logits = model(user, item, genre, tag)
            _, predicted = torch.max(logits, 1)
            
            # 将预测的类别索引转换回评分值
            predicted_ratings = [RATING_CLASSES[idx] for idx in predicted.cpu().numpy()]
            
            predictions.extend(predicted_ratings)
            true_ratings.extend(rating.numpy())
            users.extend(batch['user'].numpy())
            items.extend(batch['item'].numpy())
            
            # 计算准确率
            total += rating.size(0)
            rating_indices = torch.tensor([RATING_CLASSES.index(r) for r in rating])
            correct += (predicted.cpu() == rating_indices).sum().item()
    
    # 计算评估指标
    accuracy = 100 * correct / total
    mse = np.mean((np.array(predictions) - np.array(true_ratings)) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(np.array(predictions) - np.array(true_ratings)))
    
    print(f"测试集结果：")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    # 保存预测结果
    results_df = pd.DataFrame({
        'user': users,
        'item': items,
        'true_rating': true_ratings,
        'predicted_rating': predictions
    })
    
    # 将用户ID和物品ID映射回原始ID
    reverse_user_mapping = {v: k for k, v in user_mapping.items()}
    reverse_item_mapping = {v: k for k, v in item_mapping.items()}
    results_df['user'] = results_df['user'].map(reverse_user_mapping)
    results_df['item'] = results_df['item'].map(reverse_item_mapping)
    
    # 保存结果
    test_results_dir = os.path.dirname(TEST_RESULTS_PATH)
    if not os.path.exists(test_results_dir):
        os.makedirs(test_results_dir)
    results_df.to_csv(TEST_RESULTS_PATH, index=False)
    print(f"预测结果已保存到：{TEST_RESULTS_PATH}")

if __name__ == "__main__":
    test_trained_model() 