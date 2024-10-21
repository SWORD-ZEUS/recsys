import torch

# 常量
USER_COL = 'userId'
ITEM_COL = 'movieId'
TAG_COL = 'tag'
RATING_COL = 'rating'
PREDICT_COL = 'prediction'
ITEM_FEAT_COL = 'genres'
SEED = 42

# 参数
TOP_K = 10
MOVIELENS_DATA_SIZE = "100k"
EVALUATE_WHILE_TRAINING = True
RANDOM_SEED = SEED

# 模型参数
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
EMBEDDING_DIM = 32
HIDDEN_UNITS = [64, 32]
DROPOUT = 0.2

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据路径
MERGED_DATA_PATH = "/storage/zhuangkai/data/recsys/ml-latest-small/merged_data.csv"

# 保存路径
SAVE_PATH = "/storage/zhuangkai/data/recsys/weights/wide_and_deep_model.pth"

