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
LOAD_WEIGHT_PATH = "/storage/zhuangkai/data/recsys/weights/best_model.pth"
SAVE_DIR = "/storage/zhuangkai/data/recsys/weights"
TEST_RESULTS_PATH = "/storage/zhuangkai/data/recsys/results/test_results.csv"

# 日志路径
LOG_DIR = "/storage/zhuangkai/data/recsys/logs"

# 分类数（评分范围1-5，共5个类别）
NUM_CLASSES = 10

# 添加评分映射
RATING_CLASSES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

