# MovieLens 推荐系统

基于Wide&Deep模型的电影推荐系统，使用MovieLens数据集进行训练和评估。

## 项目结构 
.
├── README.md
├── requirements.txt
├── config.py # 配置文件
├── data_utils.py # 数据处理工具
├── model.py # 模型定义
├── train.py # 训练脚本
├── test_model.py # 测试脚本
└── main.py # 主程序入口

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA (可选，用于GPU加速)

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/SWORD-ZEUS/recsys.git
cd recsys
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 数据准备

1. 下载MovieLens数据集并解压
2. 运行数据预处理脚本：
```bash
python process_data.py
```

## 模型说明

使用Wide&Deep模型进行推荐，包含以下特征：
- 用户ID嵌入
- 电影ID嵌入
- 电影类型特征
- 标签特征

评分预测被视为10分类问题，对应评分：[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

## 训练模型

```bash
python main.py
```

训练过程中会：
- 自动保存最佳模型
- 记录训练日志（可使用TensorBoard查看）
- 在验证集上评估模型性能

## 测试模型

```bash
python test_model.py
```

测试结果包括：
- 准确率（Accuracy）
- 均方误差（MSE）
- 均方根误差（RMSE）
- 平均绝对误差（MAE）

## 配置说明

主要配置参数（在config.py中）：
- BATCH_SIZE: 批次大小
- EPOCHS: 训练轮数
- LEARNING_RATE: 学习率
- EMBEDDING_DIM: 嵌入维度
- HIDDEN_UNITS: 隐藏层单元数
- DROPOUT: Dropout比率

## 文件说明

- `config.py`: 配置参数和常量
- `data_utils.py`: 数据加载和预处理函数
- `model.py`: Wide&Deep模型定义
- `train.py`: 模型训练和评估函数
- `test_model.py`: 模型测试和结果输出
- `main.py`: 主程序入口

## 许可证

MIT License

## 作者

SWORD-ZEUS
