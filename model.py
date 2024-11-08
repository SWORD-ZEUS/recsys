import torch
import torch.nn as nn
from config import *

class WideAndDeepModel(nn.Module):
    def __init__(self, num_users, num_items, num_genres, num_tags):
        super(WideAndDeepModel, self).__init__()
        
        # Wide part
        self.wide = nn.Linear(num_users + num_items, HIDDEN_UNITS[-1])
        
        # Deep part
        self.user_embedding = nn.Embedding(num_users, EMBEDDING_DIM)
        self.item_embedding = nn.Embedding(num_items, EMBEDDING_DIM)
        
        deep_input_dim = EMBEDDING_DIM * 2 + num_genres + num_tags
        self.deep_layers = nn.ModuleList()
        for i in range(len(HIDDEN_UNITS)):
            if i == 0:
                self.deep_layers.append(nn.Linear(deep_input_dim, HIDDEN_UNITS[i]))
            else:
                self.deep_layers.append(nn.Linear(HIDDEN_UNITS[i-1], HIDDEN_UNITS[i]))
            self.deep_layers.append(nn.ReLU())
            self.deep_layers.append(nn.Dropout(DROPOUT))
        
        # 修改最后一层为分类层（rating范围通常是1-5，所以是5个类别）
        self.final = nn.Linear(HIDDEN_UNITS[-1] * 2, NUM_CLASSES)
        
    def forward(self, user, item, genre, tag):
        # 添加检查
        assert torch.all(user >= 0) and torch.all(user < self.user_embedding.num_embeddings), \
            f"User indices out of bounds. Max user index: {torch.max(user)}, num_embeddings: {self.user_embedding.num_embeddings}"
        assert torch.all(item >= 0) and torch.all(item < self.item_embedding.num_embeddings), \
            f"Item indices out of bounds. Max item index: {torch.max(item)}, num_embeddings: {self.item_embedding.num_embeddings}"
        
        # Wide part
        wide_input = torch.cat([
            nn.functional.one_hot(user, num_classes=self.user_embedding.num_embeddings).float(),
            nn.functional.one_hot(item, num_classes=self.item_embedding.num_embeddings).float()
        ], dim=1).to(user.device)
        wide_output = self.wide(wide_input)
        
        # Deep part
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        deep_input = torch.cat([user_emb, item_emb, genre, tag], dim=1)
        
        for layer in self.deep_layers:
            deep_input = layer(deep_input)
        
        # Combine wide and deep
        combined = torch.cat([deep_input, wide_output], dim=1)
        logits = self.final(combined)
        return logits
