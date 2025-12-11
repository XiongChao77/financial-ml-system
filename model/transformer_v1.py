import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # shape: [max_len, d_model] -> [1, max_len, d_model] 方便 batch 广播
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: [Batch, Seq_Len, d_model]
        # 截取对应长度的位置编码相加
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class Transformer1D(nn.Module):
    """
    用于时间序列分类的 Transformer Encoder 模型
    输入: [Batch, T, F]
    输出: [Batch, n_classes]
    """
    def __init__(self, input_size, n_classes, d_model=64, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1, max_len=5000):
        super().__init__()
        
        # 1. 特征投影层: 将原始特征维度 F 映射到 d_model
        # 这一步是必须的，因为 Transformer 的 d_model 通常大于原始特征数
        self.input_proj = nn.Linear(input_size, d_model)
        
        # 2. 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        
        # 3. Transformer Encoder
        # batch_first=True 确保输入输出格式为 [Batch, Seq, Feature]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. 分类头
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x):
        # x: [Batch, Seq_Len, Features]
        
        # 投影到隐层维度
        x = self.input_proj(x) # -> [Batch, Seq, d_model]
        
        # 加上位置信息
        x = self.pos_encoder(x)
        
        # 通过 Transformer 层
        # Mask 说明: 对于时间序列分类(判断整个窗口的性质)，通常不需要 mask (即允许看到未来)，
        # 因为我们是用过去 256 个点来判断当前的 Label，这 256 个点相对于 Label 都是"过去"。
        x = self.transformer_encoder(x) # -> [Batch, Seq, d_model]
        
        # 聚合策略: Global Average Pooling (GAP)
        # 对时间维度 (dim=1) 求平均，这通常比只取最后一个时间步更鲁棒
        x = x.mean(dim=1) # -> [Batch, d_model]
        
        # 分类
        out = self.fc(x) # -> [Batch, n_classes]
        return out