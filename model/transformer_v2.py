import torch
import torch.nn as nn

class Transformer1D_V2(nn.Module):
    def __init__(self, input_size, n_classes, d_model=128, nhead=8, num_layers=3, 
                 dim_feedforward=512, dropout=0.1, max_len=5000):
        super().__init__()
        
        # 1. 特征投影
        self.input_proj = nn.Linear(input_size, d_model)
        
        # 【优化点1】: [CLS] Token
        # 创建一个可学习的参数，维度为 [1, 1, d_model]
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 【优化点2】: 可学习的位置编码 (Learnable PE)
        # 这里的 max_len + 1 是因为加了 CLS token
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len + 1, d_model))
        
        # 【优化点3 & 4】: Pre-Norm 和 GELU
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            activation='gelu',   # 使用 GELU
            batch_first=True,
            norm_first=True      # 使用 Pre-Norm 结构，训练更稳定
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout)
        
        # 分类头
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model), # 最后的归一化
            nn.Linear(d_model, n_classes)
        )

    def forward(self, x):
        # x: [Batch, Seq_Len, Features]
        b, s, _ = x.shape
        
        # 1. 投影
        x = self.input_proj(x) # -> [Batch, Seq, d_model]
        
        # 2. 拼接 [CLS] Token
        # 扩展 cls_token 到当前 batch 大小
        cls_tokens = self.cls_token.expand(b, -1, -1) 
        x = torch.cat((cls_tokens, x), dim=1) # -> [Batch, Seq+1, d_model]
        
        # 3. 加上位置编码 (截取当前长度)
        # 这里的 seq_len 变成了 s+1
        x += self.pos_embedding[:, :(s + 1)]
        x = self.dropout(x)
        
        # 4. Transformer 编码
        x = self.transformer_encoder(x)
        
        # 5. 取出 [CLS] 对应的输出 (第0个位置)
        cls_out = x[:, 0, :] # -> [Batch, d_model]
        
        # 6. 分类
        out = self.fc(cls_out)
        return out