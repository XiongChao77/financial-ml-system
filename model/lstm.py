import torch.nn as nn
import torch.nn.functional as F
import math
# --- 新增模型：LSTM ---
class LSTM1D(nn.Module):
    """
    一个用于时间序列分类的标准 PyTorch LSTM 模型。
    输入: [B, T, F]   输出: logits [B, n_classes]
    
    T = 窗口大小 (e.g., 256)
    F = 特征数量 (e.g., 9)
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, n_classes=3, p_drop=0.3):
        super().__init__()
        
        # 1. LSTM 层
        # batch_first=True 表示输入/输出是 [Batch, Sequence Length, Features]
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True,
            dropout=p_drop if num_layers > 1 else 0, # 仅在多层时使用 dropout
            bidirectional=False # 单向 LSTM
        )
        
        # 2. Dropout 层
        self.dropout = nn.Dropout(p_drop)
        
        # 3. 全连接分类层
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x):                  # x: [B,T,F]
        # LSTM 的输出：
        # - out: [B, T, hidden_size]，包含序列中每个时间步的隐藏状态
        # - (h_n, c_n): 最后一个时间步的隐藏状态和细胞状态
        # 我们只需要最后一个时间步的隐藏状态 h_n 来进行分类。
        
        out, (h_n, c_n) = self.lstm(x)
        
        # 1. 使用最后一个时间步的隐藏状态 h_n 作为特征。
        # h_n 的形状是 [num_layers, B, hidden_size]
        # 取最后一层的隐藏状态: h_n[-1] -> [B, hidden_size]
        last_hidden_state = h_n[-1] 
        
        # 2. 或者，使用 'out' 中最后一个时间步的状态: out[:,-1,:]
        # last_step_out = out[:, -1, :] 
        
        out = self.dropout(last_hidden_state)
        
        return self.fc(out)                # [B, n_classes]