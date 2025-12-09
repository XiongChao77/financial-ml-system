import torch
import torch.nn as nn

class LSTM1D(nn.Module):
    """
    双向 LSTM 支持版本
    输入: [B, T, F]   输出: logits [B, n_classes]
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, n_classes=3, p_drop=0.3, bidirectional=True):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # 1. LSTM 层
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True,
            dropout=p_drop if num_layers > 1 else 0,
            bidirectional=bidirectional  # 【修改点1】开启双向
        )
        
        # 2. Dropout 层
        self.dropout = nn.Dropout(p_drop)
        
        # 3. 全连接分类层
        # 【修改点2】如果是双向，LSTM 输出的特征维度会翻倍
        fc_input_dim = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_input_dim, n_classes)

    def forward(self, x):                  # x: [B,T,F]
        # output: [B, T, num_directions * hidden_size]
        # h_n:    [num_layers * num_directions, B, hidden_size]
        output, (h_n, c_n) = self.lstm(x)
        
        if self.bidirectional:
            # 【修改点3】双向时的隐藏状态处理
            # h_n 的形状是 [num_layers * 2, B, hidden_size]
            # 这种布局下：
            # h_n[-2] 是最后一层的前向 (Forward) 最终状态
            # h_n[-1] 是最后一层的后向 (Backward) 最终状态
            
            # 我们将这两个方向的状态拼接起来，得到包含完整上下文的向量
            last_hidden_state = torch.cat((h_n[-2], h_n[-1]), dim=1) # -> [B, hidden_size * 2]
        else:
            # 单向时，直接取最后一层的状态
            last_hidden_state = h_n[-1] # -> [B, hidden_size]
        
        out = self.dropout(last_hidden_state)
        return self.fc(out)