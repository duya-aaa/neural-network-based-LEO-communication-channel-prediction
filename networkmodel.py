import torch.nn as nn
import torch.nn.functional as F
import torch

class CNNAugmentedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
        super(CNNAugmentedLSTM, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=2, out_channels=8, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout_rate, batch_first=True)
        self.bn = nn.BatchNorm1d(num_features=8)
        self.fc = nn.Linear(hidden_size, 512)  # 512
        self.output = nn.Linear(512, 1)  # 512

    def forward(self, x):
        # 1D Conv expects: [batch, channels, seq_len]
        x = self.conv1d(x)
        #batch nomorlization
        x = self.bn(x)
        # LSTM expects: [batch, seq_len, features]
        x = x.permute(0, 2, 1)
        x, (ht, ct) = self.lstm(x)
        
        # Use the last hidden state
        x = ht[-1]
        
        # Pass through the fully connected layers
        x = F.leaky_relu(self.fc(x))
        x = self.output(x)
        return x


class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size_lstm, hidden_size_fc, output_size, num_layers, dropout_rate):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size=hidden_size_lstm, num_layers=num_layers, dropout=dropout_rate, batch_first=True)
        self.fc = nn.Linear(hidden_size_lstm, output_size)

        self.fc = nn.Linear(hidden_size_lstm, hidden_size_fc)
        # 输出层，将512维度映射到output_size维度
        self.output = nn.Linear(hidden_size_fc, output_size)

    def forward(self, x):

        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        out = self.output(out)
        return out


class AT_CNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
        super(AT_CNNLSTM, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=2, out_channels=8, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout_rate, batch_first=True)
        self.bn = nn.BatchNorm1d(num_features=8)
        # 添加自注意力层，假设使用单个注意力头，注意力头的数量可以根据需要进行调整
        self.self_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, 512)
        self.output = nn.Linear(512, 1)

    def forward(self, x):
        # 1D Conv
        assert not torch.isinf(x).any(), "Input contains inf"
        x = self.conv1d(x)
        if torch.isnan(x).any():
            raise ValueError("NaN detected in conv1d output")
        # Batch normalization
        x = self.bn(x)
        # Permute for LSTM
        x = x.permute(0, 2, 1)
        x, (ht, ct) = self.lstm(x)
        
        # 引入自注意力机制，注意：输入和输出的维度需要匹配
        x, _ = self.self_attention(x, x, x)
        
        # 使用LSTM的最后一个隐藏状态
        x = ht[-1]
        
        # 通过全连接层
        x = F.leaky_relu(self.fc(x))
        x = self.output(x)
        return x

# 定义RNN模型
class Rnn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Rnn, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.hidden_size = hidden_size
        self.fc = nn.Linear(hidden_size, output_size)
 
    def forward(self, x):
        
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(torch.device("cuda"))
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

'''class Rnn(nn.Module):
    def __init__(self, input_size,hidden_size,num_layers,output_size=1):
        super(Rnn, self).__init__()# 调用nn.Module的构造函数,进行重载
        # 定义RNN网络
        ## hidden_size是自己设置的，取值都是32,64,128这样来取值
        ## num_layers是隐藏层数量，超过2层那就是深度循环神经网络了
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True  # 输入形状为[批量大小, 数据序列长度, 特征维度]，若为False则为[数据序列长度, 批量大小, 特征维度]
        )
        # 定义全连接层
        self.out = nn.Linear(hidden_size, output_size)
 
    # 定义前向传播函数
    def forward(self, x, h_0=None):# 初始隐藏层状态
        if h_0 is None:
            h_0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(x.device)
        
        r_out, h_n = self.rnn(x, h_0)
        # print("数据输出结果；隐藏层数据结果", r_out, h_n)
        # print("r_out.size()， h_n.size()", r_out.size(), h_n.size())
        outs = []
        # r_out.size=[1,10,32]即将一个长度为10的序列的每个元素都映射到隐藏层上
        for time in range(r_out.size(1)):
            # print("映射", r_out[:, time, :])
            # 依次抽取序列中每个单词,将之通过全连接层并输出.r_out[:, 0, :].size()=[1,32] -> [1,1]
            outs.append(self.out(r_out[:, time, :]))
            # print("outs", outs)
        # stack函数在dim=1上叠加:10*[1,1] -> [1,10,1] 同时h_n已经被更新
        return torch.stack(outs, dim=1), h_n'''