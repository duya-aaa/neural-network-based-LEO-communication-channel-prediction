# 导入必要的工具包
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
 

# 定义RNN模型
class Rnn(nn.Module):
    def __init__(self, input_size,hidden_size,num_layers):
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
        self.out = nn.Linear(hidden_size, num_layers)
 
    # 定义前向传播函数
    def forward(self, x, h_0):
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
        return torch.stack(outs, dim=1), h_n
    

if __name__ == '__main__':
    TIME_STEP = 10
    INPUT_SIZE = 1
    LR = 0.02
    model = Rnn(INPUT_SIZE)
    print(model)

    # 此处使用的是均方误差损失
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)# Adam梯度下降
 
    h_state = None  # 初始化h_state为None
 
    for step in range(300):
        # 人工生成输入和输出,输入x.size=[1,10,1],输出y.size=[1,10,1]
        start, end = step * np.pi, (step + 1) * np.pi
        # np.linspace生成一个指定大小，指定数据区间的均匀分布序列，TIME_STEP是生成数量
        steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)
        # print("steps", steps)
        x_np = np.sin(steps)
        y_np = np.cos(steps)
        # print("x_np,y_np", x_np, y_np)
        # 从numpy.ndarray创建一个张量 np.newaxis增加新的维度
        x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])
        y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])
        # print("x,y", x,y)
 
        # 将x通过网络,长度为10的序列通过网络得到最终隐藏层状态h_state和长度为10的输出prediction:[1,10,1]
        prediction, h_state = model(x, h_state)
        h_state = h_state.data
        # 这一步只取了h_state.data.因为h_state包含.data和.grad 舍弃了梯度
        # print("precision, h_state.data", prediction, h_state)
        # print("prediction.size(), h_state.size()", prediction.size(), h_state.size())
 
        # 反向传播
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        # 更新优化器参数
        optimizer.step()
    # 对最后一次的结果作图查看网络的预测效果
    plt.plot(steps, y_np.flatten(), 'r-')
    plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
    plt.show()


    