# 导入必要的工具包

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_prepare import CustomDataset, LoadAndCreate_data, collate_fn
import torch.nn.functional as F
import argparse
import os
import pickle
from torch.utils.tensorboard import SummaryWriter
from train import train_model
from networkmodel import Rnn

def main(args):
    #选择cuda
    device = torch.device('cuda')
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    batch_size = args.batch_size
    input_size = args.input_size
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    dropout_rate = args.dropout_rate
    epochs = args.epochs
    lr = args.lr
    model_type = args.model_type

    DataSet_path = "datasets_std.pkl"
    ChannelData_path = "matlab/results-1300.mat"

    #DataSet_path = 'datasets.pkl'
    #ChannelData_path = 'CSI_data/results.mat'

    #是否存在dataset的路径，若不存在要新建
    if os.path.exists(DataSet_path):
        print("datasets exist. Proceeding to load data...")
        with open(DataSet_path, 'rb') as f:
            datasets = pickle.load(f)

        # 从字典中取回四个列表
        train_features = datasets['train_features_std']
        train_targets = datasets['train_targets_std']

    else:
        print("datasets do not exist. Proceeding to creat datasets...")

        train_features, train_targets = LoadAndCreate_data(ChannelData_path)# 这几个函数要自己改
        
        # 创建训练集和自定义数据集实例
    train_dataset = CustomDataset(train_features, train_targets)

        # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size, collate_fn=collate_fn, shuffle=True)

    # 实例化模型
    
    #model = LSTMNet(input_size=1, hidden_size=256, output_size=1)
    model = Rnn(input_size, hidden_size, num_layers)
    model = model.to(device)

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr)

    # 训练模型
    train_model(model, train_loader, criterion, optimizer, epochs, device, model_type)

#这部分还没改嗷
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for channel prediction")
    parser.add_argument("--batch_size", type=int, default=int(100), help=" Batch size of dataset")
    parser.add_argument("--input_size", type=int, default=10, help="Input size of LSTM/RNN layer")
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size of LSTM/RNN layer")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM/RNN")
    parser.add_argument("--dropout_rate", type=float, default=0.25, help="Dropout rate of LSTM")
    parser.add_argument("--lr", type = float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=300, help="Epoch of the training process")
    parser.add_argument("--model_type", type=str, default="RNN", help="Decide which model will be used")
    args = parser.parse_args()
    main(args)