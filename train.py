from torch.utils.tensorboard import SummaryWriter
import torch
import os

def savemodel(model, epoch, model_type):
    filename = f'model_epoch_{epoch}.pth'
    directory = os.path.join('channel prediction/save_net/rnn', filename)

def train_model(model, train_loader, criterion, optimizer, epochs, device, model_type):
    writer = SummaryWriter('runs/train')
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for i, (sequences, targets) in enumerate(train_loader):
            sequences = sequences.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            # outputs = torch.squeeze(outputs)# 去除输出中的单维条目，让输出的形状与目标匹配
            loss = criterion(outputs, targets)# 一个2维，一个100维，需要修改
            loss.backward()# 反向传播计算梯度（这行和下面一行都不是很懂怎么计算的）
            optimizer.step()# 根据计算出的梯度更新模型的权重
            total_loss += loss.item()
            if i % 100 == 0:
                writer.add_scalar('Train Loss', loss.item(), epoch * len(train_loader) + i)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader)}')
    writer.close()
    savemodel(model, epoch, model_type)