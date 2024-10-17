import torch
import scipy.io
import numpy as np
import random
from torch.utils.data import Dataset
import torch.nn.functional as F
import pickle
from sklearn.preprocessing import StandardScaler
import joblib
from torch.nn.utils.rnn import pad_sequence

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature = self.data[idx]
        target = self.labels[idx]

        # 将数据转换为torch.Tensor（如果它们还不是）
        feature = torch.tensor(feature, dtype=torch.float32)  # 转换为Float
        target = torch.tensor(target, dtype=torch.float32)    # 同样转换为Float

        return feature, target
    
# 数据加载和预处理
def LoadAndCreate_data(channel_file):
    channel_data = scipy.io.loadmat(channel_file)
    feature_len = 30
    total_features = []
    total_targets = []

    for i in range(len(channel_data['results'])):
        CSI = channel_data['results'][i][0][0].flatten()
        ANGLE = channel_data['results'][i][0][1].flatten()
        ANGLE = np.degrees(ANGLE)  # 弧度转换为度
        data = np.vstack([CSI, ANGLE])
        inf_count_feature = 0
        inf_count_target = 0
        features = []
        targets = []
        # 遍历数据并构建特征和目标
        for k in range(data.shape[1] - feature_len):
            elevation = data[1, k]
            data_len = get_data_length(elevation)

            if data_len > 0:
                seq_feature = data[:, k: k + data_len]
                seq_target = data[0, k + feature_len]  # 当前的CSI值
                inf_count_feature += np.sum(np.isinf(seq_feature))
                inf_count_target += np.sum(np.isinf(seq_target))
                features.append(seq_feature)
                targets.append(seq_target)

        print('inf_count_feature', inf_count_feature)
        print('inf_count_target', inf_count_target)
        if inf_count_feature == 0 & inf_count_target == 0:
            total_features.extend(features)
            total_targets.extend(targets)
            print("no inf data in this user, accept")
        else:
            print("find inf data, discord this user")


    train_features_std, train_targets_std, test_features_std, train_features, train_targets, test_features, test_targets = creat_samples(total_features, total_targets)  # 从总数据集中采样

    sample_std = {'train_features_std': train_features_std, 'train_targets_std': train_targets_std, 'test_features_std': test_features_std}
    sample = {'train_features': train_features, 'train_targets': train_targets, 'test_features': test_features, 'test_targets': test_targets}
    with open("datasets_std.pkl", 'wb') as f:
        # 原本是'E:\大学\景文鹏的实验室实习\代码复现\\rnn\\total_datasets.pkl'
        pickle.dump(sample_std, f)
    '''with open("datasets_std.pkl", 'wb') as f:
        pickle.dump(sample, f)'''
    # 将特征和目标转换为NumPy数组
    return train_features_std, train_targets_std
# 确定序列长度的函数
def get_data_length(elevation):
    if 20 <= elevation < 40:
        return 12
    elif 40 <= elevation < 50:
        return 10
    elif 50 <= elevation < 60:
        return 7
    elif 60 <= elevation < 70:
        return 6
    elif 70 <= elevation < 80:
        return 5
    elif 80 <= elevation <= 90:
        return 4
    else:
        return 0   
    
def standardize_features(train_samples, test_samples):
    # 分别计算训练集中CSI值和仰角的平均值和标准差
    csi_train = np.concatenate([s[0, :] for s in train_samples], axis=0)
    angle_train = np.concatenate([s[1, :] for s in train_samples], axis=0)

    csi_mean, csi_std = csi_train.mean(), csi_train.std()
    angle_mean, angle_std = angle_train.mean(), angle_train.std()

    # 定义标准化函数
    def standardize_sample(sample, csi_mean, csi_std, angle_mean, angle_std):
        standardized_sample = np.empty_like(sample)
        standardized_sample[0, :] = (sample[0, :] - csi_mean) / csi_std
        standardized_sample[1, :] = (sample[1, :] - angle_mean) / angle_std
        return standardized_sample

    # 标准化训练集和测试集
    train_samples_std = [standardize_sample(s, csi_mean, csi_std, angle_mean, angle_std) for s in train_samples]
    test_samples_std = [standardize_sample(s, csi_mean, csi_std, angle_mean, angle_std) for s in test_samples]

    return train_samples_std, test_samples_std

def standardize_targets(train_targets):
    train_labels = np.array(train_targets).reshape(-1, 1)

    # 初始化标签的标准化器
    target_scaler = StandardScaler()

    # 用训练标签拟合标准化器，并将其转换
    train_labels_scaled = target_scaler.fit_transform(train_labels)
    joblib.dump(target_scaler, 'label_scaler.pkl')

    return train_labels_scaled

def creat_samples(features, targets):
    # 数据点的总数量
    total_samples = len(features)
    # 计算训练集和测试集的大小
    train_size = 400000  # 训练集400000
    test_size = 40000  # 测试集40000
    # 生成索引列表并随机打乱
    indices = np.arange(total_samples)
    np.random.shuffle(indices)

    # 分割数据索引
    train_indices = indices[:train_size]
    test_indices = indices[train_size:train_size + test_size]

    train_features = [features[i] for i in train_indices]
    train_targets = [targets[i] for i in train_indices]

    test_features = [features[i] for i in test_indices]
    test_targets = [targets[i] for i in test_indices]

    train_features_std, test_features_std = standardize_features(train_features, test_features)
    train_targets_std = standardize_targets(train_targets)
    # 对于测试集的标签不进行标准化
    return train_features_std, train_targets_std, test_features_std, train_features, train_targets, test_features, test_targets



#用0填充序列长度函数
def collate_fn(batch):
    # batch中的每个元素形式为(data, label)
    data, labels = zip(*batch)
    
    # 找到最长的序列
    max_length = max([s.shape[1] for s in data])  # 假设s的形状为[2, seq_len]

    data = [s.clone().detach() for s in data]
    # 填充序列
    padded_data = [F.pad(s, (0, max_length - s.shape[1])) for s in data]
    
    # 将填充后的数据转换为张量
    padded_data = torch.stack(padded_data)
    
    return padded_data, torch.tensor(labels)

