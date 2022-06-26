'''kaggle 房价预测实战'''

from pickletools import optimize
from pyexpat import features
from unicodedata import numeric
from xml.sax.handler import all_features
import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l


'''数据路径'''
train_data = pd.read_csv("../kaggle/train.csv")
test_data = pd.read_csv("../kaggle/test.csv")


'''查看数据信息'''
print(train_data.shape, test_data.shape)    #(47439, 41), (31626, 40)
print([i for i in train_data.columns if i not in test_data.columns])    #['sold price]
print(train_data_columns)   #查看列个数


'''剔除无用的文本信息'''
all_features = pd.concat(
    (train_data.iloc[:, 4:-1], test_data.iloc[:, 3:-1])
)


'''处理缺失的数据 '''
missing_values_count = all_features.isnull().sum()
missing_values_count.loc[missing_values_count > 0]

total = np.product(all_features.shape)
total_missing = missing_values_count.sum()

percent_missing = (total_missing / total) * 100


'''对Na值补零'''
all_features = all_features.fillna(0)


'''标准化数据'''
all_features.dtypes.unique()
numeric_features = all_features.dtypes[all_features != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std())
)
all_features[numeric_features] = all_features[numeric_features].fillna(0)


'''one-hot 编码'''
all_features.dtypes[all_features.dtypes == 'object']    #查看类型为object的特征

features = list(numeric_features)
features.append('Type')     #加上类别较少的Type
all_features = all_features[features]
print(all_features.shape)
all_features = pd.get_dummies(all_features, dummy_na=True)
print(all_features.shape)


'''转为张量'''
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[: n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(train_data['sold_price'].values.reshape(-1, 1), dtype=torch.float32)


'''网络'''
loss = nn.MSELoss()
in_features = train_features.shape[1]

def get_net():
    net = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),     
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )
    return net

def log_rmse(net, features, labels):
    '''对于价格预测，用相对误差(y-y'/y)来衡量, 再用log缩小'''
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))

    return rmse.item()


'''训练'''
def train(net, train_features, train_labels, test_feaures, test_labels, num_epochs, lr, wd, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)

    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad
            l = loss(net(X), y)
            l.backward()
            optimizer.step()

        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))

    return train_ls, test_ls

def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid

def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
        print(f'折{i + 1}, 训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k


'''
超参数设置
1.对于浅层(单层)MLP, Lr需很多(如20)
2.对于多层MLP, Lr则很小(如0.1)
3.batch_size调大可使曲线减少波动
'''
k, num_epochs, lr, w_d, batch_size = 5, 100, 0.1, 0.1, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, w_d, batch_size)

print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
      f'平均验证log rmse: {float(valid_l):f}')


'''预测'''
def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    print(f'训练log rmse：{float(train_ls[-1]):f}')
    # 将网络应用于测试集。
    preds = net(test_features).detach().numpy()
    # 将其重新格式化以导出到Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)

train_and_pred(train_features, test_features, train_labels, test_data,
               num_epochs, lr, w_d, batch_size)