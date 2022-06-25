import torch
from d2l import torch as d2l


def cross_entropy(y_hat, y):
    '''交叉熵损失函数'''
    return -torch.log(y_hat[range(len(y_hat)), y])


def accuracy(y_hat, y):
    
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        '''若是二维矩阵 和 列数大于1'''
        y_hat = y_hat.argmax(axis=1)        #取该行最大值

    cmp = y_hat.type(y.dtype) == y          
    return float(cmp.type(y.dtype).sum())   #统计预测正确样本数


def evaluate_accuracy(net, data_iter):
    '''数据集上的模型精度'''
    if isinstance(net, torch.nn.Module):
        '''若为nn.Module类型则转换为评估模式'''
        net.eval()
    
    metric = Accumulator(2)
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())

    return metric[0] / metric[1]


def train_epoch_ch3(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        '''若为nn.Module类型则转换为训练模式'''
        net.train()

    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)

        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            updater.step()
            metric.add(
                float(l) * len(y), accuracy(y_hat, y),
                y.size().numel()
            )
        
        else:
            l.sum().backward()
            updater(X.shape[0])
            metirc.add(float(l.sum()), accuracy(y_hat, y), y.numel())

    return metric[0] / metric[2], metric[1] / metric[2]
