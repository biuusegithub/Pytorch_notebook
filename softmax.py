import torch


'''softmax公式: soft(X)ij = exp(xij) / Σk exp(xik)'''

num_inputs, num_outputs = 784, 10
w = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

def softmax(X):
    x_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return x_exp / partition


def net(X):
    '''softmax回归模型'''
    return softmax(torch.matmul(X.reshape((-1, w.shape[0])), w) + b)



