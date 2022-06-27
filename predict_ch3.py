import torch
from torch import nn
from d2l import torch as d2l


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_ch3(net, test_iter,n=100): 
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X.to(device)).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(X[0:n].reshape((n, 224, 224)), 1, n, titles=titles[0:n])