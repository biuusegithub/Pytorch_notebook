import torch
from torch import nn
from d2l import torch as d2l


net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Dropout(0.7),
    nn.Linear(256, 10)
)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

trainer = torch.optim.SGD(net.parameters(), lr=lr)

lr, num_epochs, batch_siez = 0.01, 10, 256

loss = nn.CrossEntropyLoss()

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_siez)

d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
