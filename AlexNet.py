import torch
from torch import nn
from d2l import torch as d2l


net = nn.Sequential(
    nn.Conv2d(1, 96, kernel_size=3, stride=4, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),

    nn.Conv2d(96, 256, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),

    nn.Conv2d(256, 384, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),

    nn.Flatten(),

    nn.Linear(6400, 4096),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(4096, 10)
)

batch_size, lr, epochs = 256, 0.01, 10
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size, resize=224)

d2l.train_ch6(train_iter, test_iter, epochs, lr, d2l.try_gpu())