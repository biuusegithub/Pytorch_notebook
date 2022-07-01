import os
import torch
import torchvision
from torch import nn
from d2l import torch as d2l


d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip', 'fba480ffa8aa7e0febbb511d181409f899b9baa5')
data_dir = d2l.download_extract('hotdog')


train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))


'''查看图片'''
hotdogs = [train_imgs[i][0] for i in range(10)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(10)]
d2l.show_images(hotdogs + not_hotdogs, 2, 10, scale=1.4)
# d2l.plt.show()


# 使用RGB通道的均值和标准差，以标准化每个通道,这里由于本地显卡不行，所以我把图片从224调小至96，batch_size降至64
normalize = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(96),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    normalize
])

test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(96),    # 中心裁剪
    torchvision.transforms.ToTensor(),
    normalize
])


pretrained_net = torchvision.models.resnet18(pretrained=True)
print(pretrained_net.fc)


finetune_net = torchvision.models.resnet18(pretrained=True)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
nn.init.xavier_uniform_(finetune_net.fc.weight)


'''如果param_group=True, 输出层中的模型参数将使用十倍的学习率'''
def train_fine_tuning(net, learning_rate, batch_size=64, num_epochs=2, param_group=True):
    train_iter = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_augs),
        batch_size=batch_size, 
        shuffle=True
    )

    test_iter = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=test_augs),
        batch_size=batch_size
        )


    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction="none")

    if param_group:
        params_1x = [param for name, param in net.named_parameters()
             if name not in ["fc.weight", "fc.bias"]]
        '''如果不是最后一层全连接层的参数和偏移, 则拉大lr至原来的10倍'''     
        trainer = torch.optim.SGD([{'params': params_1x},
                                   {'params': net.fc.parameters(),
                                    'lr': learning_rate * 10}],
                                lr=learning_rate, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.001)
    
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)


train_fine_tuning(finetune_net, 5e-5)
d2l.plt.show()


'''比较不用预训练权重的效果'''
scratch_net = torchvision.models.resnet18()
scratch_net.fc = nn.Linear(scratch_net.fc.in_features, 2)
train_fine_tuning(scratch_net, 5e-1, param_group=False)


