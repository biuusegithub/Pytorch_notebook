import torch
from torch import nn
from d2l import torch as d2l
import torchvision
from torch.utils import data
from torchvision import transforms


''' 1.数据集下载'''
trans = transforms.ToTensor()


mnist_train = torchvision.datasets.FashionMNIST(
    root='../data', train=True, transform=trans, download=True
)


mnist_test = torchvision.datasets.FashionMNIST(
    root='../data', train=False, transform=trans, download=True
)


print(len(mnist_train), len(mnist_test))



'''2.可视化数据集'''
def get_fashion_mnist_labels(labels):
    '''返回Fashion_MNIST数据集的文本标签'''
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat', 
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
    ]
    return [text_labels[int(i)] for i in labels]


def show_images(imgs, num_rows, num_cols, title=None, scale=1.5):
    '''绘制图像列表'''
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)

    axes = axes.flatten()
    for i, (ax,img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            '''图片张量'''
            ax.imshow(img.numpy())
        else:
            '''PIL图片'''
            ax.imshow(img)

        '''隐藏x, y轴'''
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if title:
            ax.set_title(title[i])

    return axes


'''3.定义load_data_fashion_mnist函数'''
def load_data_fashion_mnist(batch_size, resize=None):
    '''图片加载至内存'''
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))

    trans = transforms.Compose(trans)

    mnist_train = torchvision.datasets.FashionMNIST(
    root='../data', train=True, transform=trans, download=True
)
    mnist_test = torchvision.datasets.FashionMNIST(
        root='../data', train=False, transform=trans, download=True
    )
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=2), 
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=2)
    )


'''4.几个样本的图像及其相应的标签'''
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, title=get_fashion_mnist_labels(y))
d2l.plt.show()



