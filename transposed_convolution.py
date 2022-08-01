import torch
from torch import nn
from d2l import torch as d2l


def trans_conv(X, K):
    #  无padding 和 stride=1时的实现
    h, w = K.shape
    Y = torch.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i: i + h, j: j + w] += X[i, j] * K
    return Y


# 验证上述输出
X = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print(trans_conv(X, K))


# 使用pytorch高级API进行反卷积
X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False)
tconv.weight.data = K
print(tconv(X))


'''填充、步幅、多通道'''
# padding与常规卷积不同, 在转置卷积中, 填充被应用于的输出（常规卷积将填充应用于输入） 
# 例如, 当将高和宽两侧的填充数指定为1时, 转置卷积的输出中将删除第一和最后的行与列, 因为最外层已经全填充为1
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, padding=1, bias=False)
tconv.weight.data = K
print(tconv(X))


# stride
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, bias=False)
tconv.weight.data = K
print(tconv(X))


# 多通道
X = torch.rand(size=(1, 10, 16, 16))
conv = nn.Conv2d(10, 20, kernel_size=5, padding=2, stride=3)
tconv = nn.ConvTranspose2d(20, 10, kernel_size=5, padding=2, stride=3)
print(tconv(conv(X)).shape == X.shape)



'''与矩阵变换的联系'''
X = torch.arange(9.0).reshape(3, 3)
K = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
Y = d2l.corr2d(X, K)


# 卷积 --> 矩阵乘法
# 将卷积核K重写为包含大量0的稀疏权重矩阵W, 权重矩阵的形状是（4, 9）, 其中非0元素来自卷积核K
def kernel2matrix(K):
    # 这里为什么W要设置成4*9的矩阵, 因为输出是2*2所以是4次卷积, 这里对应4次矩阵乘法, 且输入X为3*3的矩阵拉长后为9
    k, W = torch.zeros(5), torch.zeros((4, 9))
    
    # 这里K是一个2*2的矩阵, 把K的第一行放入k的前两个位置, K的第二行放入k的后两个位置
    k[:2], k[3:5] = K[0, :], K[1, :]    # tensor([1., 2., 0., 3., 4.])

    W[0, :5], W[1, 1:6], W[2, 3:8], W[3, 4:] = k, k, k, k

    return W


W = kernel2matrix(K)
print(W)


# 判断卷积是否成功转换为矩阵乘法
print(Y == torch.matmul(W, X.reshape(-1)).reshape(2, 2))


# 使用矩阵乘法来实现转置卷积
Z = trans_conv(Y, K)
Z == torch.matmul(W.T, Y.reshape(-1)).reshape(3, 3)


