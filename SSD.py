'''多尺度的目标检测'''
'''在特征图上做anchor boxes的生成'''
import torch
from d2l import torch as d2l

img = d2l.plt.imread("catdog.jpg")
h, w = img.shape[:2]
print(h, w)

# 显示图片
# d2l.plt.imshow(img)
# d2l.plt.show()


# 在features map上生成若干个anchor boxes, 每个像素作为anchor box的中心, 最后再映射回原图上(即乘回原图尺寸大小)
def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()

    # 这里生成一个样本数为1 通道数为10 高宽为fmap_h, fmap_w的fmap
    fmap = torch.zeros((1, 10, fmap_h, fmap_w))

    # 这里anchors出来的是一堆(0, 1)之间的值, 表示占整张图的占比大小, 所以后续需要乘回原图大小才是每个生成锚框在真实图片中的大小
    anchors = d2l.multibox_prior(
        fmap,
        sizes=s,        # sizes表示anchor占长宽的比
        ratios=[1, 2, 0.5]      # ratios指高宽比
    )

    # 原图的大小尺寸
    bbox_scale = torch.tensor((w, h, w, h))

    d2l.show_bboxes(
        d2l.plt.imshow(img).axes, 
        anchors[0] * bbox_scale
    )


# 这里假设特征图的高宽都为4, 并不是真实的特征图大小
display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
d2l.plt.show()


# 将特征图的高宽大小减半, 然后使用较大的锚框来检测较大的目标(底层的特征图检测小物体, 高层的特征图检测大物体)
display_anchors(fmap_w=2, fmap_h=2, s=[0.4])
d2l.plt.show()


'''SSD(单发多框检测)'''
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


# 预测锚框的类别
# 设目标类别的数量为q。这样一来，锚框有q+1个类别，其中0类是背景。 
# 在某个尺度下，设特征图的高和宽分别为h和w。 如果以其中每个单元为中心生成个a锚框，那么我们需要对hwa个锚框进行分类。
# 这里如果最后用全连接层作为输出会导致参数量过大, 这里参考NiN网络中使用卷积层的通道来输出类别预测的方法
def cls_predictor(num_inputs, num_anchors, num_classes):    # 输入的通道数、锚框数、类别
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1), kernel_size=3, padding=1)


# 边界框预测层, 这里需要为每个锚框预测4个偏移量(所以要乘以4)
def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)


# 连接多层间的预测
def forward(x, block):
    return block(x)

Y1 = forward(torch.zeros((2, 8, 20, 20)), cls_predictor(8, 5, 10))
Y2 = forward(torch.zeros((2, 16, 10, 10)), cls_predictor(16, 3, 10))
print(Y1.shape, Y2.shape)


def flatten_pred(pred):
    # 把生成的4d向量拉成一个2d的向量, 并把通道数移到最后一个维度
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)

def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)

print(concat_preds([Y1, Y2]).shape)


'''高宽减半块'''
# 即是定义一个CNN的网络
def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels,
                             kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)


'''基本网络块'''
def base_net():
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)


'''完整的单发多框检测模型由五个模块组成'''
# 这里是手动构造5个stage组成的网络, 也可以直接用如resnet网络的5个stage
def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1,1))
    else:
        blk = down_sample_blk(128, 128)
    return blk


# 每个块定义前向传播 (核心函数)
# 与之前卷积神经网络不同, 卷积神经网络是输入一个X后输出一个Y, 而目标检测我们还需要对锚框进行处理
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)   # 生成锚框
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)


'''超参数设置'''
sizes = [[0.2, 0.272], 
        [0.37, 0.447], 
        [0.54, 0.619], 
        [0.71, 0.79],
        [0.88, 0.961]]

ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1


'''定义完整的模型TinySSD'''
class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            # 即赋值语句self.blk_i=get_blk(i)
            # setattr(对象, 对象属性(字符串), 属性值), 设置一个对象属性
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i], num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i], num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # getattr(self,'blk_%d'%i)即访问self.blk_i, 返回一个对象属性
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds


# 创建一个实例模型, 执行前向运算查看效果
net = TinySSD(num_classes=1)
X = torch.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)

print('output anchors:', anchors.shape)
print('output class preds:\n', cls_preds.shape)
print('output bbox preds:\n', bbox_preds.shape)



'''读取香蕉检测数据集'''
# 初始化参数和定义优化算法
batch_size = 32
train_iter, _ = d2l.load_data_bananas(batch_size)

device, net = d2l.try_gpu(), TinySSD(num_classes=1)
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)


# 定义损失函数和评价函数
cls_loss = nn.CrossEntropyLoss(reduction='none')
bbox_loss = nn.L1Loss(reduction='none')

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    # bbox_mask 为0表示该锚框是背景框, 为1表示该锚框为非背景框
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1, num_classes),
                   cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    bbox = bbox_loss(bbox_preds * bbox_masks,
                     bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox


# 用准确率评价分类结果， 由于偏移量使用了L1范数损失，我们使用平均绝对误差来评价边界框的预测结果
def cls_eval(cls_preds, cls_labels):
    # 由于类别预测结果放在最后一维，argmax需要指定最后一维。
    return float((cls_preds.argmax(dim=-1).type(cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())


'''模型训练'''
num_epochs, timer = 20, d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['class error', 'bbox mae'])
net = net.to(device)
for epoch in range(num_epochs):
    # 训练精确度的和，训练精确度的和中的示例数
    # 绝对误差的和，绝对误差的和中的示例数
    metric = d2l.Accumulator(4)
    net.train()
    for features, target in train_iter:
        timer.start()
        trainer.zero_grad()
        X, Y = features.to(device), target.to(device)
        # 生成多尺度的锚框，为每个锚框预测类别和偏移量
        anchors, cls_preds, bbox_preds = net(X)
        # 为每个锚框标注类别和偏移量
        bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y)
        # 根据类别和偏移量的预测和标注值计算损失函数
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
        l.mean().backward()
        trainer.step()
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.numel())
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    animator.add(epoch + 1, (cls_err, bbox_mae))

print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on 'f'{str(device)}')
d2l.plt.show()


'''预测目标'''
X = torchvision.io.read_image('../img/banana.jpg').unsqueeze(0).float()
img = X.squeeze(0).permute(1, 2, 0).long()

# 使用下面的multibox_detection函数，我们可以根据锚框及其预测偏移量得到预测边界框。然后，通过NMS来移除相似的预测边界框
def predict(X):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)

    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)

    # multibox_detection函数输出为经过处理之后的锚框，形状为（批量大小, 锚框的数量, 6）
    '''
    [[[ 0.00,  0.90,  0.10,  0.08,  0.52,  0.92],
         [ 1.00,  0.90,  0.55,  0.20,  0.90,  0.88],
         [-1.00,  0.80,  0.08,  0.20,  0.56,  0.95],
         [-1.00,  0.70,  0.15,  0.30,  0.62,  0.91]]]
    如这个输出表示：(1, 4, 6) 批量大小为1, 锚框数为4, 6个元素信息(预测的类别标签、置信度、边界框四个坐标)
    '''
    # 最后一维由6个元素组成分别为: 预测的类别标签、置信度、边界框四个坐标（归一化处理）
    # output[0]是第一个批量的全部锚框，把不是背景类锚框的存入idx
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]    

    # 返回所有非背景类的锚框
    return output[0, idx]

output = predict(X)


# 筛选所有置信度不低于0.9的边界框，做为最终输出
def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

display(img, output.cpu(), threshold=0.9)