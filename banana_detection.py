import os
import pandas as pd
import torch
import torchvision
from d2l import torch as d2l

d2l.DATA_HUB['banana-detection'] = (
    d2l.DATA_URL + 'banana-detection.zip', 
    '5de26c8fce5ccdea9f91267273464dc968d20d72'
)


'''读取香蕉数据集'''
def read_data_bananas(is_train=True):
    data_dir = d2l.download_extract('banana-detection')
    print(data_dir)
    csv_fpath = os.path.join(data_dir, 'bananas_train' if is_train else 'bananas_val', 'label.csv')
    csv_file = pd.read_csv(csv_fpath)
    csv_file = csv_file.set_index('img_name')
    
    images,targets=[], []
    for image_name,target in csv_file.iterrows():
        images.append(torchvision.io.read_image(path=os.path.join(data_dir, 'bananas_train'if is_train else 'bananas_val', 'images', f'{image_name}')))
        targets.append(list(target))

    return images, torch.tensor(targets).unsqueeze(1) / 256


"""一个用于加载香蕉检测数据集的自定义数据集"""
class BananasDataset(torch.utils.data.Dataset):
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)


"""加载香蕉检测数据集"""
def load_data_bananas(batch_size):
    train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True), batch_size, shuffle=True)

    val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False), batch_size, shuffle=False)

    return train_iter, val_iter


'''读取小批量，打印'''
batch_size,edge_size = 32,256
train_iter,valid_iter = load_data_bananas(batch_size)
batch = next(iter(train_iter))
print(batch[0].shape, batch[1].shape)



#permute()函数可以同时多次交换tensor的维度, 如：b = a.permute(0, 2 ,1) 将a的维度索引1和维度索引2交换位置
imgs = (batch[0][0:10].permute(0, 2, 3, 1)) / 255 #除以255是为了对图片中每一个像素进行标准化

axes = d2l.show_images(imgs, 2, 5, scale=2)

for axe,label in zip(axes, batch[1][0:10]):
    d2l.show_bboxes(axe, bboxes=[label[0][1:5] * edge_size], colors=['b']) #label[0][1:5]*256乘256是因为加载数据集时bounding box边缘框除以256,256是图片的高和宽
d2l.plt.show()