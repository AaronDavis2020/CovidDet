""" 
@ author: Qmh
@ file_name: data_gen.py
@ time: 2019:11:15:20:39
"""
from torch.utils.data import Dataset
import os
import numpy as np
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from PIL import Image, ImageFile
from solution.args import args

ImageFile.LOAD_TRUNCATED_IMAGES = True


class TestDataset(Dataset):
    def __init__(self, dcm_list, dataset, transform=None):
        '''
        root:保存测试数据集路径的txt
        transforms: 进行数据增强
        '''
        super(TestDataset, self).__init__()

        self.length = len(dcm_list)
        self.transform = transform
        self.env = dcm_list
        self.dataset = dataset # 需要三通道图像

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img_name = self.env[index-1].split(os.sep)[-1].split('.')[0] + '.png'
        try:
            # img = Image.open(img_path.replace('/', os.sep))
            # img = np.expand_dims(img, axis=2)
            img = self.dataset[index]
            img = np.squeeze(img)[:, :, :3]
            # from torchvision import transforms
            # img = transforms.ToPILImage()(img)  # ndarray -> PILImage
        except:
            return self[index + 1]

        if self.transform is not None:
            img = self.transform(img)

        return img, img_name


class resizeNormalize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.toTensor()

    def __call__(self, img):
        ratio = self.size[0] / self.size[1]
        w, h = img.size
        # 图像的等比填充缩放裁减
        if w / h < ratio:
            t = int(h * ratio)  # 缩放后高度
            w_padding = (t - w) // 2
            img = img.crop((-w_padding, 0, w + w_padding, h))  # 宽度=高度
        else:
            t = int(w * ratio)  # 缩放后宽度
            h_padding = (t - h) // 2
            img = img.crop((0, -h_padding, w, h + h_padding))  # 宽度=高度
        # 双线性插值
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)  ### ???
        return img


def Split_datatset(dataset_txt_path, train_txt_path, test_txt_path):
    '''
    划分数据集:训练和测试
    data_path：数据集的保存路径
    '''
    img_paths, labels = [], []
    dict_skin = {'Normal': 0, 'CAP': 1, 'COVID': 2}
    with open(dataset_txt_path, 'r') as f:
        lines = f.read().split('\n')
        for line in lines:
            if line:
                img_paths.append(line.split(',')[0])
                if line.split(',')[1] in dict_skin.keys():
                    labels.append(dict_skin[line.split(',')[1]])
                else:
                    labels.append(line.split(',')[1])

    train_x, test_x, train_y, test_y = train_test_split(img_paths, labels, stratify=labels, test_size=0.5,
                                                        random_state=42)

    # print(f"train samples:{len(train_x)}, test samples:{len(test_x)}")

    train_set = (train_x, train_y)
    test_set = (test_x, test_y)

    write_dataset_to_txt(train_set, train_txt_path)

    write_dataset_to_txt(test_set, test_txt_path)


def write_dataset_to_txt(data_set, txt_path):
    '''
    将数据集的路径写入txt文件保存
    data_set: 保存图片路径和标签的元组
    txt_path： 待保存的txt文件路径
    '''
    img_paths, labels = data_set

    with open(txt_path, 'w') as f:
        for index, img_path in enumerate(img_paths):
            f.write(img_path + "," + str(labels[index]))
            if index != len(img_paths) - 1:
                f.write('\n')
    print(f"write to {txt_path} successed")


if __name__ == "__main__":
    #     # 只调用一次
    # Split_datatset(args.dataset_txt_path,args.train_txt_path,args.test_txt_path)
    Split_datatset(args.val_txt_path, args.val_txt_path, args.test_txt_path)
    # data = Dataset(args.train_txt_path)
    # print(data.__getitem__(2))
#     data = TestDataset(args.test_txt_path)
#     print(data.__getitem__(10))
