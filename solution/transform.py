import random
from torchvision import transforms
from PIL import Image,ImageOps,ImageFilter


class Resize(object):
    def __init__(self,size,interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
    
    def __call__(self,img):
        ratio = self.size[0]/self.size[1]
        w, h = img.size
        if w/h < ratio:
            t = int(h*ratio)
            w_padding = (t-w)//2
            img = img.crop((-w_padding,0,w+w_padding,h))
        else:
            t = int(w*ratio)
            h_padding = (t-h)//2
            img = img.crop((0,-h_padding,w,h+h_padding))
        img = img.resize(self.size,self.interpolation)
        return img

class RandomRotate(object):
    '''
    随机旋转图片
    '''
    def __init__(self,degree,p=0.5):
        self.degree = degree
        self.p = p
    
    def __call__(self,img):
        if random.random() < self.p:
            rotate_degree = random.uniform(-1*self.degree,self.degree)
            img = img.rotate(rotate_degree,Image.BILINEAR)
        return img
    

class RandomGaussianBlur(object):
    def __init__(self,p=0.5):
        self.p = p
    def __call__(self,img):
        if random.random() < self.p:
            # 高斯模糊是高斯低通滤波器 不保留细节   高斯滤波是高斯高通滤波器 保留细节
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        return img

def get_train_transform(mean,std,size):
    train_transform = transforms.Compose([
        #Resize((int(size*(256/224)), int(size*(256/224)))),
        #transforms.RandomCrop(size),
        #transforms.RandomRotation(degrees=15),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,std=std),
    ])
    return train_transform

def get_test_transform(mean,std,size):
    return transforms.Compose([
        #Resize((int(size*(256/224)),int(size*(256/224)))),
        #transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,std=std),
    ])


def get_transforms(input_size=512,test_size=512,backbone=None):
    mean,std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] # 对三通道
    if backbone is not None and backbone in ['pnasnet5large', 'nasnetamobile']:
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    transformations = {}
    transformations['test'] = get_test_transform(mean,std,test_size)
    return transformations