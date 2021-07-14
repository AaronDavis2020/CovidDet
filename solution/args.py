""" 
@ author: Qmh
@ file_name: args.py
@ time: 2019:11:20:11:14
""" 
import argparse
import sys # sys.path[0]获得当前路径
import os


parser = argparse.ArgumentParser()

parser.add_argument('--mode',type=str,required=False, default='test')# for train: default='train'; for test:default='test'

# datasets

parser.add_argument('-dataset_path',type=str,default=r'../test_case/used_test_dataset',
                    help='the path to save imgs') # the path of the train/test images
#parser.add_argument('-dataset_txt_path',type=str,default='./dataset-2/segment/small_dataset_segment.txt') # for lung segment pngs
parser.add_argument('-test_txt_path',type=str,default=r'../test_case/SPGC_test.txt')

# optimizer
parser.add_argument('--optimizer',default='sgd',choices=['sgd','rmsprop','adam','radam'])
parser.add_argument("--lr",type=float,default=0.001)
parser.add_argument('--lr-fc-times', '--lft', default=5, type=int,
                    metavar='LR', help='initial model last layer rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--no_nesterov', dest='nesterov',
                         action='store_false',
                         help='do not use Nesterov momentum')
parser.add_argument('--alpha', default=0.99, type=float, metavar='M',
                         help='alpha for ')
parser.add_argument('--beta1', default=0.9, type=float, metavar='M',
                         help='beta1 for Adam (default: 0.9)')
parser.add_argument('--beta2', default=0.999, type=float, metavar='M',
                         help='beta2 for Adam (default: 0.999)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

#training
parser.add_argument("--checkpoint",type=str,default='./checkpoints/segment')
parser.add_argument("--resume",default='',type=str,
                    metavar='PATH',help='path to save the latest checkpoint') # segment

parser.add_argument("--batch_size",type=int,default=12)
parser.add_argument("--start_epoch",default=0,type=int,metavar='N')
parser.add_argument('--epochs',default=30,type=int,metavar='N')

parser.add_argument('--image-size',type=int,default=512)
parser.add_argument('--arch',default='resnet50',choices=['resnet34','resnet18','resnet50','resnet101'])
parser.add_argument('--num_classes',default=3,type=int)

# model path for testing
parser.add_argument('--model_path',default='resnet50.pth', type=str)
parser.add_argument('--result_csv',default='../result.csv') #to save the y_pre of slices in csv

args = parser.parse_args()
