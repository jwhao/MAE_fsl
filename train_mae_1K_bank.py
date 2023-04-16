import torch
from torch import nn
from data.datamgr import SimpleDataManager , SetDataManager
from data.imagenet import ImageNet
from models.predesigned_modules import resnet12
import sys
import os
from utils import *
# os.environ['CUDA_VISIBLE_DEVICES'] = "3"
# from svrg import SVRG
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')
# fix seed
np.random.seed(1)
torch.manual_seed(1)
import tqdm
from torch.nn.parallel import DataParallel
import torchvision.transforms as transforms

# torch.backends.cudnn.benchmark = True
from models.models_mae import mae_vit_base_patch16
#--------------参数设置--------------------
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image_size', default=84, type=int, choices=[84, 224], help='input image size, 84 for miniImagenet and tieredImagenet, 224 for cub')
parser.add_argument('--dataset', default='mini_imagenet', choices=['mini_imagenet','tiered_imagenet','cub'])
parser.add_argument('--data_path', default='/home/jiangweihao/data/mini-imagenet/',type=str, help='dataset path')

parser.add_argument('--train_n_episode', default=300, type=int, help='number of episodes in meta train')
parser.add_argument('--val_n_episode', default=300, type=int, help='number of episodes in meta val')
parser.add_argument('--train_n_way', default=5, type=int, help='number of classes used for meta train')
parser.add_argument('--val_n_way', default=5, type=int, help='number of classes used for meta val')
parser.add_argument('--n_shot', default=5, type=int, help='number of labeled data in each class, same as n_support')
parser.add_argument('--n_query', default=15, type=int, help='number of unlabeled data in each class')
parser.add_argument('--num_classes', default=64, type=int, help='total number of classes in pretrain')

parser.add_argument('--batch_size', default=128, type=int, help='total number of batch_size in pretrain')
parser.add_argument('--freq', default=10, type=int, help='total number of inner frequency')

parser.add_argument('--momentum', default=0.9, type=int, help='parameter of optimization')
parser.add_argument('--weight_decay', default=5.e-4, type=int, help='parameter of optimization')

parser.add_argument('--gpu', default='1')
parser.add_argument('--epochs', default=100)

params = parser.parse_args()

# 设置日志记录路径
log_path = os.path.dirname(os.path.abspath(__file__))
log_path = os.path.join(log_path,'save/{}_{}_{}_mae_1K_bank'.format(params.dataset,params.train_n_episode,params.n_shot))
ensure_path(log_path)
set_log_path(log_path)
log('log and pth save path:  %s'%(log_path))
log(params)

# -------------设置GPU--------------------
set_gpu(params.gpu)
# -------------导入数据--------------------
preprocess = transforms.Compose([
                                transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
                                            ])
root_path = '/home/dataset'
imagenet = ImageNet(root_path, params.n_shot, preprocess)

test_loader = torch.utils.data.DataLoader(imagenet.test, batch_size=64, num_workers=8, shuffle=False)

train_loader_cache = torch.utils.data.DataLoader(imagenet.train, batch_size=128, num_workers=8, shuffle=False) # batch_size=256
train_loader_F = torch.utils.data.DataLoader(imagenet.train, batch_size=256, num_workers=8, shuffle=True)

#   ------查看导入的数据----------
# target, label = next(iter(base_loader))
# print(len(base_loader))
# print(target.size())
# print(label.size())

# ----------- 导入模型 -------------------------
model = mae_vit_base_patch16()
model.load_state_dict(torch.load('/home/jiangweihao/code/MAE_fsl/mae_pretrain_vit_base.pth')['model'],strict=False)
model.cuda()

# from torchinfo import summary
# summary(model,[5,3,224,224])

# del model.fc                         # 删除最后的全连接层
model.eval()
def cache_model(support,query,model,mask_ratio=[0, 0.25, 0.5, 0.75]):
    
    with torch.no_grad():
        # Data augmentation for the cache model
        for i, mask in enumerate(mask_ratio):
            
            support_f_m, _, _ = model.forward_encoder(support,mask_ratio=mask)
            query_f_m, _, _ = model.forward_encoder(query,mask_ratio=mask)
            support_cls_token_m = support_f_m[:,0,:]                # 把cls_token分离出来
            query_cls_token_m = query_f_m[:,0,:]
            support_f_m = support_f_m[:,1:,:].mean(dim=1,keepdim=True)
            query_f_m = query_f_m[:,1:,:].mean(dim=1,keepdim=True)
            if i==0:
                support_f = support_f_m 
                query_f = query_f_m 
                support_cls_token = support_cls_token_m
                query_cls_token = query_cls_token_m
            else:
                support_f = torch.cat((support_f,support_f_m),1)
                query_f = torch.cat((query_f,query_f_m),1) 
                support_cls_token = torch.cat((support_cls_token,support_cls_token_m),1)
                query_cls_token = torch.cat((query_cls_token,query_cls_token_m),1) 

    
    support_f = support_f.mean(dim=1).squeeze(1)   
    query_f = query_f.mean(dim=1).squeeze(1) 

    # support_cls_token = support_cls_token.mean(dim=1)  
    # query_cls_token = query_cls_token.mean(dim=1) 

    # 归一化
    # support_f_m = support_f.mean(dim=-1, keepdim=True)
    # support_f = support_f - support_f_m
    support_f /= support_f.norm(dim=-1, keepdim=True)
    support_cls_token /= support_cls_token.norm(dim=-1, keepdim=True)
    # query_f_m = query_f.mean(dim=-1, keepdim=True)
    # query_f = query_f - query_f_m
    query_f /= query_f.norm(dim=-1, keepdim=True)
    query_cls_token /= query_cls_token.norm(dim=-1, keepdim=True)

    return support_f, support_cls_token, query_f, query_cls_token

def catch_feature(query, model, mask_ratio=0):

    with torch.no_grad():    

        feature, _, _ = model.forward_encoder(query,mask_ratio=mask_ratio)

    return feature[:,0,:],feature[:,1:,:]

# ---------------------------------------------
loss_fn = torch.nn.CrossEntropyLoss()

epochs = 100

start = time.time()


log('==========start testing on train set===============')

# for epoch in range(epochs):   
    
out_avg_loss = []
timer = Timer()
                
avg_loss = 0
total_correct = 0
val_acc = []
support_f, support_cls_token, cache_values = [], [], []
for idy, (temp2,target) in enumerate(train_loader_cache):   

    cache_values_b = F.one_hot(target).half()
    temp2 = temp2.cuda()
    # -----------feature extractor------------------
    mask_ratio = 0       # 0,0.25,0.5,0.75
    support_f_b , support_cls_token_b = catch_feature(temp2,model,mask_ratio=mask_ratio)
    support_f.append(support_f_b)
    support_cls_token.append(support_cls_token_b)
    cache_values.append(cache_values)


query_f, query_cls_token, q_values = [], [], []
for idy, (temp2,target) in enumerate(test_loader):  

    temp2 = temp2.cuda()
    # -----------feature extractor------------------
    mask_ratio = 0       # 0,0.25,0.5,0.75
    query_f_b , query_cls_token_b = catch_feature(temp2,model,mask_ratio=mask_ratio)
    query_f.append(query_f_b)
    query_cls_token.append(query_cls_token_b)
    q_values.append(target)

# ===============================================================================
  # 采用Tip-CLIP里的度量方式
# ===============================================================================
beta = 1.0
affinity = query_f @ support_f.t()
affinity2 = query_cls_token @ support_cls_token.t()
# affinity += affinity2
cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values.float()
cache_logits2 = ((-1) * (beta - beta * affinity2)).exp() @ cache_values.float()
cache_logits += cache_logits2        
acc = cls_acc(cache_logits, q_values)            # %百分制
acc /= 100
val_acc.append(acc)

    
val_acc_ci95 = 1.96 * np.std(np.array(val_acc)) / np.sqrt(params.val_n_episode)
val_acc = np.mean(val_acc) * 100


log('test size:%d , test_acc:%.2f ± %.2f %% '%(len(test_loader), val_acc, val_acc_ci95))
log('test epoch time: {:.2f}'.format(timer.t()))

log(time.time()-start)
log('===========================test end!===================================')



