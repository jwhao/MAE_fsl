import torch
from torch import nn
from data.datamgr import SimpleDataManager , SetDataManager
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
# torch.backends.cudnn.benchmark = True
from models.models_mae import mae_vit_base_patch16, mae_vit_large_patch16
#--------------参数设置--------------------
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image_size', default=224, type=int, choices=[84, 224], help='input image size, 84 for miniImagenet and tieredImagenet, 224 for cub')
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
log_path = os.path.join(log_path,'save/{}_{}_{}_mae_bank'.format(params.dataset,params.train_n_episode,params.n_shot))
ensure_path(log_path)
set_log_path(log_path)
log('log and pth save path:  %s'%(log_path))
log(params)

# -------------设置GPU--------------------
set_gpu(params.gpu)
# -------------导入数据--------------------

json_file_read = False
if params.dataset == 'mini_imagenet':
        base_file = 'train'
        val_file = 'val'
        params.num_classes = 64
elif params.dataset == 'cub':
    base_file = 'base.json'
    val_file = 'val.json'
    json_file_read = True
    params.num_classes = 200
elif params.dataset == 'tiered_imagenet':
    base_file = 'train'
    val_file = 'val'
    params.num_classes = 351
else:
    ValueError('dataset error')

# -----------  base data ----------------------
base_datamgr = SimpleDataManager(params.data_path, params.image_size, batch_size=params.batch_size, json_read=json_file_read)
base_loader = base_datamgr.get_data_loader(base_file, aug=True)

#-----------  train data ----------------------
train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot)
train_datamgr = SetDataManager(params.data_path, params.image_size, n_query=params.n_query, n_episode=params.train_n_episode, json_read=json_file_read, **train_few_shot_params)
train_loader = train_datamgr.get_data_loader(base_file, aug=True)

#------------ val data ------------------------
test_few_shot_params = dict(n_way=params.val_n_way, n_support=params.n_shot)
val_datamgr = SetDataManager(params.data_path, params.image_size, n_query=params.n_query, n_episode=params.val_n_episode, json_read=json_file_read, **test_few_shot_params)
val_loader = val_datamgr.get_data_loader(val_file, aug=False)

#   ------查看导入的数据----------
# target, label = next(iter(base_loader))
# print(len(base_loader))
# print(target.size())
# print(label.size())
# print('--------------------')
# target1, label1 = next(iter(train_loader))
# print(len(train_loader))
# print(target1.size())
# print(label1.size())
# print('--------------------')
# target2, label2 = next(iter(val_loader))
# print(len(val_loader))
# print(target2.size())
# print(label2.size())

# ----------- 导入模型 -------------------------
model = mae_vit_base_patch16()
# model = mae_vit_large_patch16()
# state_dict = torch.load('/root/autodl-tmp/jiangweihao/A_FSL_Lab_JWH/mae_pretrain_vit_base.pth')
state_dict = torch.load('/home/jiangweihao/code/MAE_fsl/mae_pretrain_vit_base.pth')
state_dict = state_dict['model']
model.load_state_dict(state_dict,strict=False)  # 
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
            support_f_m = support_f_m[:,1:,:].mean(dim=1,keepdim=True)    # 所有patch特征取平均
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
for idy, (temp2,target) in enumerate(train_loader):   
    # temp2, _ =next(iter(train_loader))

    support,query = temp2.split([params.n_shot,params.n_query],dim=1)
    cache_values, q_values = target.split([params.n_shot,params.n_query],dim=1)

    cache_values = F.one_hot(cache_values).half()
    cache_values = cache_values.reshape(-1,cache_values.shape[-1])
    q_values = q_values.reshape(-1)
    cache_values, q_values = cache_values.cuda(), q_values.cuda()

    n,k,c,h,w = support.shape
    support = support.reshape(-1,c,h,w)
    support = support.cuda()
    query = query.reshape(-1,c,h,w)
    query = query.cuda()

    # -----------feature extractor------------------
    mask_ratio=[0,0.25,0.5,0.75]       # 0,0.25,0.5,0.75
    support_f , support_cls_token, query_f, query_cls_token = cache_model(support,query,model,mask_ratio=mask_ratio)

# ===============================================================================
  # 采用Tip-CLIP里的度量方式
# ===============================================================================
    beta = 1.0
    affinity = query_f @ support_f.t()
    affinity2 = query_cls_token @ support_cls_token.t()          # 会让效果变差
    # affinity += affinity2
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values.float()
    cache_logits2 = ((-1) * (beta - beta * affinity2)).exp() @ cache_values.float()
    # cache_logits += cache_logits2        
    acc = cls_acc(cache_logits, q_values)            # %百分制
    acc /= 100
    val_acc.append(acc)
# ===========================直接采用相似性度量=============================
    y = np.repeat(range(params.val_n_way),params.n_query)
    y = torch.from_numpy(y)
    y = y.cuda()
    # metric_cos = affinity.reshape(-1,n,k).mean(dim=-1)           # 直接度量
    support_f = support_f.reshape(n,k,-1).mean(dim=1)
    support_cls_token = support_cls_token.reshape(n,k,-1).mean(dim=1)
    
    metric_cos = query_f @ support_f.t()
    metric_cos2 = query_cls_token @ support_cls_token.t()
    # metric_cos += metric_cos2


    pred = metric_cos.data.max(1)[1]
    cos_acc = pred.eq(y).sum()/(params.train_n_way*params.n_query)
    total_correct += pred.eq(y).sum()
    
val_acc_ci95 = 1.96 * np.std(np.array(val_acc)) / np.sqrt(params.val_n_episode)
val_acc = np.mean(val_acc) * 100

cos_acc = total_correct/len(train_loader)/(params.train_n_way*params.n_query) * 100

log('test size:%d , test_acc:%.2f ± %.2f %% '%(len(train_loader), val_acc, val_acc_ci95))
log('cos acc: %.2f %% '%(cos_acc))
log('test epoch time: {:.2f}'.format(timer.t()))

log(time.time()-start)
log('===========================training end!===================================')



