import torch
from torch import nn
from data.datamgr import SimpleDataManager , SetDataManager
from models.predesigned_modules import resnet12
import sys
import os
from utils import *
# os.environ['CUDA_VISIBLE_DEVICES'] = "3"

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
from models.models_mae import mae_vit_base_patch16,mae_vit_large_patch16
# from sklearn import svm     #导入算法模块
import timm

# assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
from models import models_vit

import scipy as sp
import scipy.stats
#--------------参数设置--------------------
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image_size', default=224, type=int, choices=[84, 224], help='input image size, 84 for miniImagenet and tieredImagenet, 224 for cub')
parser.add_argument('--dataset', default='mini_imagenet', choices=['mini_imagenet','tiered_imagenet','cub'])
parser.add_argument('--data_path', default='/home/jiangweihao/CodeLab/data/mini-imagenet/',type=str, help='dataset path')

parser.add_argument('--train_n_episode', default=1000, type=int, help='number of episodes in meta train')
parser.add_argument('--val_n_episode', default=300, type=int, help='number of episodes in meta val')
parser.add_argument('--train_n_way', default=5, type=int, help='number of classes used for meta train')
parser.add_argument('--val_n_way', default=5, type=int, help='number of classes used for meta val')
parser.add_argument('--n_shot', default=5, type=int, help='number of labeled data in each class, same as n_support')
parser.add_argument('--n_query', default=5, type=int, help='number of unlabeled data in each class')
parser.add_argument('--num_classes', default=64, type=int, help='total number of classes in pretrain')
parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
parser.add_argument('--global_pool', action='store_true')
parser.add_argument('--batch_size', default=128, type=int, help='total number of batch_size in pretrain')
parser.add_argument('--print_freq', default=10, type=int, help='total number of inner frequency')

parser.add_argument('--momentum', default=0.9, type=float, help='parameter of optimization')
parser.add_argument('--weight_decay', default=5.e-4, type=float, help='parameter of optimization')

parser.add_argument('--gpu', default='0')
parser.add_argument('--epochs', default=10,type=int)

parser.add_argument('--ft', action='store_true')

params = parser.parse_args()

# 设置日志记录路径
log_path = os.path.dirname(os.path.abspath(__file__))
log_path = os.path.join(log_path,'save/{}_train_task-{}_shot-{}_mae[{}]_image_compolement_FT[{}]_mm_coslr'.format(params.dataset,params.train_n_episode,params.n_shot,params.model,params.ft))
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
# model = mae_vit_base_patch16()
# state_dict = torch.load('/home/jiangweihao/CodeLab/PytorchCode/mae_fsl/checkpoint/mae_pretrain_vit_base.pth')
# state_dict = state_dict['model']
# model.load_state_dict(state_dict,strict=False)  # 
# model.cuda()

# # from torchinfo import summary
# # summary(model,[5,3,224,224])

# # del model.fc                         # 删除最后的全连接层
# model.eval()

def cache_model(support,query,model,mask_ratio=[0, 0.25, 0.5, 0.75],modal='mean'):
    
    with torch.no_grad():
        # Data augmentation for the cache model
        for i, mask in enumerate(mask_ratio):
            
            support_f_m, _, _ = model.forward_encoder(support,mask_ratio=mask)
            query_f_m, _, _ = model.forward_encoder(query,mask_ratio=mask)
            support_cls_token_m = support_f_m[:,0,:]                # 把cls_token分离出来
            query_cls_token_m = query_f_m[:,0,:]
            if modal == 'mean':
                support_f_m = support_f_m[:,1:,:].mean(dim=1,keepdim=True)
                query_f_m = query_f_m[:,1:,:].mean(dim=1,keepdim=True)
            else:
                support_f_m = support_f_m[:,1:,:]
                query_f_m = query_f_m[:,1:,:]
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

    if modal == 'mean':
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

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def save_checkpoint(state, filename='checkpoint.pth.tar'):
	torch.save(state, filename)

def mean_confidence_interval(data, confidence=0.95):
	a = [1.0*np.array(data[i].cpu()) for i in range(len(data))]
	n = len(a)
	m, se = np.mean(a), scipy.stats.sem(a)
	h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
	return m,h
       
def train(train_loader,params,model,optimizer,loss_fn,epoch_index):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    for episode_index, (temp2,target) in enumerate(train_loader):   
    # temp2, _ =next(iter(train_loader))
        
        # Measure data loading time
        data_time.update(time.time() - end)
                
        support,query = temp2.split([params.n_shot,params.n_query],dim=1)
        cache_values, q_values = target.split([params.n_shot,params.n_query],dim=1)

        # cache_values = F.one_hot(cache_values).half()
        cache_values = cache_values.reshape(-1)
        q_values = q_values.reshape(-1)
        cache_values, q_values = cache_values.cuda(), q_values.cuda()

        n,k,c,h,w = support.shape
        support = support.reshape(-1,c,h,w)
        support = support.cuda()
        query = query.reshape(-1,c,h,w)
        query = query.cuda()

        # ---------图像组合--------------
        
        #--------方法2：将support取50%，query填充其掩码部分，互补拼接-----------
        mask_ratio = np.random.uniform(0.25,0.75)                              # 设置不同mask
        query_patch = patchify(query)          # torch.Size([75, 196, 768])
        support_patch = patchify(support)  
        imags = random_compose(query_patch,support_patch,mask_ratio=mask_ratio)
        # query_patch, _, _ = random_masking(query_patch)         # torch.Size([75, 98, 768])
        # support_patch, _, _ = random_masking(support_patch)
        # # print(query_patch.shape)
        # # print(support_patch.shape)
        # imags = torch.cat((query_patch.unsqueeze(1).repeat(1,params.train_n_way*params.n_shot,1,1), support_patch.unsqueeze(0).repeat(params.train_n_way*params.n_query,1,1,1)), dim=2)
        # # print(imags.shape)
        imags = imags.reshape(-1,imags.shape[2],imags.shape[3])
        imags = unpatchify(imags)
        # print(imags.shape)
        label = torch.eq(q_values.unsqueeze(1).repeat(1,params.train_n_way*params.n_shot),cache_values.unsqueeze(0).repeat(params.train_n_way*params.n_query,1)).type(torch.float32)
        label = label.reshape(-1)

        outputs = model(imags)
        outputs = F.sigmoid(outputs).reshape(-1)
        loss = loss_fn(outputs,label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # pred = outputs.reshape(-1,params.train_n_way*params.n_shot).data.max(1)[1]
        pred = outputs.reshape(-1,params.train_n_way,params.n_shot).sum(-1).data.max(1)[1]
        y = np.repeat(range(params.val_n_way),params.n_query)
        y = torch.from_numpy(y)
        y = y.cuda()
        pred = pred.eq(y).sum()/query_patch.shape[0]

        losses.update(loss.item(), query_patch.shape[0])
        top1.update(pred, query_patch.shape[0])

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        #============== print the intermediate results ==============#
        if episode_index % params.print_freq == 0 and episode_index != 0:

            log('Eposide-({0}): [{1}/{2}]\t'
				'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
				'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
					epoch_index, episode_index, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses, top1=top1))


    return loss, pred
    
def validate(val_loader,params,model,epoch_index,best_prec1,loss_fn):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
  

	# switch to evaluate mode
    model.eval()
    accuracies = []


    end = time.time()
    for episode_index, (temp2,target) in enumerate(val_loader):   
    # temp2, _ =next(iter(train_loader))

        support,query = temp2.split([params.n_shot,params.n_query],dim=1)
        cache_values, q_values = target.split([params.n_shot,params.n_query],dim=1)

        # cache_values = F.one_hot(cache_values).half()
        cache_values = cache_values.reshape(-1)
        q_values = q_values.reshape(-1)
        cache_values, q_values = cache_values.cuda(), q_values.cuda()

        n,k,c,h,w = support.shape
        support = support.reshape(-1,c,h,w)
        support = support.cuda()
        query = query.reshape(-1,c,h,w)
        query = query.cuda()

        # ---------图像组合--------------
        
        #--------方法2：将support取50%，query填充其掩码部分，互补拼接-----------
        mask_ratio = np.random.uniform(0.25,0.75)  
        query_patch = patchify(query)          # torch.Size([75, 196, 768])
        support_patch = patchify(support)  
        imags = random_compose(query_patch,support_patch,mask_ratio=mask_ratio)
        # query_patch, _, _ = random_masking(query_patch)         # torch.Size([75, 98, 768])
        # support_patch, _, _ = random_masking(support_patch)
        # # print(query_patch.shape)
        # # print(support_patch.shape)
        # imags = torch.cat((query_patch.unsqueeze(1).repeat(1,params.train_n_way*params.n_shot,1,1), support_patch.unsqueeze(0).repeat(params.train_n_way*params.n_query,1,1,1)), dim=2)
        # # print(imags.shape)
        imags = imags.reshape(-1,imags.shape[2],imags.shape[3])
        imags = unpatchify(imags)
        # print(imags.shape)
        label = torch.eq(q_values.unsqueeze(1).repeat(1,params.train_n_way*params.n_shot),cache_values.unsqueeze(0).repeat(params.train_n_way*params.n_query,1)).type(torch.float32)
        label = label.reshape(-1)
        with torch.no_grad():
            outputs = model(imags)
        outputs = F.sigmoid(outputs).reshape(-1)
        loss = loss_fn(outputs,label)

        pred = outputs.reshape(-1,params.train_n_way,params.n_shot).sum(-1).data.max(1)[1]
        y = np.repeat(range(params.val_n_way),params.n_query)
        y = torch.from_numpy(y)
        y = y.cuda()
        pred = pred.eq(y).sum()/query_patch.shape[0]
        
        losses.update(loss.item(), query_patch.size(0))
        top1.update(pred, query_patch.size(0))
        accuracies.append(pred)


		# measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        best_prec1 = max(best_prec1,top1.val)
        #============== print the intermediate results ==============#
        if episode_index % params.print_freq == 0 and episode_index != 0:

            log('Test-({0}): [{1}/{2}]\t'
				'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
				'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
					epoch_index, episode_index, len(val_loader), batch_time=batch_time, loss=losses, top1=top1))
	
        # log(' * Prec@1 {top1.avg:.3f} Best_prec1 {best_prec1:.3f}'.format(top1=top1, best_prec1=best_prec1))

    return top1.avg, accuracies

    


def main():
    model = models_vit.__dict__[params.model](
        num_classes=1,
        global_pool=params.global_pool,
    )

    if params.model == 'vit_base_patch16':
        checkpoint = torch.load('/home/jiangweihao/CodeLab/PytorchCode/mae_fsl/checkpoint/mae_pretrain_vit_base.pth')
    else:
        checkpoint = torch.load('/home/jiangweihao/CodeLab/PytorchCode/mae_fsl/checkpoint/mae_pretrain_vit_large.pth')



    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # interpolate position embedding
    # interpolate_pos_embed(model, checkpoint_model)

    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)

    if params.global_pool:
        assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
    else:
        assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

    # manually initialize fc layer: following MoCo v3
    trunc_normal_(model.head.weight, std=0.01)

    # for linear prob only
    # hack: revise model's head with BN
    model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
    # freeze all but the head
    # parameters = []
    for _, p in model.named_parameters():
        p.requires_grad = False
        if params.ft and 'blocks.11.mlp' in _:
            log(_)
            p.requires_grad = True
            # parameters.append(_)
    for _, p in model.head.named_parameters():
        p.requires_grad = True

    model.to('cuda')
    # ---------------------------------------------
    loss_fn = torch.nn.MSELoss()

    # optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr = 0.01, momentum=params.momentum, weight_decay=params.weight_decay)
    
    if params.ft:
        parameters = [p for _, p in model.blocks.named_parameters() if p.requires_grad]
        parameter = [
             {'params': parameters, 'lr': 1e-3},
            {'params': model.head.parameters(), 'lr': 1e-1}]
        optimizer = torch.optim.SGD(parameter, lr=0.001, momentum=params.momentum, weight_decay=params.weight_decay)
    else:
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr = 0.01, momentum=params.momentum, weight_decay=params.weight_decay)


    # schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[2, 4, 6,8],gamma=0.1)     #[30,60]
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=2)     #[30,60]
    # epochs = params.epochs
    log('==========start training ===============')
    
    loss_all = []
    pred_all = []
    best_prec1 = 0
    for epoch in range(params.epochs): 
        log('==========training on train set===============')
        # epoch_learning_rate = 0.01
        for param_group in optimizer.param_groups:
            epoch_learning_rate = param_group['lr']          # scheduler.get_lr()
            
        log( 'Train Epoch: {}\tLearning Rate: {:.4f}'.format(
                            epoch, epoch_learning_rate))
        loss,pred = train(train_loader,params,model,optimizer,loss_fn,epoch)

        loss_all.append(loss.item())
        pred_all.append(pred.item())

        schedule.step()

        if epoch % 1 == 0:
            log('============ Validation on the val set ============')
            prec1, _ = validate(train_loader,params,model,epoch,best_prec1,loss_fn)
        
        	# record the best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        # save the checkpoint
        if is_best:
            save_checkpoint(
                {
                    'epoch_index': epoch,
                    'arch': 'opt.basemodel',
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer' : optimizer.state_dict(),
                }, os.path.join(log_path, 'model_best.pth.tar'))


        if epoch % 2 == 0:
            filename = os.path.join(log_path, 'epoch_%d.pth.tar' %epoch)
            save_checkpoint(
            {
                'epoch_index': epoch,
                'arch': "opt.basemodel",
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, filename)


if __name__ == '__main__':

    start = time.time()
    main()
    log(time.time()-start)
    log('===========================training end!===================================')



