import torch

from models.models_mae import mae_vit_base_patch16


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

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    # -------------------- 导入预训练模型 -------------------------
    model = mae_vit_base_patch16()

    model_dict = model.state_dict()                                    # 取出自己网络的参数字典
    # 查看网络参数
    for k, v in model_dict.items():
        print(k)
        print(v)
        break

    model.load_state_dict(torch.load('/home/jiangweihao/code/MAE_fsl/mae_pretrain_vit_base.pth')['model'],strict=False)
    model.cuda()
    model.eval()

    model_dict = model.state_dict()                                    # 取出自己网络的参数字典
    # 查看网络参数
    for k, v in model_dict.items():
        print(k)
        print(v)
        break

    img = torch.randn([5,3,224,224]).cuda()
    cls_token, patch_feature = catch_feature(img,model)
    print(cls_token.shape)
    print(patch_feature.shape)

