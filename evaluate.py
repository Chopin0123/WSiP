from __future__ import print_function
import torch
from model3 import highwayNet
from utils3 import ngsimDataset,maskedNLL,maskedMSETest,maskedNLLTest
from torch.utils.data import DataLoader
# from plot import trajectoryPlot
import time
import numpy as np
import pandas as pd
## 超参等
args = {}
args['use_cuda'] = True
args['encoder_size'] = 64
args['decoder_size'] = 128
args['in_length'] = 16
args['out_length'] = 25
args['grid_size'] = (13,5)
args['soc_conv_depth'] = 64
args['conv_3x1_depth'] = 16
args['dyn_embedding_size'] = 32
args['input_embedding_size'] = 32
args['num_lat_classes'] = 3
args['num_lon_classes'] = 2
args['use_maneuvers'] = True
args['use_planning'] = True
args['train_flag'] = False

# 评价指标
metric = 'rmse'  #or rmse
cuda_idx=0
torch.cuda.set_device(cuda_idx)

# Initialize network
net = highwayNet(args)
net.load_state_dict(torch.load('./trained_models/depthwise/planning_wave_6.tar'))
if args['use_cuda']:
    net = net.cuda(cuda_idx)

tsSet = ngsimDataset('data/TestSet.mat','./data/TestPlans.mat')
tsDataloader = DataLoader(tsSet,batch_size=128,shuffle=True,num_workers=16,collate_fn=tsSet.collate_fn)

lossVals_nll = torch.zeros(25).cuda(cuda_idx)
counts_nll = torch.zeros(25).cuda(cuda_idx)
lossVals_rmse = torch.zeros(25).cuda(cuda_idx)
counts_rmse = torch.zeros(25).cuda(cuda_idx)
avg_ts_time=0
for i, data in enumerate(tsDataloader):
    # print(i)
    st_time = time.time()
    hist, nbrs, plan, mask, plan_mask, lat_enc, lon_enc, fut, op_mask = data

    # 
    # if args['use_cuda']:
    hist = hist.cuda(cuda_idx)
    nbrs = nbrs.cuda(cuda_idx)
    plan = plan.cuda(cuda_idx)
    mask = mask.cuda(cuda_idx)
    plan_mask = plan_mask.cuda(cuda_idx)
    lat_enc = lat_enc.cuda(cuda_idx)
    lon_enc = lon_enc.cuda(cuda_idx)
    fut = fut.cuda(cuda_idx)
    op_mask = op_mask.cuda(cuda_idx)
    
    # if metric == 'nll':
        # Forward 
    if args['use_maneuvers']:
        fut_pred, lat_pred, lon_pred = net(hist, nbrs, plan, mask, plan_mask, lat_enc, lon_enc)
        l_nll,c_nll = maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask)
        fut_pred_max = torch.zeros_like(fut_pred[0])
        for k in range(lat_pred.shape[0]):
            lat_man = torch.argmax(lat_pred[k, :]).detach()
            lon_man = torch.argmax(lon_pred[k, :]).detach()
            indx = lon_man*3 + lat_man
            fut_pred_max[:,k,:] = fut_pred[indx][:,k,:] 
        l_rmse, c_rmse = maskedMSETest(fut_pred_max, fut, op_mask)
    else:
        fut_pred = net(hist, nbrs, plan, mask, plan_mask, lat_enc, lon_enc)
        l_nll, c_nll = maskedNLLTest(fut_pred, 0, 0, fut, op_mask,use_maneuvers=False)
        l_rmse, c_rmse = maskedMSETest(fut_pred, fut, op_mask)
            
    # if i==0:
    #     data=np.ones([25*128,2],float)
    #     data2=np.ones([25*128,2],float)
    #     fut_copy=fut.cpu().detach().numpy()
    #     pred_copy=fut_pred_max.cpu().detach().numpy()
    #     for j in range(128):
    #         data[j*25:(j+1)*25,:]=fut_copy[:,j,:]
    #         data2[j*25:(j+1)*25,:]=pred_copy[:,j,:2]
    #     # writer = pd.ExcelWriter('tg.csv')
    #     data_copy= pd.DataFrame(data)    
    #     data_copy.to_csv('truth.csv', index=False)
    #     data_copy= pd.DataFrame(data2)    
    #     data_copy.to_csv('pred.csv', index=False)
    #     data2=np.ones([25*128,5],float)
    #     for  j in range(len(fut_pred)):
    #         pred_copy=fut_pred[j].cpu().detach().numpy()
    #         for k in range(128):
    #             data2[k*25:(k+1)*25,:]=pred_copy[:,k,:]
    #     # writer = pd.ExcelWriter('tg.csv')
    #             data_copy= pd.DataFrame(data2)    
    #             data_copy.to_csv(str(j)+'.csv', index=False)
    lossVals_nll +=l_nll.detach()
    counts_nll += c_nll.detach()
    lossVals_rmse +=l_rmse.detach()
    counts_rmse += c_rmse.detach()
    # batch_time = time.time()-st_time
    # avg_ts_time+=batch_time
    # if i%100 == 99:
    #     num=i+1
    #     with open('./logs/prediction_time.txt', 'a') as f:
    #         print("batch:",num,"time:",avg_ts_time*1000,file=f)
    
        
# if metric == 'nll':
print("NLLLoss")
print(lossVals_nll / counts_nll)
# else:
print("RMSELoss")
print(torch.pow(lossVals_rmse / counts_rmse,0.5)*0.3048)   # Calculate RMSE and convert from feet to meters


