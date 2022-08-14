from __future__ import print_function
from pickle import FALSE
import torch
from model3 import highwayNet
from utils import ngsimDataset,maskedNLL,maskedMSE,maskedNLLTest,maskedMSETest
from torch.utils.data import DataLoader
import time
import math
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
## Network Arguments
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
args['train_flag'] = True

# Initialize network
net = highwayNet(args)
cuda_idx=0
torch.cuda.set_device(cuda_idx)
if args['use_cuda']:
    net = net.cuda(cuda_idx)

## Initialize optimizer
pretrainEpochs = 4
trainEpochs = 4
optimizer = torch.optim.Adam(net.parameters())
batch_size = 128
crossEnt = torch.nn.BCELoss()

## Initialize data loaders
trSet = ngsimDataset('./data/TrainSet.mat','./data/TrainPlans.mat')
valSet = ngsimDataset('./data/ValSet.mat','./data/ValPlans.mat')
trDataloader = DataLoader(trSet,batch_size=batch_size,shuffle=True,num_workers=8,collate_fn=trSet.collate_fn,pin_memory=True)
valDataloader = DataLoader(valSet,batch_size=batch_size,shuffle=True,num_workers=8,collate_fn=valSet.collate_fn,pin_memory=True)


## Variables holding train and validation loss values:
train_loss = []
val_loss = []
prev_val_loss = math.inf

for epoch_num in range(pretrainEpochs+trainEpochs):
    if epoch_num == 0:
        print('Pre-training with MSE loss')
    elif epoch_num == pretrainEpochs:
        print('Training with NLL loss')


    ## Train:_________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
    net.train_flag = True

    # Variables to track training performance:
    avg_tr_loss = 0
    avg_tr_time = 0
    avg_lat_acc = 0
    avg_lon_acc = 0
    net.train()

    for i, data in enumerate(trDataloader):
        # print(i)
        st_time = time.time()
        # start =time.time()
        hist, nbrs, plan, mask, plan_mask, lat_enc, lon_enc, fut, op_mask = data
        # end = time.time() 
        # print('Running time: %s Seconds'%(end-start))
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

        # Forward pass
        if args['use_maneuvers']:
            fut_pred, lat_pred, lon_pred = net(hist, nbrs, plan, mask, plan_mask, lat_enc, lon_enc)
            # Pre-train with MSE loss to speed up training
            if epoch_num < pretrainEpochs:
                l = maskedMSE(fut_pred, fut, op_mask)
            else:
            # Train with NLL loss
                l = maskedNLL(fut_pred, fut, op_mask) + crossEnt(lat_pred, lat_enc) + crossEnt(lon_pred, lon_enc)
                avg_lat_acc += (torch.sum(torch.max(lat_pred.data, 1)[1] == torch.max(lat_enc.data, 1)[1])).item() / lat_enc.size()[0]
                avg_lon_acc += (torch.sum(torch.max(lon_pred.data, 1)[1] == torch.max(lon_enc.data, 1)[1])).item() / lon_enc.size()[0]
        else:
            fut_pred = net(hist, nbrs, plan, mask, plan_mask, lat_enc, lon_enc)
            if epoch_num < pretrainEpochs:
                l = maskedMSE(fut_pred, fut, op_mask)
            else:
                l = maskedNLL(fut_pred, fut, op_mask)

        # Backprop and update weights
        optimizer.zero_grad()
        l.backward()
        a = torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
        optimizer.step()

        # Track average train loss and average train time:
        batch_time = time.time()-st_time
        avg_tr_loss += l.item()
        avg_tr_time += batch_time

        if i%100 == 99:
            eta = avg_tr_time/100*(len(trSet)/batch_size-i)
            with open('./logs/result_depthwise.txt', 'a') as f:
                print("Epoch no:",epoch_num+1,"| Epoch progress(%):",format(i/(len(trSet)/batch_size)*100,'0.2f'), 
                "| Avg train loss:",format(avg_tr_loss/100,'0.4f'),"| Acc:",format(avg_lat_acc,'0.4f'),
                format(avg_lon_acc,'0.4f'), "| Validation loss prev epoch",format(prev_val_loss,'0.4f'),
                 "| ETA(s):",int(eta),file=f)
            train_loss.append(avg_tr_loss/100)
            avg_tr_loss = 0
            avg_lat_acc = 0
            avg_lon_acc = 0
            avg_tr_time = 0
    # _________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

    # Validate:______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
    net.train_flag = False
    with open('./logs/result_depthwise.txt', 'a') as f:
        print("Epoch",epoch_num+1,'complete. Calculating validation loss...',file=f)
    avg_val_loss = 0
    avg_val_lat_acc = 0
    avg_val_lon_acc = 0
    val_batch_count = 0
    total_points = 0

    for i, data  in enumerate(valDataloader):
        # print(i)
        net.eval()
        st_time = time.time()
        hist, nbrs, plan, mask, plan_mask, lat_enc, lon_enc, fut, op_mask = data

        if args['use_cuda']:
            hist = hist.cuda(cuda_idx)
            nbrs = nbrs.cuda(cuda_idx)
            plan = plan.cuda(cuda_idx)
            mask = mask.cuda(cuda_idx)
            plan_mask = plan_mask.cuda(cuda_idx)
            lat_enc = lat_enc.cuda(cuda_idx)
            lon_enc = lon_enc.cuda(cuda_idx)
            fut = fut.cuda(cuda_idx)
            op_mask = op_mask.cuda(cuda_idx)
            
        # Forward pass
        if args['use_maneuvers']:
            if epoch_num < pretrainEpochs:
                # During pre-training with MSE loss, validate with MSE for true maneuver class trajectory
                net.train_flag = True
                fut_pred, _ , _ = net(hist, nbrs, plan, mask, plan_mask, lat_enc, lon_enc)
                l = maskedMSE(fut_pred, fut, op_mask)
            else:
                # During training with NLL loss, validate with NLL over multi-modal distribution
                fut_pred, lat_pred, lon_pred = net(hist, nbrs, plan, mask, plan_mask, lat_enc, lon_enc)
                l = maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask,avg_along_time = True)
                avg_val_lat_acc += (torch.sum(torch.max(lat_pred.data, 1)[1] == torch.max(lat_enc.data, 1)[1])).item() / lat_enc.size()[0]
                avg_val_lon_acc += (torch.sum(torch.max(lon_pred.data, 1)[1] == torch.max(lon_enc.data, 1)[1])).item() / lon_enc.size()[0]
        else:
            fut_pred = net(hist, nbrs, plan, mask, plan_mask, lat_enc, lon_enc)
            if epoch_num < pretrainEpochs:
                l = maskedMSE(fut_pred, fut, op_mask)
            else:
                l = maskedNLL(fut_pred, fut, op_mask)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
        avg_val_loss += l.item()
        val_batch_count += 1

    with open('./logs/result_depthwise.txt', 'a') as f:
        print(avg_val_loss/val_batch_count,file=f)

    # Print validation loss and update display variables
    with open('./logs/result_depthwise.txt', 'a') as f:
        print('Validation loss :',format(avg_val_loss/val_batch_count,'0.4f'),"| Val Acc:",format(avg_val_lat_acc/val_batch_count*100,'0.4f'),format(avg_val_lon_acc/val_batch_count*100,'0.4f'),file=f)
    val_loss.append(avg_val_loss/val_batch_count)
    prev_val_loss = avg_val_loss/val_batch_count
    
    torch.save(net.state_dict(), './trained_models/depthwise/planning_wave_'+str(epoch_num)+'.tar')
    # __________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

torch.save(net.state_dict(), './trained_models/depthwise/planning_wave.tar')

## 超参等
args = {}
args['use_cuda'] = True
args['encoder_size'] = 64
args['decoder_size'] = 128
args['in_length'] = 16
args['out_length'] = 25
args['grid_size'] = (13,3)
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
# metric = 'rmse'  #or rmse
# cuda_idx=2
# torch.cuda.set_device(cuda_idx)

# Initialize network
net = highwayNet(args)
net.load_state_dict(torch.load('./trained_models/depthwise/planning_wave.tar'))
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
            
    lossVals_nll +=l_nll.detach()
    counts_nll += c_nll.detach()
    lossVals_rmse +=l_rmse.detach()
    counts_rmse += c_rmse.detach()
   
        
# if metric == 'nll':
print("NLLLoss")
print(lossVals_nll / counts_nll)
# else:
print("RMSELoss")
print(torch.pow(lossVals_rmse / counts_rmse,0.5)*0.3048)   # Calculate RMSE and convert from feet to meters





