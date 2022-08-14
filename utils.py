from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
import scipy.io as scp
import numpy as np
import torch
from scipy.optimize import curve_fit
#___________________________________________________________________________________________________________________________

### Dataset class for the NGSIM dataset
cuda_idx=3
class ngsimDataset(Dataset):

    def __init__(self, mat_file, plan_file,t_h=30, t_f=50, d_s=2, enc_size = 64, grid_size = (13,5)):
        self.D = scp.loadmat(mat_file)['traj']
        self.T = scp.loadmat(mat_file)['tracks']
        self.P = scp.loadmat(plan_file)['tracks']
        self.t_h = t_h  # length of track history
        self.t_f = t_f  # length of predicted trajectory
        self.d_s = d_s  # down sampling rate of all sequences
        self.enc_size = enc_size # size of encoder LSTM
        self.grid_size = grid_size # size of social context grid
        # self.fit_plan_traj = fit_plan_traj
        # self.further_ds_plan = fit_plan_further_ds
    def __len__(self):
        return len(self.D)

    def __getitem__(self, idx):

        dsId = self.D[idx, 0].astype(int) #数据集id
        vehId = self.D[idx, 1].astype(int) #汽车id
        # if vehId>2800:
        #     print(vehId)
        t = self.D[idx, 2]  #帧号
        grid = self.D[idx,8:] #每个网格内的汽车的id号
        neighbors = []  #邻居
        neighbors_plan=[]
        hist = self.getHistory(vehId,t,vehId,dsId)
        fut = self.getFuture(vehId,t,dsId)
        for i in grid:
            neighbors.append(self.getHistory(i.astype(int), t,vehId,dsId))
            neighbors_plan.append(np.empty([0,2]))#得到邻居的未来规划轨迹(相对位置)   
         #得到规划路径
        mids=[19,32,45]
        for mid in mids:
            flag=[False]*2
            for i in range(1,3):
                if flag[0]==False and len(neighbors[mid-i])>0:
                    flag[0]=True 
                    planFut=self.getFitted(grid[mid-i],t,vehId,dsId)
                    neighbors_plan[mid-i]=planFut
                if flag[1]==False and len(neighbors[mid+i])>0:
                    flag[1]=True 
                    planFut2=self.getFitted(grid[mid+i],t,vehId,dsId)
                    neighbors_plan[mid+i]=planFut2
                if flag[0] and flag[1]:
                    break
        if len(neighbors[19])>0:
            planFut=self.getFitted(grid[19],t,vehId,dsId)
            neighbors_plan[19]=planFut
        if len(neighbors[45])>0:
            planFut2=self.getFitted(grid[45],t,vehId,dsId)
            neighbors_plan[45]=planFut2
        #通过拟合预测的轨迹
        # if self.fit_plan_traj:
        #     for i in range(16,50):
        #         history=neighbors[i]
        #         planFut=neighbors_plan[i]
        #         if len(planFut)>0 and len(history)>0:
        #             planFut=self.getFitted(history)
        #             neighbors_plan[i]=planFut 
        # arr=np.array(neighbors_plan)
        # Maneuvers 'lon_enc' = one-hot vector, 'lat_enc = one-hot vector
        lon_enc = np.zeros([2])
        lon_enc[int(self.D[idx, 7] - 1)] = 1
        lat_enc = np.zeros([3])
        lat_enc[int(self.D[idx, 6] - 1)] = 1

        return hist,fut,neighbors,neighbors_plan,lat_enc,lon_enc


    ## Helper function to get track history
    def getHistory(self,vehId,t,refVehId,dsId):
        
        if vehId == 0:
            return np.empty([0,2])
        else:
            if self.T.shape[1]<=vehId-1:
                return np.empty([0,2])
            refTrack = self.T[dsId-1][refVehId-1].transpose()
            vehTrack = self.T[dsId-1][vehId-1].transpose()
            refPos = refTrack[np.where(refTrack[:,0]==t)][0,1:3]

            if vehTrack.size==0 or np.argwhere(vehTrack[:, 0] == t).size==0:
                 return np.empty([0,2])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s,1:3]-refPos#相对位置

            if len(hist) < self.t_h//self.d_s + 1:
                return np.empty([0,2])
            return hist

    def getFitted(self,vehId, t,refVehId,dsId):
        vehId=vehId.astype(int)
        if vehId == 0:
            return np.empty([0,2])
        else:
            if self.T.shape[1]<=vehId-1:
                return np.empty([0,2])
            refTrack = self.T[dsId-1][refVehId-1].transpose()
            refPos = refTrack[np.where(refTrack[:,0]==t)][0,1:3]
            vehTrack = self.P[dsId-1][vehId-1].transpose()
            if vehTrack.size==0 or np.argwhere(vehTrack[:, 0] == t).size==0:
                 return np.empty([0,2])
            else:
                plan = vehTrack[np.where(vehTrack[:,0]==t)][0,1:21]
                if not np.where(plan[plan<0]):
                    return np.empty([0,2])
                else:
                    fut=np.zeros((10,2))
                    fut[:,0]=plan[::2]
                    fut[:,1]=plan[1:21:2]
                    fut = fut-refPos #绝对位置
                    return fut
            

    ## Helper function to get track future
    def getFuture(self, vehId, t,dsId):
        vehTrack = self.T[dsId-1][vehId-1].transpose()
        refPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:3] #当前位置
        stpt = np.argwhere(vehTrack[:, 0] == t).item() + self.d_s
        enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + self.t_f + 1)
        fut = vehTrack[stpt:enpt:self.d_s,1:3]-refPos #相对位置
        return fut

    def getRelativeFuture(self, vehId, t,refVehId,dsId):
        if vehId == 0:
            return np.empty([0,2])
        else:
            if self.T.shape[1]<=vehId-1:
                return np.empty([0,2])
            refTrack = self.T[dsId-1][refVehId-1].transpose()
            refPos = refTrack[np.where(refTrack[:,0]==t)][0,1:3]
            vehTrack = self.T[dsId-1][vehId-1].transpose()
            if vehTrack.size==0 or np.argwhere(vehTrack[:, 0] == t).size==0:
                 return np.empty([0,2])
            stpt = np.argwhere(vehTrack[:, 0] == t).item() + self.d_s
            enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + self.t_f + 1)
            fut = vehTrack[stpt:enpt:self.d_s,1:3]-refPos #绝对位置
            if len(fut)<self.t_f//self.d_s:
                return np.empty([0,2])
            return fut

    ## Collate function for dataloader
    def collate_fn(self, samples):
        #每一个sample的邻居数
       
        # Initialize neighbors and neighbors length batches:
        nbr_batch_size = 0
        # count=0 这个batch中共有多少个邻居
        for _,_,nbrs,_,_,_ in samples:
            nbr_batch_size += sum([len(nbrs[i])!=0 for i in range(len(nbrs))])
        plan_batch_size = 0
        for _,_,_,plans,_,_ in samples:
            plan_batch_size += sum([len(plans[i])!=0 for i in range(len(plans))])
        # count+=1
        maxlen = self.t_h//self.d_s + 1
        nbrs_batch = torch.zeros(maxlen,nbr_batch_size,2)
        #规划长度
        planlen = 20//self.d_s
        plan_batch =  torch.zeros(planlen,nbr_batch_size,2)

        # Initialize social mask batch:
        pos = [0, 0]
        mask_batch = torch.zeros(len(samples), self.grid_size[1],self.grid_size[0],self.enc_size)
        mask_batch = mask_batch.byte().bool()
        plan_mask_batch = torch.zeros(len(samples), self.grid_size[1],self.grid_size[0],self.enc_size)
        plan_mask_batch=plan_mask_batch.byte().bool()
        
        # Initialize history, history lengths, future, output mask, lateral maneuver and longitudinal maneuver batches:
        hist_batch = torch.zeros(maxlen,len(samples),2)
        fut_batch = torch.zeros(self.t_f//self.d_s,len(samples),2)
        op_mask_batch = torch.zeros(self.t_f//self.d_s,len(samples),2)
        lat_enc_batch = torch.zeros(len(samples),3)
        lon_enc_batch = torch.zeros(len(samples), 2)


        count = 0
        count2 = 0
        nbrs_num=torch.zeros(len(samples))
        plans_num=torch.zeros(len(samples))
        for sampleId,(hist, fut, nbrs, plans, lat_enc, lon_enc) in enumerate(samples):

            hist_batch[0:len(hist),sampleId,0] = torch.from_numpy(hist[:, 0])
            hist_batch[0:len(hist), sampleId, 1] = torch.from_numpy(hist[:, 1])
            fut_batch[0:len(fut), sampleId, 0] = torch.from_numpy(fut[:, 0])
            fut_batch[0:len(fut), sampleId, 1] = torch.from_numpy(fut[:, 1])
            op_mask_batch[0:len(fut),sampleId,:] = 1
            lat_enc_batch[sampleId,:] = torch.from_numpy(lat_enc)
            lon_enc_batch[sampleId, :] = torch.from_numpy(lon_enc)

            num=0
            # mid = [self.grid_size[0]/2,self.grid_size[1]/2] #(6,2)
            for id,nbr in enumerate(nbrs):
                if len(nbr)!=0:  #该邻居的历史记录条数，即是否有该邻居
                    nbrs_batch[0:len(nbr),count,0] = torch.from_numpy(nbr[:, 0]) #x历史坐标
                    nbrs_batch[0:len(nbr), count, 1] = torch.from_numpy(nbr[:, 1]) #y历史坐标
                    pos[0] = id % self.grid_size[0] #在网格中的位置 ,行，0-12
                    pos[1] = id // self.grid_size[0] #列，0-5
                    mask_batch[sampleId,pos[1],pos[0],:] = torch.ones(self.enc_size).byte().bool()
                    num+=1
                    count+=1
                    #局部邻居的mask
            for id, plan in enumerate(plans):
                if len(plan)!=0:
                    plan_batch[0:len(plan),count2,0] = torch.from_numpy(plan[:, 0]) #x规划坐标
                    plan_batch[0:len(plan), count2, 1] = torch.from_numpy(plan[:, 1]) #y规划坐标
                    pos[0] = id % self.grid_size[0] #在网格中的位置 ,行，0-12
                    pos[1] = id // self.grid_size[0] #列，0-5
                    # if abs(pos[1]-mid[1]<=1) and abs(pos[0]-mid[0]<=3):
                    plan_mask_batch[sampleId,pos[1],pos[0],:] = torch.ones(self.enc_size).byte().bool()
                    count2+=1
            nbrs_num[sampleId]=num
        #该车的历史轨迹，所有邻居的历史轨迹，邻居的mask，横向运动模式，纵向运动模式，该车的未来轨迹，局部邻居的
        return hist_batch, nbrs_batch, plan_batch, mask_batch, plan_mask_batch, lat_enc_batch, lon_enc_batch, fut_batch, op_mask_batch
    
   
#________________________________________________________________________________________________________________________________________

 ## Quintic spline definition.
# def quintic_spline(x, z, a, b, c, d, e):
#     return z + a * x + b * x ** 2 + c * x ** 3 + d * x ** 4 + e * x ** 5
def quintic_spline(x, z, a, b, c):
    return z + a * x + b * x ** 2 + c * x ** 3

## Fitting the trajectory of one planning circle by quintic spline, with the current location fixed.
#五次样条插值,返回的是z,a,b,c,d,e
# def fitting_traj_by_qs(x, y):
#     param, loss = curve_fit(quintic_spline, x, y,
#         bounds=([y[0], -np.inf, -np.inf, -np.inf, -np.inf, -np.inf], [y[0]+1e-6, np.inf, np.inf, np.inf, np.inf, np.inf]))
#     return param
def fitting_traj_by_qs(x, y):
    param, loss = curve_fit(quintic_spline, x, y,
        bounds=([y[0], -np.inf, -np.inf, -np.inf], [y[0]+1e-6, np.inf, np.inf, np.inf]))
    return param
def outputActivation(x,displacement=True):
    if displacement:
        x[:, :, 0:2] = torch.stack([torch.sum(x[0:i, :, 0:2], dim=0) for i in range(1, x.shape[0] + 1)], 0)
    muX = x[:,:,0:1]
    muY = x[:,:,1:2]
    sigX = x[:,:,2:3]
    sigY = x[:,:,3:4]
    rho = x[:,:,4:5]
    sigX = torch.exp(sigX)
    sigY = torch.exp(sigY)
    rho = torch.tanh(rho)
    out = torch.cat([muX, muY, sigX, sigY, rho],dim=2)
    return out

def maskedNLL(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:,:,0]
    muY = y_pred[:,:,1]
    sigX = y_pred[:,:,2]
    sigY = y_pred[:,:,3]
    rho = y_pred[:,:,4]
    ohr = torch.pow(1-torch.pow(rho,2),-0.5)
    x = y_gt[:,:, 0]
    y = y_gt[:,:, 1]
    out = 0.5*torch.pow(ohr, 2)*(torch.pow(sigX, 2)*torch.pow(x-muX, 2) + torch.pow(sigY, 2)*torch.pow(y-muY, 2) - 2*rho*torch.pow(sigX, 1)*torch.pow(sigY, 1)*(x-muX)*(y-muY)) - torch.log(sigX*sigY*ohr) + 1.8379
    # out = 0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2) - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr) + 1.8379 - 0.5160
    acc[:,:,0] = out
    acc[:,:,1] = out
    acc = acc*mask
    lossVal = torch.sum(acc)/torch.sum(mask)
    return lossVal

def maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask, num_lat_classes=3, num_lon_classes = 2,use_maneuvers = True, avg_along_time = False):
    if use_maneuvers:
        acc = torch.zeros(op_mask.shape[0],op_mask.shape[1],num_lon_classes*num_lat_classes).cuda(cuda_idx)
        count = 0
        for k in range(num_lon_classes):
            for l in range(num_lat_classes):
                wts = lat_pred[:,l]*lon_pred[:,k]
                wts = wts.repeat(len(fut_pred[0]),1)
                y_pred = fut_pred[k*num_lat_classes + l]
                y_gt = fut
                muX = y_pred[:, :, 0] #x轴的均值
                muY = y_pred[:, :, 1] #y轴的均值
                sigX = y_pred[:, :, 2] #x轴的方差
                sigY = y_pred[:, :, 3] #y轴的方差
                rho = y_pred[:, :, 4] 
                ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
                x = y_gt[:, :, 0]
                y = y_gt[:, :, 1]
           
                out = -(0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2) - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr) + 1.8379 - 0.5160)
                acc[:, :, count] =  out + torch.log(wts)
                count+=1
        acc = -logsumexp(acc, dim = 2)
        acc = acc * op_mask[:,:,0]
        if avg_along_time:
            lossVal = torch.sum(acc) / torch.sum(op_mask[:, :, 0])
            return lossVal
        else:
            lossVal = torch.sum(acc,dim=1)
            counts = torch.sum(op_mask[:,:,0],dim=1)
            return lossVal,counts
    else:
        acc = torch.zeros(op_mask.shape[0], op_mask.shape[1], 1).cuda(cuda_idx)
        y_pred = fut_pred
        y_gt = fut
        muX = y_pred[:, :, 0]
        muY = y_pred[:, :, 1]
        sigX = y_pred[:, :, 2]
        sigY = y_pred[:, :, 3]
        rho = y_pred[:, :, 4]
        ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
        x = y_gt[:, :, 0]
        y = y_gt[:, :, 1]
        out = 0.5*torch.pow(ohr, 2)*(torch.pow(sigX, 2)*torch.pow(x-muX, 2) + torch.pow(sigY, 2)*torch.pow(y-muY, 2) - 2 * rho*torch.pow(sigX, 1)*torch.pow(sigY, 1)*(x-muX)*(y-muY)) - torch.log(sigX*sigY*ohr) + 1.8379
        # out = 0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2) - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr) + 1.8379 - 0.5160
        acc[:, :, 0] = out
        acc = acc * op_mask[:, :, 0:1]
        if avg_along_time:
            lossVal = torch.sum(acc[:, :, 0]) / torch.sum(op_mask[:, :, 0])
            return lossVal
        else:
            lossVal = torch.sum(acc[:,:,0], dim=1)
            counts = torch.sum(op_mask[:, :, 0], dim=1)
            return lossVal,counts

def maskedMSE(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:,:,0]
    muY = y_pred[:,:,1]
    x = y_gt[:,:, 0]
    y = y_gt[:,:, 1]
    out = torch.pow(x-muX, 2) + torch.pow(y-muY, 2)
    acc[:,:,0] = out
    acc[:,:,1] = out
    acc = acc*mask
    lossVal = torch.sum(acc)/torch.sum(mask)
    return lossVal

def maskedMSETest(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask
    lossVal = torch.sum(acc[:,:,0],dim=1)
    counts = torch.sum(mask[:,:,0],dim=1)
    return lossVal, counts

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs
