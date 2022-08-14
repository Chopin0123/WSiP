from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
import scipy.io as scp
import numpy as np
from scipy.optimize import curve_fit
import cv2

mat_file='./data/TrainSet.mat'
fut_horizon=20
# def quintic_spline(x, z, a, b, c, d ,e):
#     return z + a * x + b * x ** 2 + c * x ** 3  + d * x ** 4 + e * x ** 5
def quintic_spline(x, z, a):
    return z + a * x 
def fitting_traj_by_qs(x, y):
    param, loss = curve_fit(quintic_spline, x, y,
        bounds=([y[0], -np.inf], [y[0]+1e-6, np.inf]))
    return param

class KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.03
    def predict(self, coordX, coordY):
        ''' This function estimates the position of the object'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        x, y = int(predicted[0]), int(predicted[1])
        return x, y
    
kf = KalmanFilter() 

class processMat():
    def __init__(self, mat_file, t_h=30, t_f=50, d_s=2, grid_size = (13,5)):
        self.D = scp.loadmat(mat_file)['traj']
        self.T = scp.loadmat(mat_file)['tracks']
        self.t_h = t_h  # length of track history
        self.t_f = t_f  # length of predicted trajectory
        self.d_s = d_s  # down sampling rate of all sequences
        self.grid_size = grid_size # size of social context grid
        
        # self.P=np.zeros((self.T.shape[0],self.T.shape[1]))
        # self.P=[[],[],[],[],[],[]]
        self.flag=np.zeros((self.T.shape[0],self.T.shape[1],20000))
        self.P=[]
  
    def getHistory(self,vehId,t,dsId):
        if vehId == 0:
            return np.empty([0,2])
        else:
            if self.T.shape[1]<=vehId-1:
                return np.empty([0,2])
            vehTrack = self.T[dsId-1][vehId-1].transpose()
            if vehTrack.size==0 or np.argwhere(vehTrack[:, 0] == t).size==0:
                    return np.empty([0,2])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s,1:3]
            if len(hist) < self.t_h//self.d_s + 1:
                return np.empty([0,2])
            return hist
            
    def getFitted(self,history): 
        planFut=np.zeros((int(fut_horizon/self.d_s),2))
        # planFitted=[]
        for i in range(history.shape[0]):
            predicted = kf.predict(history[i,0], history[i,1])
        planFut[0,1]=predicted[1]
        for i in range(int(fut_horizon/self.d_s)-1):
            predicted = kf.predict(predicted[0], predicted[1])
            planFut[i+1,1]=predicted[1]
        # return planFitted   
        wayPoint_to_fit        = np.arange(-self.t_h-1, 0, self.d_s) 
        wayPoint = np.arange(0, fut_horizon , self.d_s)
        # planFut_to_fit = history[::self.further_ds_plan, ] #26*2 未来的规划的轨迹
        laterParam = fitting_traj_by_qs(wayPoint_to_fit, history[:, 0]) #x坐标进行拟合
        # longiParam = fitting_traj_by_qs(wayPoint_to_fit, planFut_to_fit[:, 1]) #y坐标进行拟合
        x_plan=quintic_spline(wayPoint, *laterParam)
        x_plan[x_plan<(x_plan[0]-10)]=x_plan[0]-10
        x_plan[x_plan<0] = 0
        planFut[:, 0] = x_plan
        # planFut[:, 1] = quintic_spline(wayPoint, *longiParam)
        return planFut
    
    def getNum(self):
        for idx in range(self.D.shape[0]):
            if idx%100==99:
                print(idx)
            dsId = self.D[idx, 0].astype(int) #数据集id
            t = self.D[idx, 2].astype(int)  #帧号
            grid = self.D[idx,8:] #每个网格内的汽车的id号
            neighbors = []  #邻居历史轨迹
            neighbors_plan=[]
            for i in grid:
                neighbors.append(self.getHistory(i.astype(int), t, dsId))
                neighbors_plan.append(np.empty([0,2]))#得到邻居的未来规划轨迹(相对位置)   
            #得到规划路径
            mids=[19,32,45]
            for mid in mids:
                flag=[False]*2
                for i in range(1,6):
                    if flag[0]==False and len(neighbors[mid-i])>=1:
                        flag[0]=True 
                        planFut=self.getFitted(neighbors[mid-i])
                        if len(planFut)>0 and self.flag[dsId-1,grid[mid-i].astype(int)-1,t]<=0:
                           self.getIndex(dsId,grid,t,planFut,mid-i)
                    if  len(neighbors[mid+i])>=1 and (flag[1]==False):
                        flag[1]=True 
                        planFut=self.getFitted(neighbors[mid+i])
                        # print(grid[mid+i].astype(int))
                        if len(planFut)>0 and self.flag[dsId-1,grid[mid+i].astype(int)-1,t]<=0:
                            # print(str(dsId)+" "+str(grid[mid+i])+" "+str(t))
                            self.getIndex(dsId,grid,t,planFut,mid+i)
                    if flag[0] and flag[1]:
                        break
            if len(neighbors[19])>0:
                planFut=self.getFitted(neighbors[19])
                if len(planFut)>0 and self.flag[dsId-1,grid[19].astype(int)-1,t]<=0:
                    self.getIndex(dsId,grid,t,planFut,19)
            if len(neighbors[45])>0:
                planFut=self.getFitted(neighbors[45])
                if len(planFut)>0 and self.flag[dsId-1,grid[45].astype(int)-1,t]<=0:
                    self.getIndex(dsId,grid,t,planFut,45)
        return self.P
    
    def getIndex(self,dsId,grid,t,planFut,index):
        self.flag[dsId-1,grid[index].astype(int)-1,t]=1
        plan=[]
        plan.append(dsId)
        plan.append(grid[index].astype(int))
        plan.append(t)
        for i in range(len(planFut)):
            plan.append(planFut[i][0])
            plan.append(planFut[i][1])
        if grid[index].astype(int)-1<=self.T.shape[1]:
            self.P.append(plan)    
            
def main():
    p=processMat(mat_file)
    planFus=p.getNum()
    planFus=np.array(planFus)
    
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=2)   #设精度
    # np.savetxt('data_name‘, data.view(-1, 1), fmt='%.04f')   #保留4位小数
    # list_txt(path='plans.txt', list=planFus)
    np.savetxt("./data/TrainPlanSet.txt",planFus,fmt='%.02f')
    print("hello")
    # scp.savemat('plans.mat', {'data':planFus})
    # np.savetxt("./data/plans.txt",planFus)

if __name__=='__main__':
    main()