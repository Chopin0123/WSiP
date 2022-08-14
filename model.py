from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn
from utils import outputActivation
import torch.nn.functional as F
from timm.models.layers import DropPath

class highwayNet(nn.Module):

    ## Initialization
    def __init__(self,args):
        super(highwayNet, self).__init__()

        ## Unpack arguments
        self.args = args

        ## 用gpu
        self.use_cuda = args['use_cuda']

        #多模态
        self.use_maneuvers = args['use_maneuvers']

        # 是否训练
        self.train_flag = args['train_flag']

        # 是否使用规划
        self.use_planning = args['use_planning']
        
        ## Sizes of network layers
        self.encoder_size = args['encoder_size']
        self.decoder_size = args['decoder_size']
        self.in_length = args['in_length']
        self.out_length = args['out_length']
        self.grid_size = args['grid_size']
        self.soc_conv_depth = args['soc_conv_depth']
        self.conv_3x1_depth = args['conv_3x1_depth']
        self.dyn_embedding_size = args['dyn_embedding_size']
        self.input_embedding_size = args['input_embedding_size']
        self.num_lat_classes = args['num_lat_classes']
        self.num_lon_classes = args['num_lon_classes']
        self.soc_embedding_size = (((args['grid_size'][0]-4)+1)//2)*self.conv_3x1_depth

        ## Define network weights

        # 输入
        self.ip_emb = torch.nn.Linear(2,self.input_embedding_size)

        # LSTM encoder
        self.enc_lstm = torch.nn.LSTM(self.input_embedding_size,self.encoder_size,1)

        # Vehicle dynamics embedding
        self.dyn_emb = torch.nn.Linear(self.encoder_size,self.dyn_embedding_size)
     
        # Interaction decoder
        self.soc_emb=WaveBlock(self.encoder_size)
        # fusion module
        self.fus_emb=WaveBlock(self.encoder_size)
        # 未来规划
        if self.use_planning:
            self.plan_lstm = torch.nn.LSTM(self.input_embedding_size,self.encoder_size,1)

        # LSTM Decoder
        if self.use_maneuvers:
            # self.dec_lstm = torch.nn.LSTM( self.encoder_size + self.num_lat_classes + self.num_lon_classes, self.decoder_size)
            self.dec_lstm = torch.nn.LSTM(self.encoder_size + self.dyn_embedding_size + self.num_lat_classes + self.num_lon_classes, self.decoder_size)
        else:
            # self.dec_lstm = torch.nn.LSTM(self.encoder_size, self.decoder_size)
            self.dec_lstm = torch.nn.LSTM(self.encoder_size + self.dyn_embedding_size, self.decoder_size)
        # Output layers:
        self.op = torch.nn.Linear(self.decoder_size,5)
        # self.op_lat = torch.nn.Linear(self.encoder_size, self.num_lat_classes)
        # self.op_lon = torch.nn.Linear(self.encoder_size, self.num_lon_classes)
        self.op_lat = torch.nn.Linear(self.encoder_size + self.dyn_embedding_size, self.num_lat_classes)
        self.op_lon = torch.nn.Linear(self.encoder_size + self.dyn_embedding_size, self.num_lon_classes)

        # Activations:
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)


    ## Forward Pass
    def forward(self,hist,nbrs,plan,masks,plan_masks,lat_enc,lon_enc):

        ## 历史信息:
        _,(hist_enc,_) = self.enc_lstm(self.leaky_relu(self.ip_emb(hist)))
        hist_enc = self.leaky_relu(self.dyn_emb(hist_enc.view(hist_enc.shape[1],hist_enc.shape[2])))

        ## 邻居信息
        _, (nbrs_enc,_) = self.enc_lstm(self.leaky_relu(self.ip_emb(nbrs)))
        nbrs_enc = nbrs_enc.view(nbrs_enc.shape[1], nbrs_enc.shape[2])

        ## 规划信息
        if self.use_planning:
            _, (plan_enc,_) = self.plan_lstm(self.leaky_relu(self.ip_emb(plan)))
            plan_enc = plan_enc.view(plan_enc.shape[1], plan_enc.shape[2])
        
        ## Masked scatter
        soc_enc = torch.zeros_like(masks).float()
        soc_enc = soc_enc.masked_scatter_(masks, nbrs_enc)
        soc_enc = soc_enc.permute(0,3,2,1)
        _,(hist_copy,_) = self.enc_lstm(self.leaky_relu(self.ip_emb(hist)))
        soc_enc[:,:,6,2]=hist_copy[0]
        soc_enc=self.soc_emb(soc_enc) 
        soc_enc=soc_enc[:,:,6,2]
        fin_enc=soc_enc
        ## 拼接
        # enc = torch.cat((soc_enc,hist_enc),1)
        # 未来规划
        if self.use_planning:
            fusion_enc = torch.zeros_like(plan_masks).float()
            fusion_enc = fusion_enc.masked_scatter_(plan_masks, plan_enc)
            fusion_enc = fusion_enc.permute(0,3,2,1)
            fusion_enc[:,:,6,2] = soc_enc
            fusion_enc=self.fus_emb(fusion_enc) 
            fusion_enc=fusion_enc[:,:,6,2]
            fin_enc=fusion_enc
        # enc=fusion_enc
        enc = torch.cat((fin_enc,hist_enc),1)
        
        if self.use_maneuvers:
            ## Maneuver recognition:
            lat_pred = self.softmax(self.op_lat(enc))
            lon_pred = self.softmax(self.op_lon(enc))

            if self.train_flag:
                ## 拼接历史轨迹编码与运动模式的one-hot编码
                enc = torch.cat((enc, lat_enc, lon_enc), 1)
                fut_pred = self.decode(enc)
                return fut_pred, lat_pred, lon_pred
            else:
                fut_pred = []
                ## 
                for k in range(self.num_lon_classes):
                    for l in range(self.num_lat_classes):
                        lat_enc_tmp = torch.zeros_like(lat_enc)
                        lon_enc_tmp = torch.zeros_like(lon_enc)
                        lat_enc_tmp[:, l] = 1
                        lon_enc_tmp[:, k] = 1
                        enc_tmp = torch.cat((enc, lat_enc_tmp, lon_enc_tmp), 1)
                        fut_pred.append(self.decode(enc_tmp))
                return fut_pred, lat_pred, lon_pred
        else:
            fut_pred = self.decode(enc)
            return fut_pred


    def decode(self,enc):
        enc = enc.repeat(self.out_length, 1, 1)
        h_dec, _ = self.dec_lstm(enc)
        h_dec = h_dec.permute(1, 0, 2)
        fut_pred = self.op(h_dec)
        fut_pred = fut_pred.permute(1, 0, 2)
        fut_pred = outputActivation(fut_pred)
        return fut_pred

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x   

class PATM(nn.Module):
    def __init__(self, dim, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,mode='fc'):
        super().__init__()
        
        self.fc_h = nn.Conv2d(dim, dim, 1, 1,bias=qkv_bias)
        self.fc_w = nn.Conv2d(dim, dim, 1, 1,bias=qkv_bias) 
        self.fc_c = nn.Conv2d(dim, dim, 1, 1,bias=qkv_bias)
        
        self.tfc_h = nn.Conv2d(2*dim, dim, (1,7), stride=1, padding=(0,7//2), groups=dim, bias=False) 
        self.tfc_w = nn.Conv2d(2*dim, dim, (7,1), stride=1, padding=(7//2,0), groups=dim, bias=False)  
        self.reweight = Mlp(dim, dim // 4, dim * 3)
        self.proj = nn.Conv2d(dim, dim, 1, 1,bias=True)
        self.proj_drop = nn.Dropout(proj_drop)   
        self.mode=mode
        #对h和w都学出相位
        if mode=='fc':
            self.theta_h_conv=nn.Sequential(nn.Conv2d(dim, dim, 1, 1,bias=True),nn.BatchNorm2d(dim),nn.ReLU())
            self.theta_w_conv=nn.Sequential(nn.Conv2d(dim, dim, 1, 1,bias=True),nn.BatchNorm2d(dim),nn.ReLU())  
        else:
            self.theta_h_conv=nn.Sequential(nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim, bias=False),nn.BatchNorm2d(dim),nn.ReLU())
            self.theta_w_conv=nn.Sequential(nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim, bias=False),nn.BatchNorm2d(dim),nn.ReLU()) 
                    


    def forward(self, x):
        B, C, H, W = x.shape
        # C, H, W = x.shape
        #相位
        theta_h=self.theta_h_conv(x)
        theta_w=self.theta_w_conv(x)
        #Channel-FC提取振幅
        x_h=self.fc_h(x)
        x_w=self.fc_w(x)      
        #用欧拉公式对特征进行展开
        x_h=torch.cat([x_h*torch.cos(theta_h),x_h*torch.sin(theta_h)],dim=1)
        x_w=torch.cat([x_w*torch.cos(theta_w),x_w*torch.sin(theta_w)],dim=1)
        #Token-FC
        h = self.tfc_h(x_h)
        w = self.tfc_w(x_w)
        c = self.fc_c(x)
        a = F.adaptive_avg_pool2d(h + w + c,output_size=1)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)
        x = h * a[0] + w * a[1] + c * a[2]
        x = self.proj(x)
        x = self.proj_drop(x)           
        return x
   
class WaveBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.BatchNorm2d, mode='fc'):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PATM(dim, qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop,mode=mode)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) 
        x = x + self.drop_path(self.mlp(self.norm2(x))) 
        return x