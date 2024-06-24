import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from math import sqrt
from transformers import MixtralConfig
from einops import repeat
import pytorch_lightning as pl
class Block(nn.Module):
    def __init__(self, inputs = 3, middles = 64, outs = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(inputs, middles, 3, 1, 1)
        self.conv2 = nn.Conv2d(middles, outs, 3, 1, 1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(outs)
        self.pool = nn.Conv2d(outs, outs, 2, stride = 2)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.bn(self.conv2(x)))  
        return self.pool(x), x
class CalculateAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V):
        attention = torch.matmul(Q,torch.transpose(K, -1, -2))
        attention = torch.softmax(attention / sqrt(Q.size(-1)), dim=-1)
        attention = torch.matmul(attention,V)
        return attention
class Multi_CrossAttention(nn.Module):
    def __init__(self,in_q_dim,in_k_dim,out_dim,heads_num):
        super().__init__()
        self.in_q_dim       = in_q_dim     #1024       # 输入维度
        self.in_k_dim       = in_k_dim
        self.all_head_size  = out_dim     # 输出维度
        self.num_heads      = heads_num          # 注意头的数量
        self.h_size         = self.all_head_size // heads_num 
        assert self.all_head_size % heads_num  == 0
        self.linear_q = nn.Linear(self.in_q_dim, self.all_head_size, bias=False)
        self.linear_k = nn.Linear(self.in_k_dim, self.all_head_size, bias=False)
        self.linear_v = nn.Linear(self.in_k_dim, self.all_head_size, bias=False)
        self.linear_output = nn.Linear(self.all_head_size, self.all_head_size)
        self.norm = sqrt(self.all_head_size)
    
    def forward(self, x, y):
        batch_size = x.size(0)
        q_s = self.linear_q(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)
        k_s = self.linear_k(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)
        v_s = self.linear_v(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)
        attention = CalculateAttention()(q_s,k_s,v_s)
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size)
        output = self.linear_output(attention)

        return output
class TransformerEncoderLayer(nn.Module):
    def __init__(self,in_q_dim,in_k_dim,out_dim,heads_num,mlp_dim,moe=False,dropout=0.0):
        super(TransformerEncoderLayer, self).__init__()
        self.cross_attn = Multi_CrossAttention(in_q_dim,in_k_dim,out_dim,heads_num)
        self.self_attn = nn.MultiheadAttention(embed_dim=in_q_dim, num_heads=heads_num)
        self.norm1 = nn.LayerNorm(in_q_dim)
        self.norm2 = nn.LayerNorm(in_k_dim)
        self.norm3 = nn.LayerNorm(in_q_dim)
        self.norm4 = nn.LayerNorm(in_q_dim)
        self.mixtralconfig = MixtralConfig(
            hidden_size = in_q_dim,
            intermediate_size = out_dim,
            num_local_experts = 8,
            router_jitter_noise = 0.5
        )
        if moe:
            self.moe = MixtralSparseMoeBlock(self.mixtralconfig)
        else:
            self.moe = nn.Sequential(
                nn.Linear(in_q_dim, mlp_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_dim, in_q_dim),
                nn.Dropout(dropout)
            )
    def forward(self, x, y):
        
        x = self.norm1(x)
        y = self.norm2(y)
        
        skip0 = x
        x = self.cross_attn(x,y)
        x = self.norm3(x + skip0)
        
        skip1 = x
        x, _ = self.self_attn(x, x, x)
        
        x = self.norm4(x + skip1)
        skip2 = x

        x = self.moe(x)
        x = x + skip2
        return x
class ConvNeXtUNet(nn.Module):
    def __init__(self,):
        super().__init__()
        self.pre_model = timm.create_model('convnextv2_base',pretrained=False,pretrained_cfg_overlay=dict(file="../model/hcmae_convnext/model-1000ckpt.pth"))
        self.upsample4 = nn.ConvTranspose2d(1024, 1024, 2, stride = 2)
        self.de4 = Block(1536, 1024, 256)
        self.upsample3 = nn.ConvTranspose2d(256, 256, 2, stride = 2)
        self.de3 = Block(512, 256, 128)
        self.upsample2 = nn.ConvTranspose2d(128, 128, 2, stride = 2)
        self.de2 = Block(256, 128, 64)
        self.de1 = Block(192, 64, 64) 
        self.upsample1 = nn.ConvTranspose2d(64, 64, 4, stride = 4)
        self.de0 = Block(67, 64, 64)
        self.conv_last = nn.Conv2d(64, 1, kernel_size=1, stride = 1, padding = 0)
        self.cv_encoder1 = TransformerEncoderLayer(in_q_dim=1024,in_k_dim=4096,out_dim=1024,heads_num=4,mlp_dim=128,moe=False)
        self.cv_encoder2 = TransformerEncoderLayer(in_q_dim=512,in_k_dim=4096,out_dim=512,heads_num=4,mlp_dim=128,moe=False)
        self.cv_encoder3 = TransformerEncoderLayer(in_q_dim=256,in_k_dim=4096,out_dim=256,heads_num=4,mlp_dim=128,moe=False)
        self.cv_encoder4 = TransformerEncoderLayer(in_q_dim=128,in_k_dim=4096,out_dim=128,heads_num=4,mlp_dim=128,moe=False)
        self.cv_encoder5 = TransformerEncoderLayer(in_q_dim=128,in_k_dim=4096,out_dim=128,heads_num=4,mlp_dim=128,moe=False)
        
    def forward(self, x, y):
        xx = x
        _,f = self.pre_model.forward_intermediates(x)
        f[4] = self.cv_encoder1(f[4].reshape(-1,1024,64).transpose(1,2),y).transpose(1,2).reshape(-1,1024,8,8) + f[4]
        f[3] = self.cv_encoder2(f[3].reshape(-1,512,256).transpose(1,2),y).transpose(1,2).reshape(-1,512,16,16) + f[3]
        f[2] = self.cv_encoder3(f[2].reshape(-1,256,32*32).transpose(1,2),y).transpose(1,2).reshape(-1,256,32,32) + f[2]
        f[1] = self.cv_encoder4(f[1].reshape(-1,128,64*64).transpose(1,2),y).transpose(1,2).reshape(-1,128,64,64) + f[1]
        f[0] = self.cv_encoder5(f[0].reshape(-1,128,64*64).transpose(1,2),y).transpose(1,2).reshape(-1,128,64,64) + f[0]
        x = f[4]
        x = self.upsample4(x)
        x = torch.cat([x, f[3]], dim=1)
        _,  x = self.de4(x)
        x = self.upsample3(x)
        x = torch.cat([x, f[2]], dim=1)
        _, x = self.de3(x)
        x = self.upsample2(x)
        x = torch.cat([x, f[1]], dim=1)
        _, x = self.de2(x)
        x = torch.cat([x, f[0]], dim=1)
        _, x = self.de1(x)
        x = self.upsample1(x)
        x = torch.cat([x, xx], dim=1)
        _, x = self.de0(x)
        x = self.conv_last(x)    
        return x
class SegModel(pl.LightningModule):
    def __init__(self):
        super(SegModel, self).__init__()
        self.model = ConvNeXtUNet()
        
    def forward(self, x):
        return self.model(x['img'],x['text'])
class MoCACUnet(torch.nn.Module):
    def __init__(self):
        super(MoCACUnet,self).__init__()
        self.convnextunet = SegModel.load_from_checkpoint('./model/seg/hicur_seg/last.ckpt').model
        self.convnextunet.requires_grad_=False
        self.convnextunet.eval()
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, 33, 1024))
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, 1024))
        self.encoder = TransformerEncoderLayer(in_q_dim=1024,in_k_dim=4096,out_dim=1024,heads_num=4,mlp_dim=2048,moe=False)
        self.encoder2 = TransformerEncoderLayer(in_q_dim=1024,in_k_dim=4096,out_dim=1024,heads_num=4,mlp_dim=2048,moe=False)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 2),
        )
    def forward(self,x):
        lvs = x['pt']
        cnn_codes = []
        with torch.no_grad():
            for img in x['imgs_timm']:
                tmp = self.convnextunet.pre_model.forward_features(img)
                tmp = F.adaptive_avg_pool2d(tmp,(1,1)).reshape(32,1024)
                cnn_codes.append(tmp)
        cnn_codes = torch.stack(cnn_codes)
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=len(x['imgs_timm']))
        pos_embedding = repeat(self.pos_embed, '() n d -> b n d', b=len(x['imgs_timm']))
        xx = torch.cat((cls_tokens, cnn_codes), dim=1) + pos_embedding
        skip1 = xx
        xx = self.encoder(xx,lvs)
        xx = xx + skip1
        skip2 = xx
        xx = self.encoder2(xx,lvs)
        xx = xx + skip2
        xx = xx.permute(0, 2, 1)[:,:,0].reshape(-1,1024)
        xx = self.mlp_head(xx)
        return xx
class PModel(pl.LightningModule):
    def __init__(self):
        super(PModel, self).__init__()
        self.pmodel = MoCACUnet()    
    def forward(self, x):
        return self.pmodel(x)
class MoCANet(torch.nn.Module):
    def __init__(self):
        super(MoCANet,self).__init__()
        self.moca = PModel.load_from_checkpoint('./model/pro/hicur_pro/last.ckpt').pmodel
    def forward(self,x):
        return self.moca(x)
    def get_pre(self,x):
        return self.moca(x)
    def get_seg(self,x1,x2):
        return self.moca.convnextunet(x1,x2)
    