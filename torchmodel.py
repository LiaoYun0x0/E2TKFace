import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmodule import *
from loss import getLossFunction
from resnet_e2 import E2_ResNetFPN_8_2
from max_vit.tkwvit import  TopKWindowViT
from transformer import LocalFeatureTransformer,GlobalFeatureTransformer,PositionEncodingSine
from einops import rearrange

class InitLoss(nn.Module):
    def __init__(self):
        super(InitLoss, self).__init__()
        self.criterion = getLossFunction('fwrse')(1.0)#论文损失，UVpos图的损失
        self.metrics = getLossFunction('nme')(1.0)

    def forward(self, posmap, gt_posmap):
        loss_posmap = self.criterion(gt_posmap, posmap)
        total_loss = loss_posmap
        metrics_posmap = self.metrics(gt_posmap, posmap)
        return total_loss, metrics_posmap

_cfg = {}
_cfg["arc_m"] = 0.2
_cfg["arc_s"] = 64
_cfg["global_feature_dim"] = 512
_cfg['temp_bug_fix'] = False

# global transformer parameters 不是空的，'等于号'传的是地址
_cfg["gl_cfg"] = {}#
_gl_cfg = _cfg["gl_cfg"]
_gl_cfg["d_model"] = 512
_gl_cfg["layer_names"] = ["self"]
_gl_cfg["nhead"] = 8
_gl_cfg["attention"] = "linear"
#{'gl_cfg': {'d_model': 1024, 'layer_names': ['self', 'self'], 'nhead': 8, 'attention': 'linear'}}

class InitPRN2(nn.Module):
    def __init__(self):
        super(InitPRN2, self).__init__()
        self.feature_size = 16
        feature_size = self.feature_size
        self.layer0 = Conv2d_BN_AC(in_channels=3, out_channels=feature_size, kernel_size=3, stride=1, padding=1)  # 256 x 256 x 16
        #self.encoder = nn.Sequential(
            # PRNResBlock(in_channels=feature_size, out_channels=feature_size * 2, kernel_size=3, stride=2, with_conv_shortcut=True),  # 128 x 128 x 32
            # PRNResBlock(in_channels=feature_size * 2, out_channels=feature_size * 2, kernel_size=3, stride=1, with_conv_shortcut=False),  # 128 x 128 x 32
            # PRNResBlock(in_channels=feature_size * 2, out_channels=feature_size * 4, kernel_size=3, stride=2, with_conv_shortcut=True),  # 64 x 64 x 64
            # PRNResBlock(in_channels=feature_size * 4, out_channels=feature_size * 4, kernel_size=3, stride=1, with_conv_shortcut=False),  # 64 x 64 x 64
            # PRNResBlock(in_channels=feature_size * 4, out_channels=feature_size * 8, kernel_size=3, stride=2, with_conv_shortcut=True),  # 32 x 32 x 128
            # PRNResBlock(in_channels=feature_size * 8, out_channels=feature_size * 8, kernel_size=3, stride=1, with_conv_shortcut=False),  # 32 x 32 x 128
            # PRNResBlock(in_channels=feature_size * 8, out_channels=feature_size * 16, kernel_size=3, stride=2, with_conv_shortcut=True),  # 16 x 16 x 256
            # PRNResBlock(in_channels=feature_size * 16, out_channels=feature_size * 16, kernel_size=3, stride=1, with_conv_shortcut=False),  # 16 x 16 x 256
            # PRNResBlock(in_channels=feature_size * 16, out_channels=feature_size * 32, kernel_size=3, stride=2, with_conv_shortcut=True),  # 8 x 8 x 512
            # PRNResBlock(in_channels=feature_size * 32, out_channels=feature_size * 32, kernel_size=3, stride=1, with_conv_shortcut=False),  # 8 x 8 x 512
       # )
        self.e2cnn = E2_ResNetFPN_8_2()  # 32*32*256
        self.topk = TopKWindowViT()  # 32*32*256->16*16*512->8*8*1024
        self.pos_encoding = PositionEncodingSine(  # 绝对位置编码
            _cfg['global_feature_dim'],  # suppose global_feature_dim equals to local_feature_dim
            temp_bug_fix=_cfg['temp_bug_fix'])
        self.global_transformer = GlobalFeatureTransformer(_cfg["gl_cfg"])
        self.decoder = nn.Sequential(
            ConvTranspose2d_BN_AC(in_channels=feature_size * 32, out_channels=feature_size * 32, kernel_size=4, stride=1),  # 8 x 8 x 512
            ConvTranspose2d_BN_AC(in_channels=feature_size * 32, out_channels=feature_size * 16, kernel_size=4, stride=2),  # 16 x 16 x 256
            ConvTranspose2d_BN_AC(in_channels=feature_size * 16, out_channels=feature_size * 16, kernel_size=4, stride=1),  # 16 x 16 x 256
            ConvTranspose2d_BN_AC(in_channels=feature_size * 16, out_channels=feature_size * 16, kernel_size=4, stride=1),  # 16 x 16 x 256
            ConvTranspose2d_BN_AC(in_channels=feature_size * 16, out_channels=feature_size * 8, kernel_size=4, stride=2),  # 32 x 32 x 128
            ConvTranspose2d_BN_AC(in_channels=feature_size * 8, out_channels=feature_size * 8, kernel_size=4, stride=1),  # 32 x 32 x 128
            ConvTranspose2d_BN_AC(in_channels=feature_size * 8, out_channels=feature_size * 8, kernel_size=4, stride=1),  # 32 x 32 x 128
            ConvTranspose2d_BN_AC(in_channels=feature_size * 8, out_channels=feature_size * 4, kernel_size=4, stride=2),  # 64 x 64 x 64
            ConvTranspose2d_BN_AC(in_channels=feature_size * 4, out_channels=feature_size * 4, kernel_size=4, stride=1),  # 64 x 64 x 64
            ConvTranspose2d_BN_AC(in_channels=feature_size * 4, out_channels=feature_size * 4, kernel_size=4, stride=1),  # 64 x 64 x 64
            ConvTranspose2d_BN_AC(in_channels=feature_size * 4, out_channels=feature_size * 2, kernel_size=4, stride=2),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 2, out_channels=feature_size * 2, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 2, out_channels=feature_size * 1, kernel_size=4, stride=2),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 1, out_channels=feature_size * 1, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 1, out_channels=3, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=3, out_channels=3, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=3, out_channels=3, kernel_size=4, stride=1, activation=nn.Tanh())
        )
        self.loss = InitLoss()

    def forward(self, inpt, gt):
        #x = self.layer0(inpt)#这里input还是256*256的，输出成255啦！是kernel size=4的锅，改成3OK
        #x = self.encoder(inpt)#
        se=self.e2cnn(inpt)
        se=se.tensor
        se=self.topk(se)
        se=self.attention_global_feature(se)

        x = self.decoder(se)
        loss, metrics = self.loss(x, gt)#loss就是论文里的loss，metrics就是评估指标45K个nme gt是UVpos图
        return loss, metrics, x

    def attention_global_feature(self,x):
        b,c,h,w = x.shape#batch/channel/h/w
        x = self.pos_encoding(x)#位置编码,实际使用绝对位置编码
        x = x.view(b,c,h*w).transpose(1,2)
        x = self.global_transformer(x)
        x = rearrange(x, 'b (h w) d -> b d h w', h=h, w=w)
        # x = x.mean(1)
        return x


class TorchNet:

    def __init__(self,
                 gpu_num=1,
                 visible_gpus='0',
                 learning_rate=1e-4
                 ):
        self.gpu_num = gpu_num
        gpus = visible_gpus.split(',')
        self.visible_devices = [int(i) for i in gpus]

        self.learning_rate = learning_rate
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.device = torch.device("cuda:" + gpus[0] if torch.cuda.is_available() else "cpu")


    def buildInitPRN(self):

        self.model = InitPRN2()

        if self.gpu_num > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.visible_devices)
        self.model.to(self.device)
        # model.cuda()

        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.learning_rate, weight_decay=0.0002)
        scheduler_exp = optim.lr_scheduler.ExponentialLR(self.optimizer, 0.8)
        self.scheduler = scheduler_exp
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.85)

    def loadWeights(self, model_path):
        if self.gpu_num > 1:
            self.model.module.load_state_dict(torch.load(model_path))  # , map_location=map_location))
        else:
            # you need to assign the same device to map_location
            self.model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
            # self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
