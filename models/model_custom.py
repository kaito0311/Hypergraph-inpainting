
import torch
import torch.nn as nn
import torch.nn.functional as F

# from .model import * 
from models.gc_layer import GatedConvolution, GatedDeConvolution
from .backbones.iresnet import iresnet160_wo_fc


class CoarseModelResnet(torch.nn.Module): 
    def __init__(self, input_size = 256, channels = 64, downsample = 3):
        super().__init__() 
        self.downsample = downsample 

        self.conv1 = GatedConvolution(in_channels=4, out_channels=channels, kernel_size=7, stride=1, dilation=1, padding='same', activation='LeakyReLU') # RGB + Mask

        # Encoder 
        self.encoder_resnet = iresnet160_wo_fc()
        try: 
            self.encoder_resnet.load_state_dict(torch.load("pretrained/r160_imintv4_statedict_wo_fc.pth"))
        except: 
            raise ValueError("Cannot load pretrain model from pretrained/r160_imintv4_statedict_wo_fc.pth")
        
        self.enc_convs = nn.ModuleList()
        in_channels = channels

        for i in range(self.downsample): 
            in_channels = 2*in_channels

            self.enc_convs.append(GatedBlockResnet(
                in_channels = in_channels,
                out_channels = in_channels,
                n_conv = 2,
                dilation = 1,
                downsample= False if i < 4 else True
            ))
    
    def forward(self, x): 
        x = self.conv1(x) 

        skip_layer = [] 
        outputs_resnet = self.encoder_resnet(x) # x_embed, x_d2, x_d4, x_d8, x_d16 
        x_embed = outputs_resnet[0] 
        x_downsample = outputs_resnet[1:]


        for i in range(self.downsample):
            if i < 4: 
                x 
                
            else: 
                x = self.enc_convs[i](x) 
            

        


    