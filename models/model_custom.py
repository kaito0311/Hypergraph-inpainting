
import torch
import torch.nn as nn
import torch.nn.functional as F

# from .model import * 
from models.gc_layer import GatedConvolution, GatedDeConvolution
from .backbones.iresnet import iresnet160_wo_fc, iresnet160_gate, GatedBlockResnet
from .model import GatedBlock, GatedDeBlock

class CoarseModelResnet(torch.nn.Module): 
    def __init__(self, input_size = 256, channels = 64, downsample = 3):
        super().__init__() 
        self.downsample = downsample 

        assert downsample >= 4, 'Resnet have 4 downsample layer'

        # Encoder for Coarse Netowork 
        self.iresnet160_gate = iresnet160_gate()
        # TM-TODO: Add load pretrained 
        self.env_convs = nn.ModuleList() 

        in_channels = channels 

        self.env_convs.append(self.iresnet160_gate)
        in_channels = in_channels * 8
        for i in range(4, downsample):
            self.env_convs.append(
                GatedBlock(
                    in_channels= in_channels, 
                    out_channels= 2*in_channels, 
                    n_conv= 2, 
                    downscale_first= True,
                    dilation= 1,
                )
            )
            in_channels = 2 * in_channels 
        
        # Center Convolutions for higher receptive field
        # These convolutions are with dilation=2
        self.mid_convs = nn.ModuleList()
        for i in range(3):
            self.mid_convs.append(
                    GatedConvolution(in_channels=in_channels,
                                out_channels=in_channels,
                                kernel_size=3,
                                stride=1,
                                dilation=2,
                                padding='same',
                                activation='LeakyReLU')
                )
        
        # Decoder Network for Coarse Network
        self.dec_convs = nn.ModuleList()
        for i in range (self.downsample):
            if i > 0:
                self.dec_convs.append(GatedDeBlock(
                                in_channels = 2*in_channels,  # Skip connection from Encoder
                                out_channels = int(in_channels//2),
                                n_conv = 2,
                            ))
            else:
                self.dec_convs.append(GatedDeBlock(
                                in_channels = in_channels,
                                out_channels = int(in_channels//2),
                                n_conv = 2,
                            ))

            in_channels = int(in_channels//2)

        self.last_dec   = GatedConvolution(in_channels=in_channels,
                                out_channels=channels,
                                kernel_size=3,
                                stride=1,
                                dilation=1,
                                padding='same',
                                activation='LeakyReLU')

        self.coarse_out = GatedConvolution(in_channels=channels,
                                out_channels=3,
                                kernel_size=3,
                                stride=1,
                                dilation=1,
                                padding='same',
                                activation=None)
        
    def forward(self, x):  
        skip_layer = []

        output_resnet_gate = self.env_convs[0](x) 

        x_embed = output_resnet_gate[0] 
        x_scale = output_resnet_gate[1:] 
        x_scale = list(x_scale)

        x_return = x_scale[-1]
        for i in x_scale: 
            print("x encode shape ", i.shape)
        for i in range(4, self.downsample):
            # print(self.env_convs[i-3])
            x_return = self.env_convs[i - 3](x_return)
            print("x encode shape ", x_return.shape)
            x_scale.append(x_return)
        
        # x_scale= skip_layer
        skip_layer = x_scale 
        for i in range(3): 
            x_return = self.mid_convs[i](x_return)

        for i in range(self.downsample):
            if i > 0:
                skip_layer_idx = self.downsample - 1 - i
                print(skip_layer_idx)
                x_return = torch.cat([x_return, skip_layer[skip_layer_idx]], dim = 1)
            print("x decode shape ", x_return.shape)
            x_return = self.dec_convs[i](x_return)
        
        print("x_return.shape ", x_return.shape)

        x_return = self.last_dec(x_return)
        print("last_dec shape ", x_return.shape)
        
        x_return = self.coarse_out(x_return)

        
        return x_scale 
             

            

        


    