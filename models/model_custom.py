
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
        for i in range(4, self.downsample):
            # print(self.env_convs[i-3])
            x_return = self.env_convs[i - 3](x_return)
            x_scale.append(x_return)
        
        # x_scale= skip_layer
        skip_layer = x_scale 
        for i in range(3): 
            x_return = self.mid_convs[i](x_return)

        for i in range(self.downsample):
            if i > 0:
                skip_layer_idx = self.downsample - 1 - i
                x_return = torch.cat([x_return, skip_layer[skip_layer_idx]], dim = 1)
            x_return = self.dec_convs[i](x_return)

        x_return = self.last_dec(x_return)        
        x_return = self.coarse_out(x_return)

        return x_return
             

            
class HyperGraphModelCustom(torch.nn.Module):
    def __init__(self,input_size=256, coarse_downsample = 5, refine_downsample= 6, channels = 64, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.coarse_model = CoarseModelResnet(input_size= input_size, channels= channels, downsample= coarse_downsample)
        self.refine_model = CoarseModelResnet(input_size= input_size, channels= channels, downsample= refine_downsample)
    
    def forward(self, img, mask): 
        # mask: 0 - original image, 1.0 - masked
        inp_coarse = torch.cat([img, mask], dim = 1)

        out_coarse = self.coarse_model(inp_coarse)
        out_coarse = torch.clamp(out_coarse, min = 0.0, max = 1.0)
        b, _, h, w = mask.size()
        mask_rp = mask.repeat(1, 3, 1, 1)
        inp_refine = out_coarse * mask_rp + img * (1.0 - mask_rp)
        inp_refine = torch.cat([inp_refine, mask], dim = 1)
        out_refine = self.refine_model(inp_refine)
        out_refine = torch.clamp(out_refine, min = 0.0, max = 1.0)
        return out_coarse, out_refine