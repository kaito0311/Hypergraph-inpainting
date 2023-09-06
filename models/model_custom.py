
import torch
import torch.nn as nn
import torch.nn.functional as F

# from .model import * 
from models.gc_layer import GatedConvolution, GatedDeConvolution, GatedConvolutionOperator
from .backbones.iresnet import iresnet160_wo_fc, iresnet160_gate, GatedBlockResnet, iresnet18_wo_fc
from .model import GatedBlock, GatedDeBlock


class CoarseModelDoubleResnet(torch.nn.Module): 
    '''
    Encoder architecture: 
    [image] -> pretrained_resnet160     ->| downsample_1 |-> downsample_2 -> ... 
    [mask]  -> custom_resnet_less_layer ->| downsample_1 |-> downsample_2 -> ... 
                                          |              |
                                              fuse_like_gc
                                                   |
                                                output downsample 1 

    '''
    def __init__(self, input_size = 256, channels = 64, downsample = 3, 
                 activation='ELU', batch_norm=False, negative_slope = 0.2, bias = True,): 
        super().__init__() 
        self.downsample = downsample 

        assert downsample >= 4, 'Resnet have 4 downsample layer'

        self.env_image_conv = nn.ModuleList() 
        self.env_mask_conv = nn.ModuleList() 
        self.list_gate_operator : list[GatedConvolutionOperator] = nn.ModuleList() 
        self.env_fuse_convs: list[GatedBlock] = nn.ModuleList() # 
        # TM-TODO: ADD gate convolution for extra downsample 
        self.extra_env_conv = nn.ModuleList() # both image + mask fuse 
        

        in_channels = channels

        self.env_image_conv.append(iresnet160_wo_fc()) 
        self.env_mask_conv.append(iresnet18_wo_fc())

        

        for i in range(downsample): 
            self.list_gate_operator.append(
                GatedConvolutionOperator(
                    activation= 'LeakyReLU',
                )
            )
            if i < 4: # only for output of resnet layer 
                self.env_fuse_convs.append(
                    GatedBlock(
                        in_channels= in_channels,
                        out_channels= in_channels, 
                        n_conv= 0,
                        downscale_first= False, 
                        dilation= 1
                    )
                )
            in_channels = in_channels * 2 
        
        in_channels = channels * 8 
        for i in range(4, downsample): 
            self.extra_env_conv.append(
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



    
    def forward(self, image, mask): 

        if mask.shape[1] != 3: 
            mask = mask.repeat(1, 3,1,1)


        
        # Encoder 

        # Take feature maps by using resnet backbone
        output_embed_image = self.env_image_conv[0](image) # x_embed, x_down_1, x_down_2, x_down_3, x_down_4 
        output_embed_mask = self.env_mask_conv[0](mask) 

        ls_feature_map_image = list(output_embed_image[1:]) # shape (batch_size, channels, height, width)
        ls_feature_map_mask = list(output_embed_mask[1:]) 
        skip_layer = [] 
        x_return = None 
        # Fuse feature maps imgage and feature map mask together to create output of encoder
        for index in range(self.downsample):
            if index < 4: 
                feature_image = ls_feature_map_image[index]
                feature_mask = ls_feature_map_mask[index] 
                # print("Featrue image shape: ", feature_image.shape)
                # print('feature_mask shape', feature_mask.shape)
               
                if feature_mask.shape[1] == 3: 
                    if index != self.downsample -1 : 
                        skip_layer.append(feature_image) 
                else: 
                    x_return = (self.list_gate_operator[index](feature_image= feature_image, feature_mask = feature_mask)) 
                    x_return = self.env_fuse_convs[index](x_return)
                    # print("x output: ", x_return.shape)
                    if index != self.downsample -1 : 
                        skip_layer.append(x_return)
            else:
                x_return = self.extra_env_conv[index - 4](x_return) 
                # print("x extra output: ", x_return.shape)
                if index != self.downsample - 1: 
                    skip_layer.append(x_return) 
        
        for i in range(3): 
            x_return = self.mid_convs[i](x_return) 
        
        for i in range(self.downsample):
            if i > 0: 
                skip_layer_idx = self.downsample - 1 - i 
                assert skip_layer_idx >= 0, 'skip layer idx must be >= 0'
                x_return = torch.cat([x_return, skip_layer[skip_layer_idx]], dim= 1) 
            x_return = self.dec_convs[i](x_return)
            # print("x output decode: ", x_return.shape)
        x_return = self.last_dec(x_return)
        x_return = self.coarse_out(x_return)
        # print("final x: ", x_return.shape)
        
        return x_return
    

class CoarseModelResnet(torch.nn.Module): 
    def __init__(self, input_size = 256, channels = 64, downsample = 3):
        super().__init__() 
        self.downsample = downsample 

        assert downsample >= 4, 'Resnet have 4 downsample layer'

        # Encoder for Coarse Netowork 
        # self.iresnet160_gate = iresnet160_gate()
        # TM-TODO: Add load pretrained 
        self.env_convs = nn.ModuleList() 

        in_channels = channels 

        self.env_convs.append(iresnet160_gate())
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

        # self.coarse_model = CoarseModelResnet(input_size= input_size, channels= channels, downsample= coarse_downsample)
        # self.refine_model = CoarseModelResnet(input_size= input_size, channels= channels, downsample= refine_downsample)
        self.coarse_model = CoarseModelDoubleResnet(input_size= input_size, channels= channels, downsample= coarse_downsample)
        self.refine_model = CoarseModelDoubleResnet(input_size= input_size, channels= channels, downsample= refine_downsample)
    
    def forward(self, img, mask): 
        # mask: 0 - original image, 1.0 - masked
        out_coarse = self.coarse_model(img, mask)
        out_coarse = torch.clamp(out_coarse, min = 0.0, max = 1.0)
        b, _, h, w = mask.size()
        mask_rp = mask.repeat(1, 3, 1, 1)
        inp_refine = out_coarse * mask_rp + img * (1.0 - mask_rp)
        out_refine = self.refine_model(inp_refine, mask)
        out_refine = torch.clamp(out_refine, min = 0.0, max = 1.0)
        return out_coarse, out_refine