import torch
from torch import nn
from models.gc_layer import GatedConvolution, GatedDeConvolution


__all__ = ['iresnet18', 'iresnet34', 'iresnet50', 'iresnet100', 'iresnet124', 'iresnet200']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)

class ConvBNBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, activation = 'PReLU'):
        super(ConvBNBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05,)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05,)
        # self.prelu = nn.PReLU(planes)
        if activation == 'PReLU':
            self.act = nn.PReLU(planes)
        elif activation == 'LeakyReLU':
            self.act = nn.LeakyReLU(negative_slope=0.2, inplace=False)
        elif activation is None:
            self.act = None
        else:
            raise NotImplementedError("Not support activation {}".format(activation))

    def forward(self, x):
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        if self.act is not None:
            x = self.act(x)
        return x
    
class IBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, activation = 'PReLU'):
        super(IBasicBlock, self).__init__()
        assert activation is not None
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05,)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05,)
        if activation == 'PReLU':
            self.prelu = nn.PReLU(planes)
        elif activation == 'LeakyReLU':
            self.prelu = nn.LeakyReLU(negative_slope=0.2, inplace=False)
        else:
            raise NotImplementedError("Not support activation {}".format(activation))
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        # print(identity.)
        out += identity
        return out

class IBasicDecodeBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, activation = 'PReLU'):
        super(IBasicDecodeBlock, self).__init__()
        assert activation is not None
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05,)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05,)
        if activation == 'PReLU':
            self.prelu1 = nn.PReLU(planes)
            self.prelu2 = nn.PReLU(planes)
        elif activation == 'LeakyReLU':
            self.prelu1 = nn.LeakyReLU(negative_slope=0.2, inplace=False)
            self.prelu2 = nn.LeakyReLU(negative_slope=0.2, inplace=False)
        else:
            raise NotImplementedError("Not support activation {}".format(activation))
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu1(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        # print(identity.)
        out += identity
        out = self.prelu2(out)
        return out


def make_upsample_layer(inplanes, planes, kernel_size = 3, activation = None):
    assert activation is not None
    convbn = nn.ConvTranspose2d(inplanes, planes, kernel_size, stride = 2, padding = 1)
    if activation == 'PReLU':
        act = nn.PReLU(planes)
    elif activation == 'LeakyReLU':
        act = nn.LeakyReLU(negative_slope=0.2, inplace=False)
    else:
        raise NotImplementedError("Not support activation {}".format(activation))
    layers = [convbn, act]
    return nn.Sequential(*layers)

def make_decoder_layer(inplanes, planes, n_blocks = 1, activation = 'PReLU'):
    assert activation is not None
    if inplanes != planes:
        first_convbn = ConvBNBlock(inplanes = inplanes, planes = planes, activation = activation)
        layers = [first_convbn]
    else:
        layers = []
    for i in range(n_blocks):
        layers.append(
            IBasicDecodeBlock(inplanes = planes, planes = planes, activation = activation))
    return nn.Sequential(*layers)

class IResNet(nn.Module):
    fc_scale = 7 * 7
    def __init__(self,
                 block, layers, dropout=0, num_features=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, fp16=False):
        super(IResNet, self).__init__()
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05,)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05, ),
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.prelu(x)
            x_56 = self.layer1(x)
            x_28 = self.layer2(x_56)
            x_14 = self.layer3(x_28)
            x_7 = self.layer4(x_14)
            x = self.bn2(x_7)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
        x = self.fc(x.float() if self.fp16 else x)
        x = self.features(x)
        return x, x_56, x_28, x_14, x_7


class ResNetComponent(IResNet):
    def __init__(self, block, layers, dropout=0, num_features=512, zero_init_residual=False,
                groups=1, width_per_group=64, replace_stride_with_dilation=None, fp16=False) -> None:
        super(IResNet, self).__init__()
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                            "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block,
                                    128,
                                    layers[1],
                                    stride=2,
                                    dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block,
                                    256,
                                    layers[2],
                                    stride=2,
                                    dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block,
                                    512,
                                    layers[3],
                                    stride=2,
                                    dilate=replace_stride_with_dilation[2])
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05,)
    def forward(self, x):
        pass 


class GatedBlockResnet(torch.nn.Module):
    '''
    Input forward func: Gateconv for generator, input is downsample feature map that produced by Resnet or sth
    '''
    def __init__(self,
                in_channels = 64,
                out_channels = 128,
                n_conv = 2,
                dilation = 1,
                activation = 'LeakyReLU', 
                downsample = False,
                ):
        super().__init__() 
        self.in_channels = in_channels 
        self.out_channels = out_channels
        self.n_conv = n_conv
        self.need_downsample = downsample 

        self.rest_conv = nn.ModuleList()

        if downsample : 
            self.gate_downsample = GatedConvolution(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=3,
                                            stride=2,
                                            dilation=1,
                                            padding='same',
                                            activation=activation)
        else: 
            self.gate_downsample = torch.nn.Identity() 


        for i in range(n_conv):
            self.rest_conv.append(
                GatedConvolution(in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=3,
                                stride=1,
                                dilation=dilation,
                                padding='same',
                                activation=activation)
            )
    
    def forward(self, feature_map_down_resnet): 
        '''
        '''
        if self.need_downsample: 
            x = self.gate_downsample(x) 
        for i in range(self.n_conv):
            x = self.rest_conv[i](feature_map_down_resnet) 
        return x 


class IResNetGateBlock(nn.Module): 
    fc_scale = 7 * 7
    
    def __init__(self,
                 block, layers, gate_channels, dropout=0, num_features=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, fp16=False):
        super(IResNetGateBlock, self).__init__()
        self.fp16 = fp16
        self.resnet_component = ResNetComponent(block, layers, dropout, num_features, zero_init_residual,
                 groups, width_per_group, replace_stride_with_dilation, fp16)
        

        in_channels = gate_channels
        
        self.conv1 = GatedConvolution(in_channels=4, out_channels=in_channels, kernel_size=7, stride=1, dilation=1, padding='same', activation='LeakyReLU') # RGB + Mask
        self.gate_blocks = nn.ModuleList() 
        # for each layer in resnet component, have a gate block respectively 

        for i in range(4): # because Resnet 160 have 4 layer 
            self.gate_blocks.append(
                GatedBlockResnet(
                    in_channels= in_channels, 
                    out_channels= in_channels, # just gen 
                    n_conv= 2, 
                    dilation= 1, 
                    downsample= False
                )
            )
            in_channels = 2 * in_channels 

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            x = self.resnet_component.bn1(x)
            x = self.resnet_component.prelu(x)
            x_56 = self.resnet_component.layer1(x)
            x_56 = self.gate_blocks[0](x_56)
            x_28 = self.resnet_component.layer2(x_56)
            x_28 = self.gate_blocks[1](x_28)
            x_14 = self.resnet_component.layer3(x_28)
            x_14 = self.gate_blocks[2](x_14)
            x_7 = self.resnet_component.layer4(x_14)
            x_7 = self.gate_blocks[3](x_7)
            x = self.resnet_component.bn2(x_7) # TM-FIXME : need before x_7? 
            x = torch.flatten(x, 1)
        return x, x_56, x_28, x_14, x_7


class IResNet_wo_fc(IResNet): 
    fc_scale = 7 * 7
    def __init__(self,
                 block, layers, dropout=0, num_features=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, fp16=False):
        super(IResNet, self).__init__()
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05,)
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.prelu(x)
            x_56 = self.layer1(x)
            x_28 = self.layer2(x_56)
            x_14 = self.layer3(x_28)
            x_7 = self.layer4(x_14)
            x = self.bn2(x_7)
            x = torch.flatten(x, 1)
        return x, x_56, x_28, x_14, x_7


def _iresnet(arch, block, layers, pretrained, progress, **kwargs):
    model = IResNet(block, layers, **kwargs).cuda()
    if pretrained:
        raise ValueError()
    return model


def iresnet18(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet18', IBasicBlock, [2, 2, 2, 2], pretrained,
                    progress, **kwargs)


def iresnet34(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet34', IBasicBlock, [3, 4, 6, 3], pretrained,
                    progress, **kwargs)


def iresnet50(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet50', IBasicBlock, [3, 4, 14, 3], pretrained,
                    progress, **kwargs)


def iresnet100(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet100', IBasicBlock, [3, 13, 30, 3], pretrained,
                    progress, **kwargs)

def iresnet124(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet124', IBasicBlock, [3, 13, 40, 5], pretrained,
                    progress, **kwargs)

def iresnet160(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet160', IBasicBlock, [3, 24, 49, 3], pretrained,
                    progress, **kwargs)

def iresnet200(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet200', IBasicBlock, [6, 26, 60, 6], pretrained,
                    progress, **kwargs)

def iresnet160_wo_fc(pretrained=False, progress=True, **kwargs):
    model = IResNet_wo_fc(IBasicBlock, [3, 24, 49, 3], **kwargs).cuda()
    return model 

def iresnet160_gate(**kwargs):
    model = IResNetGateBlock(IBasicBlock, [3, 24, 49, 3],gate_channels= 64, **kwargs).cuda()
    return model 