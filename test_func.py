from utils import *
from models.backbones.iresnet import iresnet160, iresnet160_wo_fc
import torch 

from collections import OrderedDict


iresnet160_wo_fc_model = iresnet160_wo_fc()

state_dict = iresnet160_wo_fc_model.state_dict() 

origin_dict = torch.load("pretrained/r160_imintv4_statedict.pth")
target_dict = OrderedDict() 
iresnet160_wo_fc_model.eval()

for key in origin_dict.keys(): 
    if key not in state_dict.keys(): 
        continue 
    else: 
        target_dict[key] = origin_dict[key]


torch.save(iresnet160_wo_fc_model.state_dict(), "pretrained/r160_imintv4_statedict_wo_fc.pth")

iresnet160_wo_fc_model.load_state_dict(target_dict)
dummy_input = torch.randn(1, 3, 112, 112).to("cuda")
outputs = iresnet160_wo_fc_model(dummy_input)


iresnet160_model = iresnet160()
iresnet160_model.load_state_dict(origin_dict)
iresnet160_model.eval()
outputs2 = iresnet160_model(dummy_input)

print(torch.sum(outputs[2] - outputs2[2]))