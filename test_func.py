from utils import *
from models.backbones.iresnet import iresnet160, iresnet160_wo_fc, iresnet160_gate
import torch 

from collections import OrderedDict



''' ==== TEST OUTPUT ENCODER HYPERGRAPH ORIGINAL === ''' 



'''==== TEST LOAD ENCODER RESNET 160 COMBINE GATE BLOCK === ''' 
# === convert state dict model and run check ===  
iresnet160_gate_model = iresnet160_gate() 
dummy_input = torch.randn(1, 3, 112, 112).to("cuda")
outputs = iresnet160_gate_model(dummy_input)
for out in outputs: 
    print(out.shape)

iresnet160_model = iresnet160()
iresnet160_model.eval()
outputs2 = iresnet160_model(dummy_input)
for out in outputs2: 
    print(out.shape)
  
# print(iresnet160_gate_model)

# === convert state dict model and run check ===  
# model = iresnet160_gate() 
# # print(model.state_dict().keys())
# state_dict_original = torch.load("pretrained/r160_imintv4_statedict.pth")
# target_dict = OrderedDict() 
# for key in model.state_dict().keys(): 
#     temp = key[len("resnet_component."):]
#     if temp not in state_dict_original.keys():
#         continue 
#     else: 
#         target_dict[key] = state_dict_original[temp] 

# torch.save(model.state_dict(), "pretrained/r160_imintv4_statedict_wo_fc_gate_combine.pth")
# model.load_state_dict(torch.load("pretrained/r160_imintv4_statedict_wo_fc_gate_combine.pth"))
# model.eval() 

# dummy_input = torch.randn(1, 3, 112, 112).to("cuda")
# outputs = model(dummy_input)


# iresnet160_model = iresnet160()
# iresnet160_model.load_state_dict(torch.load("pretrained/r160_imintv4_statedict.pth"))
# iresnet160_model.eval()
# outputs2 = iresnet160_model(dummy_input)

# print(torch.sum(outputs[2] - outputs2[2]))


'''====== TEST LOAD ENCODER RESNET160 AND SAVE STATE DICT === '''

# === load model and run only ===  
# iresnet160_wo_fc_model = iresnet160_wo_fc()
# iresnet160_wo_fc_model.load_state_dict(torch.load("pretrained/r160_imintv4_statedict_wo_fc.pth"))
# dummy_input = torch.randn(1, 3, 256, 256).to("cuda")
# outputs = iresnet160_wo_fc_model(dummy_input)

# # for out in outputs: 
# #     print(out.shape)

# for key in iresnet160_wo_fc_model.state_dict().keys(): 
#     print(key)


# === load state dict and remove fc ==== 


# iresnet160_wo_fc_model = iresnet160_wo_fc()

# state_dict = iresnet160_wo_fc_model.state_dict() 

# origin_dict = torch.load("pretrained/r160_imintv4_statedict.pth")
# target_dict = OrderedDict() 
# iresnet160_wo_fc_model.eval()

# for key in origin_dict.keys(): 
#     if key not in state_dict.keys(): 
#         continue 
#     else: 
#         target_dict[key] = origin_dict[key]


# torch.save(iresnet160_wo_fc_model.state_dict(), "pretrained/r160_imintv4_statedict_wo_fc.pth")

# iresnet160_wo_fc_model.load_state_dict(target_dict)
# dummy_input = torch.randn(1, 3, 112, 112).to("cuda")
# outputs = iresnet160_wo_fc_model(dummy_input)


# iresnet160_model = iresnet160()
# iresnet160_model.load_state_dict(origin_dict)
# iresnet160_model.eval()
# outputs2 = iresnet160_model(dummy_input)

# print(torch.sum(outputs[2] - outputs2[2]))