from utils import *
from models.model_custom import CoarseModelResnet, HyperGraphModelCustom, CoarseModelDoubleResnet

from models.backbones.iresnet import iresnet160, iresnet160_wo_fc, iresnet160_gate
import torch 
import torchvision
from collections import OrderedDict
import os
import cv2 

'''Take all image test '''
path_dir = "experiments/visualize/1584000"
start = 0
for i in range(start, len(os.listdir(path_dir)), 15):
    ls_image = []
    for image_name in os.listdir(path_dir)[start: start+15]:
        image = cv2.imread(os.path.join(path_dir, image_name))
        assert image is not None 
        ls_image.append(image)

    image_all = np.concatenate(ls_image, axis=0)

    image_all = cv2.resize(image_all, (1536, 2048))
    print(image_all.shape)

    cv2.imwrite(f"image_all{start}.png", image_all)
    start += 15






# model = HyperGraphModelCustom(input_size = 256, coarse_downsample = 4, refine_downsample = 5, channels = 64) 
# dict_smooth = torch.load("ckpt/hyper_graph_smooth_pretrain.pt")
# model.load_state_dict(dict_smooth)




'''==== Take parameter pretrain'''
# dict_no_smooth =torch.load("ckpt/ckpt_gen_916k.pt")
# dict_smooth = torch.load("ckpt/hyper_graph_smooth.pt")

# for key in dict_no_smooth.keys():
#     if key not in dict_smooth.keys():
#         print(key) 
#     else:
#         if "coarse_out" in key: 
#             new_key = str(key).replace("coarse_out","coarse_smooth")
#             if new_key not in dict_smooth.keys():
#                 print("new key ", new_key, " not true") 
#             else:
#                 dict_smooth[new_key] = dict_no_smooth[key]
#         else:
#             if key in dict_smooth.keys():
#                 dict_smooth[key] = dict_no_smooth[key] 
#             else:
#                 print(key, " not in dict smooth")

# torch.save(dict_smooth, 'hyper_graph_smooth_pretrain.pt')



'''==== create check point '''

# model = HyperGraphModelCustom(input_size = 256, coarse_downsample = 4, refine_downsample = 5, channels = 64) 
# torch.save(model.state_dict(), 'hyper_graph_smooth.pt')


'''=== CHECK SMOOTH CONV ==='''

# HyperGraphModelCustom(input_size = 256, coarse_downsample = 4, refine_downsample = 5, channels = 64) 

# coarse_model = CoarseModelDoubleResnet(downsample= 4)
# coarse_model.to("cuda")

# dummy_image, dummy_mask = torch.randn(1, 3, 256, 256).to("cuda"), torch.randn(1, 1, 256, 256).to("cuda") 

# output = coarse_model(dummy_image, dummy_mask)

# print(output.shape)


'''=== CHECK SAVE BACKUP FILE '''

# a = torch.nn.Conv2d(
#     in_channels= 3, 
#     out_channels= 3, 
#     kernel_size= 3, 
#     stride= 1, 
#     padding= 1
# )

# input_dummy = torch.randn(4, 3, 256, 256)
# output = a(input_dummy)
# print(output.shape)


'''===  FEATURE LOSS VGG16'''

# dummy_input1 = torch.randn(2, 3, 256, 256)
# dummy_input2 = torch.randn(2, 3, 256, 256)

# vgg16 = torchvision.models.vgg16(pretrained=True) 
# vgg16.eval() 
# feature1 = vgg16(dummy_input1)
# feature2 = vgg16(dummy_input2)
# print(feature1.shape)
# loss = torch.nn.functional.mse_loss(feature1, feature2)
# print(loss)


'''=== CHECK FREEZE BACKBONE ====='''

# import torch
# state_dict_hyper = torch.load("experiments/ckpt/ckpt_20.pt")
# state_dict_hyper_2 = torch.load("ckpt/hyper_graph_custom_pretrained_resnet.pt")

# state_dict_resnet = torch.load("ckpt/r160_imintv4_statedict.pth")
# for key in state_dict_resnet.keys(): 
#     new_key = "refine_model.env_image_conv.0." + key 
#     if str(key).endswith("running_mean"): 
#         continue 
#     if str(key).endswith("running_var"): 
#         continue 
#     if str(key).endswith("num_batches_tracked"): 
#         continue 
#     if new_key not in state_dict_hyper.keys():
#         print(new_key)
#     else:
#         # assert torch.sum(state_dict_hyper[new_key] - state_dict_hyper_2[new_key]) == 0, torch.sum(state_dict_hyper[new_key] - state_dict_hyper_2[new_key])
#         # print(state_dict_hyper_2[new_key])
#         # print(new_key)
#         assert torch.sum(state_dict_hyper[new_key] - state_dict_resnet[key]) == 0, (torch.sum(state_dict_hyper[new_key] - state_dict_resnet[key]))

#     new_key = "coarse_model.env_image_conv.0." + key 
#     if new_key not in state_dict_hyper.keys():
#         print(new_key)
#     else: 
#         assert  torch.sum(state_dict_hyper[new_key] - state_dict_resnet[key]) == 0 


'''=== CREATE CHECKPOINT WITH PRETRAINED IRESNET for COARSE MODEL DOUBLE RESNET ==='''

# import torch
# state_dict_hyper = torch.load("experiments/ckpt/ckpt_20.pt")
# state_dict_resnet = torch.load("ckpt/r160_imintv4_statedict.pth")

# for key in state_dict_resnet.keys():
#     new_key = "refine_model.env_image_conv.0." + key
#     if new_key not in state_dict_hyper.keys():
#         print(new_key)
#     else:
#         state_dict_hyper[new_key] = state_dict_resnet[key]
#     new_key = "coarse_model.env_image_conv.0." + key
#     if new_key not in state_dict_hyper.keys():
#         print(new_key)
#     else:
#         state_dict_hyper[new_key] = state_dict_resnet[key]

# hyper_graph_custom = HyperGraphModelCustom(
#     input_size= 256, 
#     coarse_downsample= 4, 
#     refine_downsample= 5, 
#     channels= 64
# )
# hyper_graph_custom.eval()

# hyper_graph_custom.load_state_dict(state_dict_hyper)
# torch.save(hyper_graph_custom.state_dict(), "ckpt/hyper_graph_custom_pretrained_resnet.pt")


'''==== TEST DOUBLE RESNET COARSE MODEL '''

# # model_coarse = CoarseModelDoubleResnet(downsample= 5) 
# model_coarse = HyperGraphModelCustom(input_size = 256, coarse_downsample = 4, refine_downsample = 4, channels = 64)
# model_coarse.to("cuda")
# dummy_image, dummy_mask = torch.randn(1, 3, 256, 256).to("cuda"), torch.randn(1, 1, 256, 256).to("cuda") 

# x_return = model_coarse(dummy_image, dummy_mask)

# while(True):
#     pass 




'''=== CREATE CHECKPOINT WITH PRETRAINED IRESNET ==='''
# import torch
# state_dict_hyper = torch.load("experiments/ckpt/ckpt_20.pt")
# state_dict_resnet = torch.load("ckpt/r160_imintv4_statedict.pth")

# for key in state_dict_resnet.keys():
#     new_key = "refine_model.env_convs.0.resnet_component." + key
#     if new_key not in state_dict_hyper.keys():
#         print(new_key)
#     else:
#         state_dict_hyper[new_key] = state_dict_resnet[key]
#     new_key = "coarse_model.env_convs.0.resnet_component." + key
#     if new_key not in state_dict_hyper.keys():
#         print(new_key)
#     else:
#         state_dict_hyper[new_key] = state_dict_resnet[key]



# hyper_graph_custom = HyperGraphModelCustom(
#     input_size= 256, 
#     coarse_downsample= 4, 
#     refine_downsample= 5, 
#     channels= 64
# )
# hyper_graph_custom.eval()

# hyper_graph_custom.load_state_dict(state_dict_hyper)
# torch.save(hyper_graph_custom.state_dict(), "ckpt/hyper_graph_custom_pretrained_resnet.pt")

# hyper_graph_custom.load_state_dict(torch.load("ckpt/hyper_graph_custom_pretrained_resnet.pt"))


'''=== CHECK FREEZE RESNET COMPONENT === '''

# hyper_graph_custom = HyperGraphModelCustom(
#     input_size= 256, 
#     coarse_downsample= 4, 
#     refine_downsample= 5, 
#     channels= 64
# )
# hyper_graph_custom.train() 
# hyper_graph_custom.coarse_model.env_convs[0].resnet_component.requires_grad_(False) 
# hyper_graph_custom.refine_model.env_convs[0].resnet_component.requires_grad_(False)
# num_parameters = 0
# for parameter in hyper_graph_custom.parameters():
#     if parameter.requires_grad:
#         num_parameters += parameter.numel()

# print(num_parameters)



# print(hyper_graph_custom.coarse_model.env_convs[0])

# import torch
# state_dict_hyper = torch.load("experiments/ckpt/ckpt_20.pt")
# state_dict_hyper_2 = torch.load("ckpt/hyper_graph_custom_pretrained_resnet.pt")

# state_dict_resnet = torch.load("ckpt/r160_imintv4_statedict.pth")
# for key in state_dict_resnet.keys(): 
#     new_key = "refine_model.env_convs.0.resnet_component." + key 
#     if str(key).endswith("running_mean"): 
#         continue 
#     if str(key).endswith("running_var"): 
#         continue 
#     if str(key).endswith("num_batches_tracked"): 
#         continue 
#     if new_key not in state_dict_hyper.keys():
#         print(new_key)
#     else:
#         # assert torch.sum(state_dict_hyper[new_key] - state_dict_hyper_2[new_key]) == 0, torch.sum(state_dict_hyper[new_key] - state_dict_hyper_2[new_key])
#         # print(state_dict_hyper_2[new_key])
#         # print(new_key)
#         assert torch.sum(state_dict_hyper[new_key] - state_dict_resnet[key]) == 0, (torch.sum(state_dict_hyper[new_key] - state_dict_resnet[key]))

#     new_key = "coarse_model.env_convs.0.resnet_component." + key 
#     if new_key not in state_dict_hyper.keys():
#         print(new_key)
#     else: 
#         assert  torch.sum(state_dict_hyper[new_key] - state_dict_resnet[key]) == 0 





'''=== TEST OUTPUT HYPERGRAPHMODELCUSTOM MODEL ==='''
# hyper_graph_custom = HyperGraphModelCustom(
#     input_size= 256, 
#     coarse_downsample= 4, 
#     refine_downsample= 5, 
#     channels= 64
# )
# hyper_graph_custom.eval()

# dummy_image, dummy_mask = torch.randn(1, 3, 256, 256).to("cuda"), torch.randn(1, 1, 256, 256).to("cuda") 
# hyper_graph_custom.to('cuda')

# hyper_graph_custom(dummy_image, dummy_mask)

# torch.save(hyper_graph_custom.state_dict(), "ckpt/hyper_graph_custom.pt")

'''=== TEST OUTPUT ENCODER CUSTOM COARSE MODEL ==='''
# coarse_model_resnet = CoarseModelResnet(downsample= 5, channels= 64)
# coarse_model_resnet.to("cuda")
# dummy_input = torch.randn(1, 4, 256, 256).to("cuda")

# outputs = coarse_model_resnet(dummy_input)


''' ==== TEST OUTPUT ENCODER HYPERGRAPH ORIGINAL === ''' 



'''==== TEST LOAD ENCODER RESNET 160 COMBINE GATE BLOCK === ''' 
# === convert state dict model and run check ===  
# iresnet160_gate_model = iresnet160_gate() 
# outputs = iresnet160_gate_model(dummy_input)
# for out in outputs: 
#     print(out.shape)
# print(iresnet160_gate_model)

# print(coarse_model_resnet)

# iresnet160_model = iresnet160()
# iresnet160_model.eval()
# dummy_input = torch.randn(1, 3, 112, 112).to("cuda")
# outputs2 = iresnet160_model(dummy_input)
# for out in outputs2: 
#     print(out.shape)
  
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
# iresnet160_model