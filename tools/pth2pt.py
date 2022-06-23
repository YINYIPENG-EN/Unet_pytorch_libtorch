import torch
from nets.unet import Unet

model = Unet(num_classes=21, in_channels=3)
pre_trained = torch.load('../model_data/unet_voc.pth')
model.load_state_dict(pre_trained)
model.eval()
input = torch.ones(1, 3, 512, 512)
trace_model = torch.jit.trace(model, input)
print(trace_model)
trace_model.save('../model_data/unet.pt')


