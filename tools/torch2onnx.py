import torch
from nets.unet import Unet

model = Unet(num_classes=21, in_channels=3)
pre_trained = torch.load(r'../model_data/unet_voc.pth')
model.load_state_dict(pre_trained)
model.eval()
input = torch.ones(1, 3, 512, 512)
# 如果没opset_version=11(需要torch>=1.3) 报错：ONNX export failed: Couldn't export operator aten::upsample_bilinear2d
onnx_model = torch.onnx.export(model, (input,), '../model_data/Unet.onnx', opset_version=11, verbose=True)