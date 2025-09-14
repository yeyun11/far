import torch
from torchvision.models.resnet import resnet18


from far import monkey_patch_


rand_input = torch.randn(1, 3, 224, 224)
model = resnet18(pretrained=False)
with torch.no_grad():
    out = model(rand_input)

# Frequency-Aware Reparameterization
monkey_patch_(model)
with torch.no_grad():
    out = model(rand_input)
