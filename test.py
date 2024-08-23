import torch
import numpy as np 
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.device_count()
tensor = torch.tensor([1,2,3])
print(tensor, tensor.device)

### Moving tensors back to gpu 
tensor_on_gpu = tensor.to(device)
print(tensor_on_gpu)

### Moving tensors back to the cpu 
tensor_on_cpu = tensor_on_gpu.cpu().numpy()
print(tensor_on_cpu)


