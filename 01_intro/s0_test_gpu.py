import torch

"""
    CUDA 11.3
    conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
"""

# Check for available GPU, use the first GPU if available, otherwise use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Create two random tensors on CPU
tensor_a = torch.rand(3, 3)
tensor_b = torch.rand(3, 3)

# Move tensors to GPU
tensor_a = tensor_a.to(device)
tensor_b = tensor_b.to(device)

# Perform tensor addition on GPU
result_tensor = tensor_a + tensor_b

# Move the result back to CPU if necessary
result_tensor = result_tensor.to("cpu")

print("Tensor A (on GPU):\n", tensor_a)
print("Tensor B (on GPU):\n", tensor_b)
print("Result Tensor (on CPU):\n", result_tensor)
