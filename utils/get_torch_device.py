import torch

def get_torch_device(cuda_idx):
    USE_CUDA = torch.cuda.is_available()
    if USE_CUDA:
        if cuda_idx != -1:
            device_str = f"cuda:{cuda_idx}"
        else:
            device_str = "cuda"
    else:
        device_str = "cpu"
    device = torch.device(device_str)
    print('Using Device: ', device_str)
    return device