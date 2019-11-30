BATCH_SIZE = 32
import torch

GPU_ID = 0
DEVICE = torch.device(f'cuda:{GPU_ID}')
QUEUE_LENGTH = 8192
