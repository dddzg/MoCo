BATCH_SIZE = 256
import torch

GPU_ID = 0
DEVICE = torch.device(f'cuda:{GPU_ID}')
QUEUE_LENGTH = 8192

FILE_PATH = './models/best_model'
MODEL = 'resnet18'
RESUME = None
