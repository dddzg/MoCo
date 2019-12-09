BATCH_SIZE = 256
import torch

GPU_ID = 0
DEVICE = torch.device(f'cuda:{GPU_ID}')
QUEUE_LENGTH = 4096

FILE_PATH = './models/best_model_cifar'
MODEL = 'resnet18'
RESUME = None
ALL_EPOCHS = 60
