from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import config
from dataset import custom_dataset
import pretrainedmodels as models
import torch
from tqdm import tqdm
from torch.nn import functional as F
import types
from utils import AverageMeter, get_shuffle_idx
import os
from network import CustomNetwork, Net
from utils import get_transform
from wideresnet import WideResNet

# torch.nn.BatchNorm1d
def parse_option():
    return None


def get_model(model_name='resnet18'):
    try:
        model = WideResNet
        # model = CustomNetwork
        model_q = model(pretrained=None)
        model_k = model(pretrained=None)

        def forward(self, input):
            x = self.features(input)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = x.view(x.size(0), -1)
            x = F.normalize(x)  # l2 normalize by default
            return x

        model_q.forward = types.MethodType(forward, model_q)
        model_k.forward = types.MethodType(forward, model_k)

        # for model k, it doesn't require grad
        for param in model_k.parameters():
            param.requires_grad = False

        device_list = [config.GPU_ID] * 4  # Shuffle BN can be applied through there is only one gpu.
        model_q = torch.nn.DataParallel(model_q, device_ids=device_list)
        model_k = torch.nn.DataParallel(model_k, device_ids=device_list)

        model_q.to(config.DEVICE)
        model_k.to(config.DEVICE)
        return model_q, model_k
    except KeyError:
        print(f'model name:{model_name} does not exist.')


def momentum_update(model_q, model_k, m=0.999):
    """ model_k = m * model_k + (1 - m) model_q """
    for p1, p2 in zip(model_q.parameters(), model_k.parameters()):
        p2.data.mul_(m).add_(1 - m, p1.detach().data)


def enqueue(queue, k):
    return torch.cat([queue, k], dim=0)


def dequeue(queue, max_len=config.QUEUE_LENGTH):
    if queue.shape[0] >= max_len:
        return queue[-max_len:]  # queue follows FIFO
    else:
        return queue


def train(train_dataloader, model_q, model_k, queue, optimizer, device, t=0.07):
    model_q.train()
    model_k.train()
    losses = AverageMeter()
    pred_meter = AverageMeter()
    for i, (img_q, img_k, _) in enumerate(tqdm(train_dataloader)):
        if queue is not None and queue.shape[0] == config.QUEUE_LENGTH:
            img_q, img_k = img_q.to(device), img_k.to(device)
            q = model_q(img_q)  # N x C

            # shuffle BN
            shuffle_idx, reverse_idx = get_shuffle_idx(config.BATCH_SIZE, device)
            img_k = img_k[shuffle_idx]
            k = model_k(img_k)  # N x C
            k = k[reverse_idx].detach()  # reverse and no graident to key

            N, C = q.shape
            # K = config.QUEUE_LENGTH

            l_pos = torch.bmm(q.view(N, 1, C), k.view(N, C, 1)).view(N, 1)  # positive logit N x 1
            l_neg = torch.mm(q.view(N, C), queue.transpose(0, 1))  # negative logit N x K
            labels = torch.zeros(N, dtype=torch.long).to(device)  # positives are the 0-th
            logits = torch.cat([l_pos, l_neg], dim=1) / t
            # print(logits[0])
            pred = logits[:, 0].mean()
            loss = criterion(logits, labels)
            losses.update(loss.item(), N)
            pred_meter.update(pred.item(), N)

            # update model_q
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update model_k by momentum
            momentum_update(model_q, model_k, 0.999)
        else:
            img_k = img_k.to(device)
            shuffle_idx, reverse_idx = get_shuffle_idx(config.BATCH_SIZE, device)
            img_k = img_k[shuffle_idx]
            k = model_k(img_k)  # N x C
            k = k[reverse_idx].detach()  # reverse and no graident to key

        # update dictionary
        queue = enqueue(queue, k) if queue is not None else k
        queue = dequeue(queue)
    return {
               'loss': losses.avg,
               'pred': pred_meter.avg
           }, queue


if __name__ == '__main__':
    args = parse_option()
    image_size = 32
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # image_size = 28
    # mean = [0.1307, ]
    # std = [0.3081, ]
    # normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = get_transform(image_size, mean=mean, std=std, mode='train')
    # datasets.mnist.MNIST
    train_dataset = custom_dataset(datasets.cifar.CIFAR10)(root='./', train=True, transform=train_transform,
                                                           download=True)
    print(len(train_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0,
                                  pin_memory=False, drop_last=True)  # drop the last batch due to irregular size

    model_q, model_k = get_model(config.MODEL)

    optimizer = torch.optim.SGD(model_q.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 50], gamma=0.1)

    # copy parameters from model_q to model_k
    momentum_update(model_q, model_k, 0)

    criterion = torch.nn.CrossEntropyLoss()

    torch.backends.cudnn.benchmark = True
    queue = None
    start_epoch = 0
    min_loss = float('inf')
    # load model from file
    if config.RESUME and os.path.isfile(config.FILE_PATH):
        print(f'loading model from {config.FILE_PATH}')
        checkpoint = torch.load(config.FILE_PATH)
        # config.__dict__.update(checkpoint['config'])
        model_q.module.load_state_dict(checkpoint['model_q'])
        model_k.module.load_state_dict(checkpoint['model_k'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        min_loss = checkpoint['min_loss']
        print(f'loaded model from {config.FILE_PATH}')

    for epoch in range(start_epoch, config.ALL_EPOCHS):
        ret, queue = train(train_dataloader, model_q, model_k, queue, optimizer, config.DEVICE)
        ret_str = ' - '.join([f'{k}:{v:.4f}' for k, v in ret.items()])
        print(f'epoch:{epoch} {ret_str}')
        scheduler.step()
        # print(type(config))
        if ret['loss'] < min_loss:
            min_loss = ret['loss']
            state = {
                # 'config': config.__dict__,
                'model_q': model_q.module.state_dict(),
                'model_k': model_k.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'min_loss': min_loss
            }
            print(f'save to {config.FILE_PATH}')
            torch.save(state, config.FILE_PATH)
