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


# torch.nn.BatchNorm1d
def parse_option():
    return None


def get_transform(image_size, mean, std, mode='train'):
    if mode == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.3, 1.0), ratio=(0.7, 1.4),
                                         interpolation=3),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8),
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(p=0.25),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size[0] + 16, image_size[1] + 16), interpolation=3),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])


def get_model(model_name='resnet18'):
    try:
        model = models.__dict__[model_name]
        model_q = model()
        model_k = model()

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

        device_list = [config.GPU_ID] * 4
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
    for i, (img_q, img_k, _) in enumerate(tqdm(train_dataloader)):
        if queue is not None and queue.shape[0] == config.QUEUE_LENGTH:
            img_q, img_k = img_q.to(device), img_k.to(device)
            shuffle_idx, reverse_idx = get_shuffle_idx(config.BATCH_SIZE, device)
            q = model_q(img_q)  # N x C

            # shuffle BN
            img_k = img_k[shuffle_idx]
            k = model_k(img_k)  # N x C
            k = k[reverse_idx].detach()  # reverse and no graident to key

            N, C = q.shape
            K = config.QUEUE_LENGTH

            l_pos = torch.bmm(q.view(N, 1, C), k.view(N, C, 1)).view(N, 1)  # positive logit N x 1
            l_neg = torch.mm(q.view(N, C), queue.view(C, K))  # negative logit N x C
            labels = torch.zeros(N, dtype=torch.long).to(device)  # positives are the 0-th
            logits = torch.cat([l_pos, l_neg], dim=1)

            loss = criterion(logits / t, labels)
            losses.update(loss.item(), N)

            # update model_q
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update model_k by momentum
            momentum_update(model_q, model_k, 0.999)
        else:
            img_k = img_k.to(device)
            k = model_k(img_k)
            k.detach()

        # update dictionary
        queue = enqueue(queue, k) if queue is not None else k
        queue = dequeue(queue)
    return {
               'loss': losses.avg
           }, queue


if __name__ == '__main__':
    args = parse_option()
    image_size = 64
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = get_transform(image_size, mean, std, mode='train')

    train_dataset = custom_dataset(datasets.cifar.CIFAR10)(root='./', train=True, transform=train_transform,
                                                           download=True)
    print(len(train_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4,
                                  pin_memory=False, drop_last=True)

    model_q, model_k = get_model('resnet18')

    optimizer = torch.optim.SGD(model_q.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)

    # copy parameters from model_q to model_k
    momentum_update(model_q, model_k, 0)

    criterion = torch.nn.CrossEntropyLoss()

    torch.backends.cudnn.benchmark = True
    queue = None
    for epochs in range(0, 90):
        ret, queue = train(train_dataloader, model_q, model_k, queue, optimizer, config.DEVICE)
        ret_str = ' - '.join([f'{k}:{v:.4f}' for k, v in ret.items()])
        print(f'epoch:{epochs} {ret_str}')
