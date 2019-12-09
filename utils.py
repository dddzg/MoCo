from torchvision import transforms, datasets


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


import torch


def get_shuffle_idx(bs, device):
    """shuffle index for ShuffleBN """
    shuffle_value = torch.randperm(bs).long().to(device)  # index 2 value
    reverse_idx = torch.zeros(bs).long().to(device)
    arange_index = torch.arange(bs).long().to(device)
    reverse_idx.index_copy_(0, shuffle_value, arange_index)  # value back to index
    return shuffle_value, reverse_idx


def get_transform(image_size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], mode='train', to_tensor=True):
    transform_to_tensor = [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ] if to_tensor else []
    if mode == 'train':
        # train_transforms =
        train_transforms = [
            transforms.RandomResizedCrop(image_size,
                                         scale=(0.8, 1.2),
                                         ratio=(0.8, 1.2),
                                         interpolation=3)] if image_size < 128 else [
            transforms.RandomResizedCrop(image_size,
                                         scale=(0.3, 1.0),
                                         ratio=(0.7, 1.4),
                                         interpolation=3)]

        return transforms.Compose(train_transforms +
                                  [
                                      transforms.RandomApply([
                                          transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomGrayscale(p=0.25),
                                      # transforms.ToTensor(),
                                      # transforms.Normalize(mean=mean, std=std)
                                  ] + transform_to_tensor)
    else:
        test_transforms = [transforms.Resize(image_size, interpolation=3)] if image_size < 128 else [
            transforms.Resize(image_size + 16, interpolation=3),
            transforms.CenterCrop(image_size)]
        return transforms.Compose(test_transforms + transform_to_tensor)
