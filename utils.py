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
