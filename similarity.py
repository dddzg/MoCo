from utils import get_transform
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
from tqdm import tqdm
from wideresnet import WideResNet
from torch.nn import functional as F
import types
import config
from torchvision import transforms
import pickle
import os


def random_int(n=100):
    return np.random.randint(0, n, 1)[0]


def get_model():
    model = WideResNet
    model_q = model(pretrained=None)

    def forward(self, input):
        x = self.features(input)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = F.normalize(x)  # l2 normalize by default
        return x

    model_q.forward = types.MethodType(forward, model_q)
    model_q.to(config.DEVICE)
    if os.path.isfile(config.FILE_PATH):
        print(f'loading model from {config.FILE_PATH}')
        checkpoint = torch.load(config.FILE_PATH, map_location=config.DEVICE)
        # config.__dict__.update(checkpoint['config'])
        model_q.load_state_dict(checkpoint['model_q'])
        # model_q = model_q.module
    return model_q


if __name__ == '__main__':
    label_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    image_size = 32
    cifar10_mean = (0.4914, 0.4822, 0.4465)  # equals np.mean(train_set.train_data, axis=(0,1,2))/255
    cifar10_std = (0.2471, 0.2435, 0.2616)  # equals np.std(train_set.train_data, axis=(0,1,2))/255
    test_transform = get_transform(image_size, mode='test', to_tensor=False)
    dataset = datasets.cifar.CIFAR10(root='./', train=True, transform=test_transform,
                                     download=True)
    features = []
    labels = []
    model_q = get_model()
    transform_to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    model_q.eval()
    if os.path.exists('./feature.pkl'):
        with open('./feature.pkl', 'rb') as f:
            features = pickle.load(f)
            labels = pickle.load(f)
    else:
        with torch.no_grad():
            for img, label in tqdm(dataset):
                labels.append(label_list[label])
                img = transform_to_tensor(img).to(config.DEVICE)
                img = img.unsqueeze(0)
                feat = model_q(img)
                features.append(feat.cpu().numpy()[0])
        with open('./feature.pkl', 'wb') as f:
            pickle.dump(features, f)
            pickle.dump(labels, f)
    pos_idx = random_int(len(features))
    pos_sample, pos_label = features[pos_idx], labels[pos_idx]
    # candidate_sample, candidate_label = features[:pos_idx] + features[pos_idx + 1:], labels[:pos_idx] + labels[
    #                                                                                                     pos_idx + 1, :]
    features = np.stack(features)
    tensor_pos_sample = torch.tensor(pos_sample).to(config.DEVICE).unsqueeze(0)
    tensor_candidate_sample = torch.tensor(features).to(config.DEVICE).transpose(0, 1)
    result = torch.mm(tensor_pos_sample, tensor_candidate_sample).cpu().numpy()[0]
    sim_with_label = [(sim, i) for i, sim in enumerate(result)]
    sim_with_label.sort(key=lambda x: x[0])  # small to large

    # visualize
    col_num = 5
    # original img
    f = plt.figure()
    pos_img, _ = dataset[pos_idx]
    f.add_subplot(3, col_num, 1)
    plt.imshow(pos_img)
    plt.title(pos_label)

    # positive img
    large_sim = sim_with_label[-col_num - 1:-1]
    print(len(large_sim))
    print(len(sim_with_label))
    for i in range(col_num):
        score, idx = large_sim[i]
        img, _ = dataset[idx]
        label = labels[idx]
        f.add_subplot(3, col_num, i + 1 + col_num)
        plt.imshow(img)
        plt.title(f'{label}_{score:.2f}')

    # negative img
    sm_sim = sim_with_label[:col_num]
    for i in range(col_num):
        score, idx = sm_sim[i]
        img, _ = dataset[idx]
        label = labels[idx]
        f.add_subplot(3, col_num, i + 1 + col_num * 2)
        plt.imshow(img)
        plt.title(f'{label}_{score:.2f}')

    # print(result)
    # print(np.array())
    # f = plt.figure()  #
    # for i in range(5):
    #     img, label = dataset[i]
    #     f.add_subplot(2, 5, i + 1)
    #     plt.imshow(img)
    #     plt.title(label_list[label])
    # for i in range(5):
    #     img, label = dataset[i]
    #     f.add_subplot(2, 5, i + 1 + 5)
    #     plt.imshow(img)
    #     plt.title(label_list[label])
    # for i in range(5):
    #     img, label = dataset[i]
    #     f.add_subplot(3, 5, i + 1)
    #     plt.imshow(img)
    #     plt.title(label_list[label])
    plt.show()
