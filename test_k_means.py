from scipy.stats import ortho_group
import pickle
import numpy as np
from torchvision import datasets
from utils import get_transform
import matplotlib.pyplot as plt


def random_int(n=100, m=1):
    if m == 1:
        return np.random.randint(0, n, 1)[0]
    else:
        return list(np.random.randint(0, n, m))


def l2_norm(l):
    all_l = np.sqrt((l * l).sum()) + 1e-10
    return l / all_l


def dis(a, b):
    return 1 - (a * b).sum()


# print(l2_norm(np.array([0.6, 0.8])))

# print(features.shape)
# random select N orthogonal
N = 2000
for N in [250, 500]:
    for iter in range(0, 5):
        with open('./feature.pkl', 'rb') as f:
            features = pickle.load(f)
            labels = pickle.load(f)
        features = np.stack(features)
        C = 128
        # idxs = random_int(C, N)
        import math

        dzg = ortho_group.rvs(dim=C)[:N]
        if N <= C:
            central_vector = ortho_group.rvs(dim=C)[:N]
        else:
            central_vector = np.random.rand(N, C)
        # t = (central_vector[0] + central_vector[1]) / math.sqrt(2)
        # print(t @ central_vector[0])
        # exit(0)
        # normalize
        for i in range(N):
            central_vector[i] = l2_norm(central_vector[i])

        iter_num = 0
        while True:
            distance = central_vector @ features.T  # (N,sample num)
            selected_central = distance.argmax(0)
            all_error = 0
            for i in range(N):
                nearest_idx = np.where(selected_central == i)[0]
                nearest_features = features[nearest_idx]
                # print(.shape)
                if nearest_features.shape[0] == 0:
                    print('random central')
                    features_center = sum([features[np.random.randint(0, features.shape[0], 1)[0]] for i in range(1)])
                    # features_center = sum([dzg[i] for i in l])
                    new_central = l2_norm(features_center)
                else:
                    # print(nearest_features.shape)
                    new_central = l2_norm(nearest_features.mean(axis=0))
                all_error += dis(central_vector[i], new_central)
                central_vector[i] = new_central
            print(f'iter:{iter_num}, error:{all_error}')
            # print((central_vector @ central_vector.T))
            iter_num += 1
            if all_error < 5e-4:
                break

        distance = central_vector @ features.T
        all_result = []
        central = distance.argmax(0)
        for i in range(N):
            nearest_idxs = np.where(central == i)[0]
            nearest_features = features[nearest_idxs]
            if nearest_features.shape[0] == 0:
                print('bug')
            else:
                central_vector_i = central_vector[[i]]
                nearest_idx = (central_vector_i @ (nearest_features.T)).argmax(1)[0]
                nearest_idx = nearest_idxs[nearest_idx]
                all_result.append(nearest_idx)
        print(len(set(all_result)))
        import torch
        import os

        root = f'~/dataset/cifar_{N}_{iter}.pkl'
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        with open(root, 'wb') as f:
            pickle.dump(all_result, f)

        from collections import defaultdict

        ret = defaultdict(lambda: 0)
        label_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        image_size = 32
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        test_transform = get_transform(image_size, mode='test', to_tensor=False)
        dataset = datasets.cifar.CIFAR10(root='./', train=True, transform=test_transform,
                                         download=True)
        for x in all_result:
            label = label_list[dataset[x][1]]
            ret[label] += 1
        print(ret)
# all_result = distance.argmax(1)
# from collections import defaultdict
#
# ret = defaultdict(lambda: 0)
#
# print(all_result)
# with open('./cifar@250_random_1.pkl', 'wb') as f:
#     pickle.dump(all_result, f)
#     # pickle.dump(labels, f)
#
# # print(all_result)
# # # selected_img = (-distance).argsort(axis=1)  # large to small
# # # print(selected_img.shape)
# # #
# label_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# image_size = 32
# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]
# test_transform = get_transform(image_size, mode='test', to_tensor=False)
# dataset = datasets.cifar.CIFAR10(root='./', train=True, transform=test_transform,
#                                  download=True)
# for x in all_result:
#     label = label_list[dataset[x][1]]
#     ret[label] += 1
# print(ret)
# result = np.array(all_result[:50]).reshape(10, 5)
# f = plt.figure(figsize=(10, 20))
# for i in range(10):
#     for j in range(5):
#         idx = result[i][j]
#         img, l = dataset[idx]
#         label = label_list[l]
#         f.add_subplot(10, 5, i * 5 + j + 1)
#         plt.imshow(img)
#         plt.title(f'{label}')
# plt.show()


# sm_sim = sim_with_label[:col_num]
# M = 5
# f = plt.figure(figsize=(10, 20))
# for i in range(N):
#     for j in range(M):
#         # score, idx = sm_sim[i]
#         idx = selected_img[i][j]
#         img, l = dataset[idx]
#         label = label_list[l]
#         f.add_subplot(N, M, i * M + j + 1)
#         plt.imshow(img)
#         plt.title(f'{label}_{distance[i][idx]:.1f}')
# plt.show()
# a = np.array([1, 2, 3, 1, 5, 4, 3])
# print(np.where(a == 1))
