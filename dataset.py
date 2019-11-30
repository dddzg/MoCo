from PIL import Image


def custom_dataset(base_dataset):
    name: str = base_dataset.__class__.__name__

    class CustomDataSet(base_dataset):
        def __init__(self, *args, **kwargs):
            super(CustomDataSet, self).__init__(*args, **kwargs)

        def __getitem__(self, index):
            img, target = self.data[index], int(self.targets[index])
            if name.startswith('MNIST'):
                img = Image.fromarray(img.numpy(), mode='L')
            else:
                img = Image.fromarray(img)

            ret_img_q = self.transform(img)
            ret_img_k = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return ret_img_q, ret_img_k, target

    return CustomDataSet


# import torchvision.models as models
#
# print(type(models.__dict__['resnet18']))
