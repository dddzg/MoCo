import torch
import numpy as np
from PIL import Image


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, fillcolor=(128, 128, 128)):
        self.n_holes = n_holes
        self.fillcolor = fillcolor

    def __call__(self, img, length=20):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size[0]
        w = img.size[1]
        pixels = img.load()
        length = min(length, min(w, h) // 2)  # length should not be larger than half of width or height
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)

            for i in range(y1, y2):
                for j in range(x1, x2):
                    pixels[i, j] = self.fillcolor
        return img


if __name__ == '__main__':
    a = Image.open('./test.jpg')
    c = Cutout(1)
    d = c(a, 16)
    print(type(d))
    d.save('dzg.jpg')
