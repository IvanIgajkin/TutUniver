import numpy as np
import numdifftools as nd
from PIL.Image import fromarray

def image_to_array(img, size=None, dtype=np.float32, flatten=False):
    if size is None:
        size = img.size

    img = img.resize(size).convert('L')
    arr = (np.array(img, dtype=np.uint8) + 1).astype(dtype)

    if flatten:
        arr = arr.flatten()

    return arr

def image_to_crop_array(img, size=None, dtype=np.float32, flatten=False):
    arr = image_to_array(img, dtype=np.uint8)

    n, m = arr.shape
    direct = lambda i: any(0 != a for a in arr[:, i])
    back = lambda i: any(0 != a for a in arr[i, :])
    l = next(filter(direct, range(m)), 0)
    r = next(filter(direct, reversed(range(m))), 0)
    t = next(filter(back, range(n)), 0)
    b = next(filter(back, reversed(range(n))), 0)

    img = fromarray(arr[t:b+1, l:r+1] != 0)
    arr = image_to_array(img, size=size, dtype=dtype, flatten=flatten)
    # with open('model9.npy', 'wb') as f:
    #     np.save(f, arr)

    return arr


class ANN(object):
    def __init__(self, image_size, threshold=0.25):
        self.image_size = image_size
        self.threshold = threshold

    def load(self, path):
        with open(path, 'rb') as f:
            self.w = np.load(f)
        

    def predict(self, image):
        X = image_to_crop_array(image, size=self.image_size, dtype=bool, flatten=True)
        s = np.logical_xor(X, self.w).sum()
        a = s / (self.image_size[0] * self.image_size[1])
        Y = a <= self.threshold
        percts = round((1 - a) * 100, 2)

        
        # for i in range(10):
        #     with open(f'.\model{i}.npy', 'rb') as f:
        #         w = np.load(f)
        #         print(np.reshape(w, self.image_size))
        #         print()
        
        self.rework(Y, percts, self.w)
        return percts / 100

    
    def E(self, x):
        return self.w.sum() * 1 / (self.w @ x)
        

    def rework(self, Y, percts, w):
        return
        for i in range(len(w)):
            w[i] = w[i] - 1.e-2 * nd.Gradient(self.E)(w)

        print(np.reshape(w, self.image_size))

        s = np.logical_xor(X, w).sum()
        a = s / (self.image_size[0] * self.image_size[1])
        Y = a <= self.threshold
        Y, round((1 - a) * 100, 2)
