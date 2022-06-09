class ANN_tf(object):
    def __init__(self, image_size):
        self.image_size = image_size

    def load(self, path):
        self.model = tf.keras.models.load_model(path)

    def predict(self, image):
        self.X = image_preprocessing(image, self.image_size, True, (50, 50))
        self.Y = self.model.predict(self.X.reshape(1, *self.X.shape, 1), verbose=0)
        return self.Y[0, 0]

def image_preprocessing(image, image_size=None, flatten=False, pad_width=(0, 0)):
    array = (np.array(image.convert('L'), dtype=np.uint8) + 1).astype(np.float32)

    n, m = array.shape
    direct = lambda i: any(0 != a for a in array[:, i])
    back = lambda i: any(0 != a for a in array[i, :])
    l = next(filter(direct, range(m)), 0)
    r = next(filter(direct, reversed(range(m))), 0)
    t = next(filter(back, range(n)), 0)
    b = next(filter(back, reversed(range(n))), 0)

    array = np.pad(array[t:b+1, l:r+1], pad_width, 'constant', constant_values=0)

    if image_size is not None:
        array = np.array(fromarray(array).resize(image_size), dtype=np.float32)

    if flatten:
        array = array.flatten()

    array = np.abs(array)

    return array
