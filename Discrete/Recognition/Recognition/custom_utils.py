import numpy as np
from PIL import Image as im
import collections as colls
from os.path import exists
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as skPCA
#Global parameters
DATA_DIR = './Data'
N = 40
D_TYPE = np.float32

VECTORS_FILE = 'original_vectors.txt'
MIDDLE_FILE = 'middle_image.txt'

ACC = 0.98
IMG_SHAPE = (64, 64)
#Global parameters

persons = {0: 'Oleg', 1: 'Dima', 2: 'Ivan', 3: 'Michel', 4: 'Alexandr', 5: 'Vova', 6: 'Alina', 7: 'Olesya'}


def print_img(data):
    plt.imshow(data.reshape(IMG_SHAPE))
    plt.show()


def get_fpath(fname):
    return '{0}/{1}'.format(DATA_DIR, fname)


def get_fdata(fname):
    return np.loadtxt('{0}/{1}'.format(DATA_DIR, fname))


def data_exists(*fnames):
    for fname in fnames:
        if not exists(get_fpath(fname)):
            return False
    
    return True


def person_func(k, vectors):
    tmp = vectors[5*k:5*k+5]
    return np.array([np.mean(col, axis=-1, dtype=D_TYPE) for col in tmp.transpose()])


class PCA:
    def __init__(self):
            #get data from original photos
            original_photos_data = np.array([np.mean( \
                (np.asarray(im.open('{0}/Images/Original/{1}.bmp'.format(DATA_DIR, n + 1)).resize(IMG_SHAPE) , dtype=D_TYPE)) / 255.0, \
                axis=-1, dtype=D_TYPE).flatten() \
                for n in range(N)])

            k_comp = 40
            p = skPCA(k_comp)
            p.fit(original_photos_data)

            self.mid = p.mean_
            self.vectors = p.components_.T
            self.basis = (original_photos_data - self.mid) @ self.vectors

            return

            #get middle image data
            mid_image_data = np.array([np.mean(fi, axis=-1, dtype=D_TYPE) \
                            for fi in original_photos_data.transpose()]).flatten()

            #exclude middle image data
            clean_data = np.array([np.abs(fi - mid_image_data)
                                    for fi in original_photos_data], dtype=D_TYPE)

            #find matrix psi * psi_T
            cov_mtrx = np.cov(clean_data)
            #find its eign values and vectors
            values, vectors = np.linalg.eigh(cov_mtrx)
            #sort them
            ind = np.abs(values).argsort()[::-1]

            L_origin = np.sum(np.abs(values))
            main_values = []
            for v in values[ind]:
              if sum(np.abs(main_values)) >= ACC * L_origin:
                break

              main_values.append(v)

            main_values = np.array(main_values)

            d = len(main_values)
            sorted_vectors = vectors[ind]
            basis = np.asarray(list(map(lambda vi: vi / np.linalg.norm(vi), sorted_vectors[:d])))

            new_vectors = basis.dot(clean_data)

            print_img(new_vectors[0] + mid_image_data)

            self.persons = []
            for k in range(N // 5):
                self.persons.append(person_func(k, new_vectors))

            self.basis, self.mid, self.vectors = basis, mid_image_data, new_vectors


    def recognise(self, face_data):
        img = im.fromarray(face_data)

        image_data = np.array(np.mean( \
                np.asarray(img.resize(IMG_SHAPE)) / 255.0, \
                axis=-1, dtype=D_TYPE).flatten())

        diff = self.basis - (image_data - self.mid) @ self.vectors
        ro = np.linalg.norm(diff, axis=1)

        print(persons[np.argmin(ro) // 5])

        return

        clean_image_data = np.abs(image_data - self.mid)

        ro = []
        for k in range(N // 5):
            tmp = np.array(clean_image_data - self.persons[k])
            ro.append(np.linalg.norm(tmp))
