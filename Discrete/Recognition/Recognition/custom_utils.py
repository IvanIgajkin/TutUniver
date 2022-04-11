import numpy as np
from PIL import Image as im
import collections as colls
from os.path import exists
import matplotlib.pyplot as plt

#Global parameters
DATA_DIR = './Data'
N = 40
D_TYPE = np.float32

BASIS_FILE = 'basis.txt'
VALUES_FILE = 'original_values.txt'
VECTORS_FILE = 'original_vectors.txt'
MIDDLE_FILE = 'middle_image.txt'

ACC = 0.9
IMG_SHAPE = (64, 64)
#Global parameters


def print_img(data):
    plt.imshow(data.reshape(IMG_SHAPE))
    plt.show()


def get_fpath(fname):
    return '{0}/{1}'.format(DATA_DIR, fname)


def data_exists():
    return exists(get_fpath(BASIS_FILE)) and \
        exists(get_fpath(VALUES_FILE)) and \
        exists(get_fpath(VECTORS_FILE)) and \
        exists(get_fpath(MIDDLE_FILE))


def person_func(k):
    vectors = np.loadtxt(get_fpath(VECTORS_FILE), delimiter=';')
    return vectors[5*k:5*k+5]


def get_original_data():
    if not data_exists():
        #get data from original photos
        original_photos_data = np.array([np.mean( \
            (np.asarray(im.open('{0}/Images/Original/{1}.bmp'.format(DATA_DIR, n + 1)).resize(IMG_SHAPE), dtype=D_TYPE)) / 255.0, \
            axis=-1, dtype=D_TYPE).flatten() \
            for n in range(N)])

        #get middle image data
        mid_image_data = np.array([np.mean(fi, axis=-1, dtype=D_TYPE) \
                        for fi in original_photos_data.transpose()]).flatten()

        #exclude middle image data
        clean_data = np.array([np.abs(fi - mid_image_data)
                                for fi in original_photos_data], dtype=np.float32)

        #find matrix psi * psi_T
        G_mtrx = clean_data.transpose().dot(clean_data)
        #find its eign values and vectors
        values, vectors = np.linalg.eigh(G_mtrx)
        #sort them
        tmp_dict = colls.OrderedDict(sorted(dict(zip(values, vectors)).items()))
        values, vectors = np.array(list(tmp_dict.keys())), \
                          np.array(list(tmp_dict.values()))

        L_origin = np.sum(np.abs(values))
        main_values = []
        for v in values:
          if sum(np.abs(main_values)) >= ACC * L_origin:
            break

          main_values.append(v)

        main_values = np.array(main_values)

        d = len(main_values)
        basis = np.asarray(list(map(lambda vi: vi / np.linalg.norm(vi), vectors[:d])))

        new_vectors = vectors.dot(basis)

        np.savetxt(get_fpath(BASIS_FILE), basis, delimiter=';')
        np.savetxt(get_fpath(VALUES_FILE), main_values, delimiter=';')
        np.savetxt(get_fpath(VECTORS_FILE), np.matrix(new_vectors), delimiter=';')
        np.savetxt(get_fpath(MIDDLE_FILE), mid_image_data, delimiter=';')

        return (basis, main_values, new_vectors, mid_image_data)

    def get_data(fname):
        return np.loadtxt(get_fpath(fname), delimiter=';')

    return (get_data(BASIS_FILE), get_data(VALUES_FILE), get_data(VECTORS_FILE), get_data(MIDDLE_FILE))


def recognise(face_data):
    image_data = np.array(np.mean( \
            (np.asarray(face_data, dtype=D_TYPE)) / 255.0, \
            axis=-1, dtype=D_TYPE).flatten())


