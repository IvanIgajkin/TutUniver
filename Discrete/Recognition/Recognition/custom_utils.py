import numpy as np
from PIL import Image as im
import collections as colls
from os.path import exists
import matplotlib.pyplot as plt

#Global parameters
DATA_DIR = './Data'
N = 40
D_TYPE = np.float32

VECTORS_FILE = 'original_vectors.txt'
MIDDLE_FILE = 'middle_image.txt'

ACC = 1.00
IMG_SHAPE = (400, 300)
#Global parameters

persons = {0: 'Oleg', 1: 'Dima', 2: 'Ivan', 3: 'Michel', 4: 'Alexandr', 5: 'Alina', 6: 'Olesya', 7: 'Vova'}


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
        if not data_exists(VECTORS_FILE, MIDDLE_FILE):
            #get data from original photos
            original_photos_data = np.array([np.mean( \
                (np.asarray(im.open('{0}/Images/Original/{1}.bmp'.format(DATA_DIR, n + 1)), dtype=D_TYPE)) / 255.0, \
                axis=-1, dtype=D_TYPE).flatten() \
                for n in range(N)])

            #get middle image data
            mid_image_data = np.array([np.mean(fi, axis=-1, dtype=D_TYPE) \
                            for fi in original_photos_data.transpose()]).flatten()

            #exclude middle image data
            clean_data = np.array([np.abs(fi - mid_image_data)
                                    for fi in original_photos_data], dtype=D_TYPE)

            #find matrix psi * psi_T
            G_mtrx = clean_data.dot(clean_data.transpose())
            #find its eign values and vectors
            values, vectors = np.linalg.eigh(G_mtrx)
            #sort them
            tmp_dict = colls.OrderedDict(sorted(dict(zip(values, vectors)).items(), reverse=True))
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

            new_vectors = basis.dot(clean_data)

            #print_img(new_vectors[-1] + mid_image_data)

            np.savetxt('{0}/{1}'.format(DATA_DIR, VECTORS_FILE), new_vectors)
            np.savetxt('{0}/{1}'.format(DATA_DIR, MIDDLE_FILE), mid_image_data)

            self.vectors, self.mid = new_vectors, mid_image_data
        else:
            self.vectors, self.mid = get_fdata(VECTORS_FILE), get_fdata(MIDDLE_FILE)


    def recognise(self, face_data):
        img = im.fromarray(face_data)

        image_data = np.array(np.mean( \
                (np.asarray(img.resize(IMG_SHAPE), dtype=D_TYPE)) / 255.0, \
                axis=-1, dtype=D_TYPE).flatten())

        print('Recognition in progress...')

        clean_image_data = np.abs(image_data - self.mid)

        print_img(clean_image_data)
        print_img(person_func(2, self.vectors))

        ro = []
        for k in range(N // 5):
            tmp = np.array(clean_image_data - person_func(k, self.vectors))
            ro.append(np.linalg.norm(tmp))

        print(persons[np.argmin(ro)])
