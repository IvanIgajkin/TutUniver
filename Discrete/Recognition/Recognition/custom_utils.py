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
VECTORS_FILE = 'original_vectors.txt'
MIDDLE_FILE = 'middle_image.txt'
WEIGHTS_FILE = 'weights.txt'

ACC = 0.9
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


def get_original_data():
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

        new_vectors = basis.dot(clean_data)

        #print_img(new_vectors[-1] + mid_image_data)

        np.savetxt('{0}/{1}'.format(DATA_DIR, VECTORS_FILE), new_vectors)
        np.savetxt('{0}/{1}'.format(DATA_DIR, MIDDLE_FILE), mid_image_data)

        return (new_vectors, mid_image_data)

    return (get_fdata(VECTORS_FILE), get_fdata(MIDDLE_FILE))


def fit():
    train_data = np.array([np.mean( \
            (np.asarray(im.open('{0}/Images/Original/{1}.bmp'.format(DATA_DIR, n * 5 + 1)), dtype=D_TYPE)) / 255.0, \
            axis=-1, dtype=D_TYPE).flatten() \
            for n in range(N // 5)])

    test_data, mid_image_data = get_original_data()

    if not data_exists(WEIGHTS_FILE):
        clean_train_data = np.array([np.abs(fi - mid_image_data)
                        for fi in train_data], dtype=D_TYPE)

        weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        for k in range(N // 5):
            tmp = clean_train_data[k]
            weights[k] = weights[k] / np.linalg.norm(np.sqrt(tmp.dot(tmp)))
        

        face_index, step = 0, 0
        for Li in test_data:
            ro = []
            for k in range(N // 5):
                tmp = np.array(Li - person_func(k, test_data))
                ro.append(np.linalg.norm(np.sqrt(tmp.dot(tmp))))

            if face_index != np.argmin(ro):
                weights[face_index] = weights[face_index] / 10.0

            step = step + 1
            if (step % 5 == 0):
                face_index = face_index + 1

        weights = np.array(weights)
        np.savetxt('{0}/{1}'.format(DATA_DIR), weights)

        return (weights, test_data)

    return (get_fdata(WEIGHTS_FILE), test_data)


def recognise(face_data):
    img = im.fromarray(face_data)

    image_data = np.array(np.mean( \
            (np.asarray(img.resize(IMG_SHAPE), dtype=D_TYPE)) / 255.0, \
            axis=-1, dtype=D_TYPE).flatten())

    print('Recognition in progress...')

    wghts, vectors = fit()

    ro = []
    for k in range(N // 5):
        tmp = np.array(image_data - person_func(k, vectors))
        ro.append(np.linalg.norm(np.sqrt(tmp.dot(tmp))))

    print(persons[np.argmin(ro)])
    #print(persons[np.argmin(wghts.dot(ro).flatten())])
