import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from app import App
from ann import ANN

def train_model_ANN():
    pass

if __name__ == '__main__':
    train_model_ANN()

    a = App((28, 28), ANN)
    #a.load_model_ANN('models\model_tf.h5')
    a.load_model_ANN('model1.npy')
    a.get_UI(title='Распознавание цифр', brush_size=16)