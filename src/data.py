import numpy as np
import tensorflow as tf
from .config import SEED

def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train.astype("float32") / 255.0, x_test.astype("float32") / 255.0
    rng = np.random.default_rng(SEED); indices = rng.permutation(len(x_train))
    return (x_train[indices[:55000], ..., None], y_train[indices[:55000]]), (x_train[indices[55000:60000], ..., None], y_train[indices[55000:60000]]), (x_test[..., None], y_test)

def datasets(batch_size=128):
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()
    make = lambda x, y, shuffle=False: tf.data.Dataset.from_tensor_slices((x,y)).shuffle(55000, seed=SEED, reshuffle_each_iteration=False).batch(batch_size) if shuffle else tf.data.Dataset.from_tensor_slices((x,y)).batch(batch_size)
    return make(x_train,y_train,True), make(x_val,y_val), make(x_test,y_test)
