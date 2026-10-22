import tensorflow as tf

def build_lenet():
    i=tf.keras.Input((28,28,1)); x=tf.keras.layers.Conv2D(6,5,activation="tanh",padding="same")(i); x=tf.keras.layers.AveragePooling2D(pool_size=2)(x); x=tf.keras.layers.Conv2D(16,5,activation="tanh")(x); x=tf.keras.layers.AveragePooling2D(pool_size=2)(x); x=tf.keras.layers.Flatten()(x); x=tf.keras.layers.Dense(120,activation="tanh")(x); x=tf.keras.layers.Dense(84,activation="tanh")(x); return tf.keras.Model(i,tf.keras.layers.Dense(10,activation="softmax")(x),name="Improved_LeNet_5")

def build_cnn():
    i=tf.keras.Input((28,28,1)); x=i
    for f in (32,64): x=tf.keras.layers.Conv2D(f,3,padding="same")(x); x=tf.keras.layers.BatchNormalization()(x); x=tf.keras.layers.ReLU()(x); x=tf.keras.layers.MaxPooling2D()(x); x=tf.keras.layers.Dropout(.15)(x)
    x=tf.keras.layers.Flatten()(x); x=tf.keras.layers.Dense(128,activation="relu")(x); x=tf.keras.layers.Dropout(.35)(x); return tf.keras.Model(i,tf.keras.layers.Dense(10,activation="softmax")(x),name="Augmented_CNN")

def build_model(name):
    from .config import model_key
    m=build_lenet() if model_key(name)=="lenet" else build_cnn(); m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss="sparse_categorical_crossentropy",metrics=["accuracy"]); return m
