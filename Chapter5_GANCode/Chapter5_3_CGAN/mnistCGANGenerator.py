import numpy as np
from tensorflow.keras.layers import Activation, BatchNormalization, Dense, Input, LeakyReLU, Reshape, Concatenate
from tensorflow.keras.models import Model

def build_generator(z_dimension, img_shape, num_classes):
    noise = Input(shape=(z_dimension,))
    label = Input(shape=(num_classes,))
    x = Concatenate()([noise, label])
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(np.prod(img_shape))(x)
    x = Activation("tanh")(x)
    img = Reshape(img_shape)(x)
    model = Model(inputs=[noise, label], outputs=img)    
    model.summary()
    return model

if __name__ == "__main__":
    z_dimension = 100
    img_shape = (28, 28, 1)
    num_classes = 10
    model = build_generator(z_dimension=z_dimension, img_shape=img_shape, num_classes=num_classes)