import numpy as np
from tensorflow.keras.layers import Activation, BatchNormalization, Dense, Input, LeakyReLU, Reshape
from tensorflow.keras.models import Model, Sequential

def build_generator(z_dimension, img_shape):
    model = Sequential()
    model.add(Dense(256, input_dim=z_dimension))
    model.add(LeakyReLU(alpha=0.2)) # Die wissenschaftlichen Paper empfehlen diese Activation
    model.add(BatchNormalization(momentum=0.8)) # Normalerweise passen die Default-Werte
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(img_shape))) # Output-Shape muss angepasst werden: (28*28*1)
    model.add(Activation("tanh")) # (-1...1)
    model.add(Reshape(target_shape=img_shape))
    model.summary()
    noise = Input(shape=(z_dimension,))
    img = model(noise)
    return Model(inputs=noise, outputs=img)

if __name__ == "__main__":
    z_dimension = 100
    img_shape = (28, 28, 1)
    model = build_generator(z_dimension=z_dimension, img_shape=img_shape)