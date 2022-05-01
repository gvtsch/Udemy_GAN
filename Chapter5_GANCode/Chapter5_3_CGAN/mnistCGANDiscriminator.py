from distutils.command.build import build
from tensorflow.keras.layers import Activation, Dense, Input, LeakyReLU, Flatten, Concatenate, Dropout
from tensorflow.keras.models import Model

def build_discriminator(img_shape, num_classes):
    img = Input(shape=(img_shape))
    label = Input(shape=(num_classes,))
    img_flatten = Flatten()(img)
    x = Concatenate()([img_flatten, label])
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(rate=0.25)(x)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(rate=0.25)(x)
    x = Dense(units=1)(x)
    d_pred = Activation("sigmoid")(x)
    model = Model(inputs=[img, label], outputs=d_pred)
    model.summary()
    return model

if __name__ == "__main__":
    img_shape = (28, 28, 1)
    num_classes = 10
    model = build_discriminator(img_shape=img_shape, num_classes=num_classes)