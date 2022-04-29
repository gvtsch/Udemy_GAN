from distutils.command.build import build
from tensorflow.keras.layers import Activation, Dense, Input, LeakyReLU, Flatten, Conv2D, Dropout
from tensorflow.keras.models import Model, Sequential

def build_discriminator(img_shape):
    model = Sequential()
    model.add(Conv2D(filters=64,kernel_size=5,strides=2,padding="same",input_shape=img_shape)) # 28x28x1-->14x14x64 (wegen strides)
        # strides gibt an, um wie viel Pixel wir den Filter verschieben. Def=1
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(rate=0.3))
        # Hilft gegen Overfitting TODO #2 #1
    model.add(Conv2D(filters=128,kernel_size=5,strides=2,padding="same")) # 14x14x64-->7x7x128
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(rate=0.3))
    model.add(Flatten()) #7x7-->49
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    model.summary()
    img = Input(shape=img_shape)
    d_pred = model(img)
    return Model(inputs=img, outputs=d_pred)

if __name__ == "__main__":
    img_shape = (28, 28, 1)
    model = build_discriminator(img_shape=img_shape)