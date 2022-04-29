from distutils.command.build import build
from tensorflow.keras.layers import Activation, Dense, Input, LeakyReLU, Flatten
from tensorflow.keras.models import Model, Sequential

def build_discriminator(img_shape):
    model = Sequential()
    model.add(Flatten(input_shape=img_shape)) # (28, 28, 1) -> (784) 
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2)) # Alpha bestimmt die Steigung im negativen Bereich
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1)) 
        # Fake oder reales Bild wären eigentlich 2 Outputs, 
        # aber mit Sigmoid wird eine Wahrscheinlichkeit für z.B. Real ausgegeben
    model.add(Activation("sigmoid"))
    model.summary()
    img = Input(shape=img_shape)
    d_pred = model(img)
    return Model(inputs=img, outputs=d_pred)

if __name__ == "__main__":
    img_shape = (28, 28, 1)
    model = build_discriminator(img_shape=img_shape)