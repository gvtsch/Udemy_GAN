from tensorflow.keras.layers import Activation, BatchNormalization, Dense, Input, LeakyReLU, Reshape, Conv2D, UpSampling2D
from tensorflow.keras.models import Model, Sequential

def build_generator(z_dimension, img_shape):
    model = Sequential()
    model.add(Dense(units=128*7*7,input_dim=z_dimension)) # 6272x1
    model.add(LeakyReLU(alpha=.2))
    model.add(Reshape(target_shape=(7,7,128))) # 7x7x128
    model.add(UpSampling2D()) # 14x14x128
    model.add(Conv2D(filters=128,kernel_size=5,strides=1,padding="same",use_bias=False))
        # Ungewöhnlicherweise scheint dieses Netz besser zu werden, wenn das Bias Neuron deaktivert ist
    model.add(BatchNormalization()) # TODO #3
    model.add(LeakyReLU(alpha=.2))
    model.add(UpSampling2D()) # 28x28x128
    model.add(Conv2D(filters=64,kernel_size=5,strides=1,padding="same",use_bias=False)) # 28x28x64
    model.add(BatchNormalization()) # TODO #3
    model.add(LeakyReLU(alpha=.2))
    model.add(Conv2D(filters=img_shape[-1],kernel_size=5,strides=1,padding="same",use_bias=False)) # 28x28x1
    model.add(Activation("tanh")) 
    model.add(Reshape(target_shape=img_shape))
    model.summary()
    noise = Input(shape=(z_dimension,))
    img = model(noise)
    return Model(inputs=noise, outputs=img)

if __name__ == "__main__":
    z_dimension = 100
    img_shape = (28, 28, 1)
    model = build_generator(z_dimension=z_dimension, img_shape=img_shape)