import os, matplotlib.pyplot as plt, numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from mnistData import MNIST
from mnistGanDiscriminator import build_discriminator
from mnistGanGenerator import build_generator

PATH = os.path.abspath("C:\Selbststudium\Udemy\Udemy_GAN")
IMAGE_PATH = os.path.join(PATH, "Chapter5_GANCode\Chapter5_1_GAN\images")

class GAN:
    pass

if __name__ == "__main__":
    gan = GAN()