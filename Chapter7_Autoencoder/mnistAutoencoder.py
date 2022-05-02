from distutils.command.build import build
import os, numpy as np, matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Dense, Flatten, Input, Reshape

from mnistData import MNIST

PATH = os.path.abspath("C:/Selbststudium/Udemy/Udemy_GAN")
IMAGES_PATH = os.path.join(PATH, "Chapter7_Autoencoder/images")

mnist_data = MNIST()
x_train, _ = mnist_data.get_train_set()
x_test, _ = mnist_data.get_test_set()

def build_autoencoder():
    pass

def plot_imgs(test_imgs, decoded_imgs):
    pass

def run_autoencoder(model):
    pass

if __name__ == "__main__":
    model = build_autoencoder()
    test_imgs, decoded_imgs = run_autoencoder(model)
    plot_imgs(test_imgs, decoded_imgs)