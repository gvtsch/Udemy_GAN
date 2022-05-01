import os, matplotlib.pyplot as plt, numpy as np
from random import sample
from zlib import Z_BEST_COMPRESSION
from scipy import rand
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from mnistData import MNIST
from mnistCGANDiscriminator import build_discriminator
from mnistCGANGenerator import build_generator

PATH = os.path.abspath("C:/Selbststudium/Udemy/Udemy_GAN")
IMAGES_PATH = os.path.join(PATH, "Chapter5_GANCode/Chapter5_3_CGAN/images")

class CGAN:
    def __init__(self):
        # Model parameters
        self.img_rows=28
        self.img_cols=28
        self.img_depth=1
        self.img_shape=(self.img_rows, self.img_cols, self.img_depth)
        self.z_dimension=100
        self.num_classes=10
        optimizer=Adam(
            learning_rate=0.0002,
            beta_1=0.5)
        # Discriminator
        self.discriminator=build_discriminator(
            img_shape=self.img_shape, 
            num_classes=self.num_classes)
        self.discriminator.compile(
            loss="binary_crossentropy",
            optimizer=optimizer,
            metrics=["accuracy"])
        # Generator
        self.generator=build_generator(
            z_dimension=self.z_dimension,
            img_shape=self.img_shape,
            num_classes=self.num_classes)
        noise=Input(shape=(self.z_dimension,)) # Input für Generator
        label=Input(shape=(self.num_classes,))
        img=self.generator([noise, label]) # Generator erstellt ein Bild
        self.discriminator.trainable=False # 
        d_pred=self.discriminator([img,label]) # Das generierte Bild als Input für Discriminator
        self.combined=Model(
            inputs=[noise, label],
            outputs=d_pred)
        self.combined.compile(
            loss="binary_crossentropy",
            optimizer=optimizer,
            metrics=[])
        # Der Generator muss mit der Prediction des Discriminators verbunden werden,
        # weil der Generator sonst nicht erfährt, ob das erstellte Bild gut ist

    def train(self, epochs, batch_size, sample_interval):
        # Daten laden und rescalen
        mnist_data=MNIST()
        x_train, y_train=mnist_data.get_train_set()
        x_train=x_train/127.5-1.0 
            # Werte halbieren [0...255] --> [0...2]
            # Mittelwertverschiebung nach 0
            # Werte liegen nun +-1 um 0 --> [-1...1]
        # Adversial ground truths
        real=np.ones(shape=(batch_size,1))
        fake=np.zeros(shape=(batch_size,1))
        # Training starten
        for epoch in range(epochs):
            # Trainingsset
            rand_idxs=np.random.randint(0,x_train.shape[0],batch_size) # batch_size viele Bilder (zufällig 0...60_000 Bilder)
            train_imgs=x_train[rand_idxs]
            train_labels = y_train[rand_idxs]
            # Erstellte Bilder
            noise=np.random.normal(0,1,(batch_size, self.z_dimension)) # mean=0, std=1
            generated_imgs=self.generator([noise, train_labels],training=False)
            # Discriminator trainieren
            d_loss_real=self.discriminator.train_on_batch([train_imgs, train_labels],real)
            d_loss_fake=self.discriminator.train_on_batch([generated_imgs, train_labels],fake)
            d_loss=np.add(d_loss_fake,d_loss_real)*0.5
            # Generator trainieren
            noise=np.random.normal(0,1,(batch_size, self.z_dimension))
            g_loss=self.combined.train_on_batch([noise, train_labels], real)
            # Fortschritt abspeichern
            if (epoch % sample_interval) == 0:
                print(
                    f"{epoch} --- d_loss {round(d_loss[0],4)}"
                    f" --- d_acc {round(d_loss[1],4)}"
                    f" --- g_loss {round(g_loss,4)}")
                self.sample_images(epoch)

    # Bilder speichern
    def sample_images(self,epoch):
        r,c=2,5
        noise=np.random.normal(0,1,(r*c,self.z_dimension))
        labels = np.random.randint(0, self.num_classes, 10)
        labels_categorical = to_categorical(labels, num_classes=self.num_classes)
        gen_imgs=self.generator.predict([noise, labels_categorical])
        gen_imgs=0.5*gen_imgs+0.5
        fig,axs=plt.subplots(r,c)
        cnt=0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:,0],cmap='gray')
                axs[i,j].set_title(f"Digit: {labels[cnt]}")
                axs[i,j].axis('off')
                cnt+=1
        fig.savefig(IMAGES_PATH+"/%d.png"%epoch)
        plt.close()

if __name__ == "__main__":
    cgan = CGAN()
    cgan.train(
        epochs=10_000,
        batch_size=32,
        sample_interval=1_000)