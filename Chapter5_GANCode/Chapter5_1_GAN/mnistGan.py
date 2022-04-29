import os, matplotlib.pyplot as plt, numpy as np
from random import sample
from zlib import Z_BEST_COMPRESSION
from scipy import rand
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from mnistData import MNIST
from mnistGanDiscriminator import build_discriminator
from mnistGanGenerator import build_generator

PATH = os.path.abspath("C:\Selbststudium\Udemy\Udemy_GAN")
IMAGES_PATH = os.path.join(PATH, "Chapter5_GANCode\Chapter5_1_GAN\images")

class GAN:
    def __init__(self):
        # Model parameters
        self.img_rows=28
        self.img_cols=28
        self.img_depth=1
        self.img_shape=(self.img_rows, self.img_cols, self.img_depth)
        self.z_dimension=100
        optimizer=Adam(
            learning_rate=0.0002,
            beta_1=0.5)
        # Discriminator
        self.discriminator=build_discriminator(img_shape=self.img_shape)
        self.discriminator.compile(
            loss="binary_crossentropy",
            optimizer=optimizer,
            metrics=["accuracy"])
        # Generator
        self.generator=build_generator(
            z_dimension=self.z_dimension,
            img_shape=self.img_shape)
        z=Input(shape=(self.z_dimension,)) # Input für Generator
        img=self.generator(z) # Generator erstellt ein Bild
        self.discriminator.trainable=False # 
        d_pred=self.discriminator(img) # Das generierte Bild als Input für Discriminator
        self.combined=Model(
            inputs=z,
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
        x_train,_=mnist_data.get_train_set()
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
            # Erstellte Bilder
            noise=np.random.normal(0,1,batch_size=self.z_dimension) # mean=0, std=1
            generated_imgs=self.generator(noise,training=False)
            # Discriminator trainieren
            d_loss_real=self.discriminator((train_imgs,real),training=True)
            d_loss_fake=self.discriminator((generated_imgs,fake),training=True)
            d_loss=np.add(d_loss_fake,d_loss_real)*0.5
            # Generator trainieren
            noise=np.random.normal(0,1,batch_size=self.z_dimension)
            g_loss=self.combined((noise,real))
            # Fortschritt abspeichern
            if (epoch % sample_interval == 0):
                print(
                    f"{epoch} --- d_loss {round(d_loss[0],4)}"
                    f" --- d_acc {round(d_loss[1],4)}"
                    f" --- g_loss {round(g_loss,4)}")

    # Bilder speichern
    def sample_images(self,epoch):
        r,c=5,5
        noise=np.random.normal(0,1,(r*c,self.z_dimension))
        gen_imgs=self.generator.predict(noise)
        gen_imgs=.5*gen_imgs+.5
        fig,axs=plt.subplots(r,c)
        cnt=0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:0],cmap="grey")
                axs[i,j].axis("off")
                cnt+=1
        fig.savefig(IMAGES_PATH+"/%d.png"%epoch)
        plt.close()

if __name__ == "__main__":
    gan = GAN()
    gan.train(
        epochs=10_000,
        batch_size=32,
        sample_interval=1_000)