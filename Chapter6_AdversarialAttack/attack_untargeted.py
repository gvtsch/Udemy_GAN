import os, numpy as np, tensorflow as tf
from tensorflow.keras.optimizers import Adam

from mnistCnn import build_cnn
from mnistData import MNIST
from plotting import plot_attack

PATH = os.path.abspath("C:/Selbststudium/Udemy/Udemy_GAN")
MODELS_PATH = os.path.join(PATH, "models")
CNN_MODEL_PATH = os.path.join(MODELS_PATH, "mnist_cnn_attack.h5")

mnist_data = MNIST()
x_train, y_train = mnist_data.get_train_set()
x_test, y_test = mnist_data.get_test_set()

def adversarial_noise(model, image, label):
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image, training=False)
        prediction = tf.reshape(prediction, (10,))
        loss = loss_object(label, prediction)
    # Gradienten vom Loss hinsichtlich des Input-Bildes ermitteln
    gradient = tape.gradient(loss, image)
    # Vorzeichen für noise holen
    signed_gradient = tf.sign(gradient)
    return signed_gradient

def train_and_save_model():
    model = build_cnn()
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=0.0005),
        metrics=["accuracy"]
    )
    model.fit(
        x=x_train,
        y=y_train,
        verbose=1,
        batch_size=256,
        epochs=10,
        validation_data=(x_test, y_test)
    )
    model.save_weights(filepath=CNN_MODEL_PATH)

def load_model():
    model = build_cnn()
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=0.0005),
        metrics=["accuracy"]
    )
    model.load_weights(filepath=CNN_MODEL_PATH)
    return model

def untargeted_attack(model):
    sample_idx = np.random.randint(low=0, high=x_test.shape[0])
    image = np.array([x_test[sample_idx]]) # Input-Shape für TF anpassen
    true_label = y_test[sample_idx] # OH Vektor
    true_label_idx = np.argmax(true_label)
    y_pred = model.predict(image)[0]
    print(f"--- Vor dem Angriff ---\n"
        f"Wahre Klasse: {true_label_idx}\n"
        f"Wahrscheinlichkeit: {y_pred[true_label_idx]}")
    
    eps = 0.005 # Step-size für den Noise-Filter
    image_adv = tf.convert_to_tensor(image, dtype=tf.float32)
    noise = tf.convert_to_tensor(np.zeros_like(image_adv), dtype=tf.float32)

    while np.argmax(y_pred) == true_label_idx: # So lange noch die korrekte Klasse prädiziert wird, weiter angreifen
        noise = adversarial_noise(model, image_adv, true_label)
        if np.sum(noise) == 0.0:
            break
        image_adv = image_adv + eps * noise
        image_adv = tf.clip_by_value(image_adv, 0.0, 1.0)
        y_pred = model.predict(image_adv)[0]
        print(f"Wahrscheinlichkeit wahre Klasse: {y_pred[true_label_idx]}"
            f"Höchste Wahrscheinlichkeit: {np.max(y_pred)}")

    plot_attack(image, image_adv.numpy())

if __name__=="__main__":
    # train_and_save_model()
    model = load_model()
    untargeted_attack(model)