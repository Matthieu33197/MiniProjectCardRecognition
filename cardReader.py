import os
import numpy as np
import tensorflow as tf
import keras as ks
from tkinter import Tk, Label, Button, filedialog
from PIL import Image, ImageTk
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.models import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Dimensions des images
IMG_SIZE = 224
BATCH_SIZE = 32

# Préparation des générateurs de données
train_dir = './train'
valid_dir = './valid'
test_dir = './test'

train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

model_path = 'card_recognition_model.h5'
if os.path.exists(model_path):
    model = ks.models.load_model(model_path)
    print("Modèle chargé depuis le fichier.")
else:
    # Création du modèle CNN
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(train_generator.num_classes, activation='softmax')
    ])

    # Compilation du modèle
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Entraînement du modèle
    history = model.fit(
        train_generator,
        epochs=10,
        validation_data=valid_generator
    )

    # Enregistrer le modèle
    model.save(model_path)


# Prédiction d'une image sélectionné
def image_normalizer(image_path):
    img = Image.open(image_path).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        print(f"Image sélectionnée : {file_path}")
        preprocessed_image = image_normalizer(file_path)
        prediction = model.predict(preprocessed_image)
        class_index = np.argmax(prediction)
        card_name = list(train_generator.class_indices.keys())[class_index]
        print(f"Prediction : {card_name}")

root = Tk()
root.title("Card Recognition")
root.geometry("300x300")

load_button = Button(root, text="Load Image", command=predict_image)
load_button.pack()


root.mainloop()