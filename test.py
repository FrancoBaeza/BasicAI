import cv2
import tensorflow as tf
import numpy as np

# Carga el modelo
model = tf.keras.models.load_model('numbers.h5')

# cargar la imagen
img = cv2.imread('./foto.png')

# Convertir a escala de grises y redimensionar la imagen
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = cv2.resize(img_gray, (28, 28)).astype('float32') / 255.0

# hacer la predicción
prediction = model.predict(img_gray.reshape(1, 784))

# imprimir la predicción
print("La predicción es:", np.argmax(prediction))