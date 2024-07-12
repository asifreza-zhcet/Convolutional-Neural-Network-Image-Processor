import cv2 as cv
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import tensorflow as tf

model = tf.keras.models.load_model('./model/cnn_model.h5')
def classifier(image):
    image = cv.resize(image,(200,200))
    image = image.reshape(1, 200, 200, 3)
    y_pred = model.predict(image, verbose=False)
    labels = ['Cat', 'Dog', 'Snake']
    y = labels[np.argmax(y_pred)]
    return y

if __name__ == '__main__':
    img = cv.imread(r"C:\Users\rezaa\Desktop\dog.jpg")
    print(classifier(img))


