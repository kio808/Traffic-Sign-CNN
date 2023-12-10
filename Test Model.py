from keras.models import load_model
import os
import cv2
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#Used only to get the labels of the cats
data_dir = ('Train 70:30 split')
dataset = keras.utils.image_dataset_from_directory('Train 70:30 split', batch_size = 15, image_size=(32,32), shuffle=True,)
labels = dataset.class_names

#loading model
reload_model = load_model(os.path.join('models',  "3 layer COnv2D 8 sign model SGD padding.keras"))

#testing model
#In order to add your own image, add the image in the folder "add cat images here", and type in the image name and format
testimg = cv2.imread(os.path.join("add image", "stop3.jpeg"))
resize = tf.image.resize(testimg, (32,32))
plt.imshow(cv2.cvtColor(testimg, cv2.COLOR_BGR2RGB))
prediction = reload_model.predict(np.expand_dims(resize/255, 0))
index = np.argmax(prediction)
print(index)
plt.title(str(labels[index]))

plt.show()

print("this is a " + str(labels[index]))