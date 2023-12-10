#Used for visualization
import matplotlib.pyplot as plt

import os

#Used for neural network
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator

#Used for accessing folders
testdata = ('Test 70:30 split')
traindata = ('Train 70:30 split')

augment = ImageDataGenerator(rescale=1.0/255,
                             rotation_range=40,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2)

test_augment = ImageDataGenerator(rescale=1.0/255)

#Creates batches for training
trainset = augment.flow_from_directory(
    traindata,
    target_size=(32, 32),
    batch_size=5,
    class_mode='sparse',
    shuffle=True)
#Creates batches for testing
testset = test_augment.flow_from_directory(
    testdata,
    target_size=(32, 32),
    batch_size=5,
    class_mode='sparse',
    shuffle=True)

model = Sequential()
#input layer where it creates a kernel to go through the image, and outputs the value in a new feature map, where the value corresponds to a specific filter and represents the response of that filter to the input image
model.add(Conv2D(64, kernel_size=(5,5), strides=(1,1), activation = 'relu', input_shape=(32,32,3), padding = "same"))

# #returns maximum value of the activation
model.add(MaxPooling2D())

#Convolutional layer where it has 64 filters, 3x3 kernel size, 1 pixel stride (how many pixels it moves)
model.add(Conv2D(32, kernel_size=(3,3), strides=(1,1), activation = 'relu', padding = "same"))
model.add(MaxPooling2D())

model.add(Conv2D(32, kernel_size=(3,3), strides=(1,1), activation = 'relu', padding = "same"))
model.add(MaxPooling2D())
#Creates a 1-dimensional vector from our 2-dimensional arrays to help feed it into the fully connected layer
model.add(Flatten())

#All the neurons inside the Dense layer are all connected to all the neurons in the previous layer
model.add(Dense(64, activation='relu'))
#Dropout ignores 20 percent of the neurons in the layer
model.add(Dropout(0.2))

model.add(Dense(32, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(16, activation='relu'))

model.add(Dropout(0.2))

#output layer
model.add(Dense(8, activation='softmax'))


#Uses the Adam optimizer and sparse categorical crossentropy to train the model
model.compile(optimizer = tf.keras.optimizers.legacy.SGD(), loss = 'sparse_categorical_crossentropy', metrics = ["accuracy"])

#Visualizes what layers are in the CNN
model.summary()

#trains model
fit_model = model.fit(trainset, epochs = 50, validation_data=testset)

#plotting loss and accuracy graphs
loss_graph = plt.figure()
plt.plot(fit_model.history['loss'], color='red', label='loss' )
plt.plot(fit_model.history['val_loss'], color='blue', label='val_loss')
loss_graph.suptitle('Loss plot', fontsize=20)
plt.legend(loc='upper left')
plt.show()

acc_graph = plt.figure()
plt.plot(fit_model.history['accuracy'], color='green', label='accuracy' )
plt.plot(fit_model.history['val_accuracy'], color='orange', label='val_accuracy')
acc_graph.suptitle('Accuracy plot', fontsize=20)
plt.legend(loc='upper left')
plt.show()

model.save(os.path.join('models', "3 layer COnv2D 8 sign model SGD padding.keras"))