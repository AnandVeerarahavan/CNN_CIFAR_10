# Dataset used - CIFAR 10

# Import Libraries

import tensorflow
import numpy as np 
import pandas as pd
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D , MaxPool2D , Flatten , Dense, Dropout
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

# Loading the dataset
(x_train,y_train),(x_test,y_test)=cifar10.load_data()

# Print the shape of X_train
print("==================================================================================")
print()
print("Shape of the training data" , x_train.shape)
print("==================================================================================")
print()
# Naming the classes
classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
print("==================================================================================")
print()
print("The classes present are : 'airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'")
print("==================================================================================")
print()

# Visualizing the images
plt1 = plt.imshow(x_train[5])
print("==================================================================================")
print()
print("Class name - ",classes[y_train[5][0]])
print()
print("==================================================================================")
plt.show()

# Normalising the data
X_train, X_test = x_train/255 , x_test/255

# Steps to be followed
# 1. Architecture
# 2. Compilation
# 3. Fit

# Step 1
# Architecture

# Building the CNN model

model = Sequential()

# Convolution Layer 1
model.add(Conv2D(36, 3, activation='relu', kernel_initializer='he_uniform',))
model.add(MaxPool2D())

# Convolution Layer 2
model.add(Conv2D(72, 3, activation='relu', kernel_initializer='he_uniform',))
model.add(MaxPool2D())

# Convolution Layer 3
model.add(Conv2D(144, 3, activation='relu', kernel_initializer='he_uniform',))
model.add(MaxPool2D())

# Flattening the matrix
model.add(Flatten())

# Building the ANN model

# Hidden Layer 1
model.add(Dense(128, activation='relu',kernel_initializer='he_uniform'))

# Hidden Layer 2
model.add(Dense(64, activation='sigmoid'))

# Hidden Layer 3
model.add(Dense(32, activation='relu',kernel_initializer='he_uniform'))

# Output Layer(ANN)
model.add(Dense(10, activation='softmax'))


# Step 2
# Compile

# Model Compilation
model.compile(optimizer='adam', loss=tensorflow.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

# Step 3
# Fit

# Fitting the model
model.fit(X_train,y_train, epochs=10, batch_size=32)

# Evaluating the model on test data
print("==================================================================================")
print()
print("Model evaluating...")
model.evaluate(X_test, y_test)
print("==================================================================================")
print()
# Prediction variable y_pred
y_pred = model.predict(X_test)

# Predicting the 100 image in the test data
print("==================================================================================")
print()
print("Prediction : 1 - Probabilities", y_pred[100])
print()
print("==================================================================================")

# Recognizing the highest probability and mapping it to the class
print("==================================================================================")
print()
print("Prediction : 1 - Best fit Probability class", classes[np.argmax(y_pred[100])])
print()
# Double checking the test image for users to visualize the correct prediction
plt2 = plt.imshow(X_test[100])
print("==================================================================================")
print()
print("Class Name - ",classes[y_test[100][0]])
print()
plt.show()
print("==================================================================================")


# Predicting the 100 image in the test data
print("==================================================================================")
print()
print("Prediction : 2 - Probabilities", y_pred[200])
print()
print("==================================================================================")

# Recognizing the highest probability and mapping it to the class
print("==================================================================================")
print()
print("Prediction : 2 - Best fit Probability class",classes[np.argmax(y_pred[200])])
print()
print("==================================================================================")

# Double checking the test image for users to visualize the correct prediction
plt3 = plt.imshow(X_test[200])
print("==================================================================================")
print()
print("Class Name - ", classes[y_test[200][0]])
print()
plt.show()
print("==================================================================================")

# Predicting the 100 image in the test data

print("==================================================================================")
print()
print("Prediction : 3 - Probabilities",y_pred[1])
print()
print("==================================================================================")

# Recognizing the highest probability and mapping it to the class
print("==================================================================================")
print()
print("Prediction : 2 - Best fit Probability class",classes[np.argmax(y_pred[1])])
print()
print("==================================================================================")

# Double checking the test image for users to visualize the correct prediction
plt4 = plt.imshow(X_test[1])
print("==================================================================================")
print()
print("Class Name - ", classes[y_test[1][0]])
print()
print("==================================================================================")

plt.show()

# Saving the model
model.save('cifar_10_model.h5')