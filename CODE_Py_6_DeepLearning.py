
### DEEP LEARNING ###
https://keras.io/
# Lots of good documentations here.

# REMEMBER:
# 1 - Scale the data!!!

# 0) PRE PROCESSING

# KERAS MAIN ARCHITECTURE: Create sequential empty object, then add layers.
# 1) REGRESSION KERAS
# 2) CLASSIFICATION KERAS
# 3) SAVING/LOADING MODEL FOR PREDICTIONS
# 4) SGD: Stochastic Gradient Descent w/Optimizer
# 5) VALIDATION SPLITTING
#		- Early Stopping		
# 6) CONVOLUTIONAL NETWORK 
# 7) RECURRENT NETWORK - RNN

# Activations: 'relu','leakyRelu','linear','softmax','softplus','tanh','sigmoid'
# Optimizer: 'adam','sgd'
# Loss: 'mean_squared_error','mse','categorical crossentropy','binary cross entropy'
# Metrics : Needs to take (y_pred & y_true)

# Check here for the rest https://keras.io/

#####################################################
################# PRE PROCESSING







#####################################################

# 1) REGRESSION Build Architecture
import numpy as np
import keras
from keras.layers import Dense
from keras.models import Sequential

predictors = np.loadtxt('predictor_data.csv', delimiter = ',')
n_cols = predictors.shape[1]
model = Sequential()

model.add(Dense(100, activation='relu', input_shape = (n_cols,))) # input layer, first hidden layer has 100 nodes
model.add(Dense(100, activation='relu'))
model.add(Dense(1))

# Compile Model
model.compile(optimizer='adam', loss='mean_squared_error') #adam is like a smart optimizer rate


# Fit the Model
model.fit(predictors, target, 
	epochs = 50,
	shuffle = True,
	verbose = 2)

# I BUILT A LINEAR NETWORK FROM END TO END IN JUPYTER NOTEBOOKS > PROJECTS > LYNDA KERAS

#####################################################

# 2) CLASSIFICATION Build Architecture
import numpy as np
import keras
from keras.layers import Dense
from keras.models import Sequential

from keras.utils import to_categorical
data = pd.read_csv('basketball_data.csv')
predictors = data.drop(['shot_result'], axis = 1).as_matrix() # REMOVES the target column
target = to_categorical(data.shot_results)
model = Sequential()

model.add(Dense(100, activation='relu', input_shape = (n_cols, )))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(optimzer = 'sgd', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(predictors, target,
	epochs = 50,
	shuffle = True)
predictions = model.predict(pred_data)

#####################################################

# 3) SAVING MODEL FOR PREDICTIONS

from keras.models import load_model
model.save('model_file.h5') # .h5 is the convention to use for models
my_model = load_model('my_model.h5')
predictions = my_model.predict(data_to_predict_with)
probability_true = predictions[:, 1] # Saves where predictions are TRUE for classification

#####################################################

# 4) SGD: Stochastic Gradient Descent
# We're using multiple learning rates for optimization too!!!
import numpy as np
import keras
from keras.layers import Dense
from keras.models import Sequential

from keras.optimizers import SGD

def get_new_model(input_shape = input_shape):
    model = Sequential()
    model.add(Dense(100, activation='relu', input_shape = input_shape))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    return(model)
lr_to_test = [0.000001, 0.01, 1]
for lr in lr_to_test:
    model = get_new_model()
    my_optimizer = SGD(lr = lr)
    model.compile(optimzer = my_optimizer, loss = 'categorical_crossentropy')
    model.fit(predictors, target,
    	epochs = 50,
    	shuffle = True)

#####################################################

# 5) VALIDATION SPLITTING
model.compile(optimzer = adam, loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(predictors, target, validation_split = 0.3, epochs = 20)

# EARLY STOPPING
from keras.callbacks import EarlyStopping

early_stopping_monitor = EarlyStopping(patience = 2) # patience means it will wait for 2 fails of lower loss score

model.fit(predictors, target, 
	validation_split = 0.3, 
	epochs = 20, 
	callbacks = [early_stopping_monitor])
# callbacks takes a list!
# Now we can set a high maximum epochs since we have early stopping


#####################################################

# 6) CONVOLUTIONAL NETWORK 

# Import matplotlib
import matplotlib.pyplot as plt
# Load the image
data = plt.imread('bricks.png')
# Display the image
plt.imshow(data)
plt.show()

# Changing an Image...
# Set the red channel in this part of the image to 1
data[:10, :10, 0] = 1
# Set the green channel in this part of the image to 0
data[:10, :10, 1] = 0
# Set the blue channel in this part of the image to 0
data[:10, :10, 2] = 0
# Visualize the result
plt.imshow(data)
plt.show()

# One Hot Encoding 
# The number of image categories
n_categories = 3
# The unique values of categories in the data
categories = np.array(["shirt", "dress", "shoe"])
# Initialize ohe_labels as all zeros
ohe_labels = np.zeros((len(labels), n_categories))
# Loop over the labels
for ii in range(len(labels)):
    # Find the location of this label in the categories variable
    jj = np.where(categories == labels[ii])
    # Set the corresponding zero to one
    ohe_labels[ii, jj] = 1

# Calculate the number of correct predictions
number_correct = (test_labels * predictions).sum()
print(number_correct)
# Calculate the proportion of correct predictions
proportion_correct = number_correct/len(predictions)
print(proportion_correct)


# DENSELY CONNECTEd CONVOLUTIONAL NETWORK
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

train_data.shape
(50, 28, 28, 1) # 50 samples, with 28 by 28 pixels, 1 being black and white channel only
train_data = train_data.reshape((50, 784)) # Convert the images into a 2 D table

model.add(Dense(10, activation='relu', input_shape=(784, ))) # 28 * 28 = 784
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(optimzer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_data, train_labels, validation_split=0.2, epochs=3)

# Run evaluation on Test Set
test_data = test_data.reshape((10, 784))
model.evaluate(test_data, test_labels)


``````

# Separately, Here is a CONVOLUTION LAYER
from keras.layers import Conv2d
Conv2d(10, kernel_size=3, activation='relu') # Not dense layer--


``````
# CONVOLUTIONAL LAYERS

# Separately, here is a CONVOLUTIONAL network NOT DENSELY connected
# Convolution layers are for "image processing" while fully connected layers are for "readout"

# Convolution network
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
model = Sequential()

train_data.shape # We DO NOT reshape with NON-dense connected CNNs as we want the pixel's spatial relationships to be retained
(50, 28, 28, 1)

model.add(Conv2D(10, kernel_size=3, activation='relu', 
              input_shape=(img_rows, img_cols, 1)))
model.add(Flatten()) # Flattens feature map to 1d array; Flattens are connectors between CONVOLUTIONS and DENSE layers
model.add(Dense(3, activation='softmax')) # 3 classes of clothing to classify from 

model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

model.fit(train_data, train_labels, validation_split=0.2, epochs=3, batch_size=10)
model.evaluate(test_data, test_labels, epochs=3, batch_size=10)


``````

# TWEAKING CONVOLUTIONS 


# PADDING: GOOD FOR NETWORKS WITH MANY LAYERS; OTHERWISE MIGHT LOSE INFO
# Zero padding
model.add(Conv2D(10, kernel_size=3, activation='relu', 
                 input_shape=(img_rows, img_cols, 1)),
                 padding='valid') # PADDING valid is also default behavior

model.add(Conv2D(10, kernel_size=3, activation='relu', 
                 input_shape=(img_rows, img_cols, 1)),
                 padding='same') # PADDING SAME is such that output of Convolution is the same size as the input


# STRIDES: How far the convolutions jump from each strides

model.add(Conv2D(10, kernel_size=3, activation='relu', 
                 input_shape=(img_rows, img_cols, 1)),
                 strides=1) # Default is 1!!!!!
# More than 1 is changing the stride. Makes output smaller!!!!


# DILATED CONVOLUTION: Putting spaces between the pixels being read

model.add(Conv2D(10, kernel_size=3, activation='relu', 
                 input_shape=(img_rows, img_cols, 1)),
                 dilation_rate=2)

`````

# DEEP CONVOLUTIONAL NETWORKS: Extra convolutional layers

model = Sequential()
model.add(Conv2D(10, kernel_size=2, activation='relu', 
                 input_shape=(img_rows, img_cols, 1), 
                 padding='same'))
# Second convolutional layer
model.add(Conv2D(10, kernel_size=2, activation='relu'))
model.add(Flatten())
model.add(Dense(3, activation='softmax'))

mode.summary()


`````

# POOLING : summarizing a group of pixels by an aggregated value (maybe max value, or mean, or min pooling)
# Pooling DRASTICALLY reduces the number of parameters in the model!

# Max Pooling Loop
for ii in range(result.shape[0]):
    for jj in range(result.shape[1]):
        result[ii, jj] = np.max(im[ii*2:ii*2+2, jj*2:jj*2+2])

# Max Pooling in Keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D

model = Sequential()
model.add(Conv2D(5, kernel_size=3, activation='relu', 
              input_shape=(img_rows, img_cols, 1)))
model.add(MaxPool2D(2)) # ADD POOLING LAYER AFTER EACH CONVOLUTIONAL LAYER

model.add(Conv2D(15, kernel_size=3, activation='relu', 
              input_shape=(img_rows, img_cols, 1)))
model.add(MaxPool2D(2))

model.add(Flatten())
model.add(Dense(3, activation='softmax'))


`````

# Tracking Learning

training = model.fit(train_data, train_labels, 
                     epochs=3, validation_split=0.2)
import matplotlib.pyplot as plt
plt.plot(training.history['loss'])
plt.plot(training.history['val_loss'])
plt.show()

from keras.callbacks import ModelCheckpoint
# This checkpoint object will store the model parameters 
# in the file "weights.hdf5"
checkpoint = ModelCheckpoint('weights.hdf5', monitor='val_loss', 
                             save_best_only=True)
# Store in a list to be used during training
callbacks_list = [checkpoint]
# Fit the model on a training set, using the checkpoint as a callback
model.fit(train_data, train_labels, validation_split=0.2, epochs=3,         
          callbacks=callbacks_list)


````` 

# REGULARIZATION: Preventing overfitting by dropout
# Dropout is choosing a random set of layers and ignoring it in the network

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout
model = Sequential()
model.add(Conv2D(5, kernel_size=3, activation='relu', 
              input_shape=(img_rows, img_cols, 1)))
model.add(Dropout(0.25)) # Add a drop out layer AFTER the layer that we want units ignored
# Choose proportion of units to drop out
model.add(Conv2D(15, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(3, activation='softmax'))


# Batch Normalization: Takes the output of a particular layer and rescales it for mean 0 and sd of 1
# Computationally expensive tho!
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization
model = Sequential()
model.add(Conv2D(5, kernel_size=3, activation='relu', 
              input_shape=(img_rows, img_cols, 1)))
model.add(BatchNormalization())
model.add(Conv2D(15, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(3, activation='softmax'))

# GENERALLY, DO NOT USE DROPOUT AND BATCH NORMALIZATION TOGETHER!!!


`````

# Interpreting the Model

# Selecting particular layers in the network
model.layers

# If we want to look at the first convolutional layer
conv1 = model.layers[0]
weights1 = conv1.get_weights()
len(weights1)
kernels1 = weights[0]
kernels1.shape
(3, 3, 1, 5) # First 2 dimensions are the kernel size. 3rd dim is the number of channels. Last is the number of kernels in this layer.
# Pulling First kernel in the layer
kernel1_1 = kernels1[:, :, 0, 0]
kernel1_1.shape

# We can then visualize this kernel directly
plt.imshow(kernel1_1) # Not super helpful individually, but if we look at a test image with it, we may be able to figure something out
# Pick an image
test_image = test_data[3, :, :, 0]
plt.imshow(test_image)

filtered_image = convolution(test_image, kernel1_1)
plt.imshow(filtered_image)






#####################################################