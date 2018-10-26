
# coding: utf-8

# Import modules

# In[70]:


import numpy as np
import argparse
import os
from skimage import io, color, exposure, transform
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
import sys
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras import callbacks
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard, ReduceLROnPlateau
from sklearn.model_selection import train_test_split


# In[71]:


DEV = False
argvs = sys.argv
argc = len(argvs)


# Setting the epoch and the initial parameters experimentally 

# In[72]:


if argc > 1 and (argvs[1] == "--development" or argvs[1] == "-d"):
  DEV = True

if DEV:
  epochs = 2
else:
  epochs = 20


# In[73]:


"""
Parameters
"""
img_width, img_height = 32, 32
batch_size = 32
samples_per_epoch = 1000
validation_steps = 300
nb_filters1 = 64
conv_size1 = 3
nb_filters2 = 32
conv_size2 = 2
pool_size = 2
classes_num = 5
lr = 0.0004


# Building CNN model

# In[74]:


def build_cnn(input_size, num):
    model = Sequential()
    
    model.add(Conv2D(nb_filters1, conv_size1, conv_size1, border_mode ="same", input_shape=(img_width, img_height, 3)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

    model.add(Conv2D(nb_filters2, conv_size2, conv_size2, border_mode ="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size), dim_ordering='th'))

    model.add(Conv2D(64, (5, 5)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Flatten())
    
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(Activation('relu'))
    
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Activation('relu'))
    
    model.add(Dropout(0.5))
    model.add(Dense(num))
    model.add(Activation("softmax"))
    model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=lr),
              metrics=['accuracy'])

    return model


# Summary of the model

# In[75]:


model = build_cnn(32,5)

print(model.summary())


# In[76]:


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

#Not required ultimately


# Function to load training data. (I divided the initial dataset into 90:10 for training:testing).
# The training data is now further split into training and validation data 90:10

# In[77]:


def load_d(path, val_size=0.1):
    if not os.path.exists(path):
        raise IOError('directory does not exist')
    classes = os.listdir(path)
    indx = {c: i for i, c in enumerate(classes)}
    X = []
    y = []
    for c in classes:
        cp = os.path.join(path, c)
        img_paths = os.listdir(cp)
        label = np.zeros(len(classes), dtype=np.float32)
        label[indx[c]] = 1.0
        for ip in img_paths:
            img = Image.open(os.path.join(cp, ip))
            img = img.resize((32, 32))
            img = np.array(img, dtype=np.float32) / 255.0
            X.append(img)
            y.append(label)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=val_size)
    return X_train, X_test, y_train, y_test, indx

def load_test(path):
    if not os.path.exists(path):
        raise IOError('directory does not exist')
    classes = os.listdir(path)
    indx = {c: i for i, c in enumerate(classes)}
    X = []
    y = []
    for c in classes:
        cp = os.path.join(path, c)
        img_paths = os.listdir(cp)
        label = np.zeros(len(classes), dtype=np.float32)
        label[indx[c]] = 1.0
        for ip in img_paths:
            img = Image.open(os.path.join(cp, ip))
            img = img.resize((32, 32))
            img = np.array(img, dtype=np.float32) / 255.0
            X.append(img)
            y.append(label)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)


    return X,y,indx

X_train, X_test, y_train, y_test, indx = load_d(path="flow/")
print(indx)


# Checkpoints let us save intermediate values, to protect long run values

# In[78]:


img_size = 32
batch_size = 32
epochs = 20

callbacks = [TensorBoard(log_dir='./logs/', batch_size=batch_size),
             ModelCheckpoint(filepath='./checkpoints/{epoch:02d}-{val_acc:.2f}.h5', monitor='val_loss', save_best_only=True),
             ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)]

if not os.path.exists('./checkpoints/'):
    os.mkdir('./checkpoints/')


# Training 

# In[79]:


model.fit(x=X_train, 
          y=y_train,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=callbacks,
          validation_data=(X_test, y_test),
          shuffle=True,
          )


model.save_weights('modelo.h5')


# Load saved model

# In[80]:


model.load_weights('modelo.h5')

indx = {'daisy': 0, 'sunflower': 1, 'tulip': 2, 'dandelion': 3, 'rose': 4}

idx2class = {v: key for key, v in indx.items()}


# Load the test dataset

# In[81]:


X_test,Y_test, indx = load_test("test/")
X_test = X_test.reshape(X_test.shape[0],32,32,3)


# Evaluating accuracy of the model :

# In[82]:


score = model.evaluate(X_test, Y_test, verbose=0)


# In[83]:


print('Test accuracy:', score[1])


# Now we will get labels of the test datset, so that we can compare the model's performance with actual labels

# In[88]:


correct_class = []
for i in range(len(Y_test)):
    for j in range(len(Y_test[i])):
        if(Y_test[i][j] == 1.0):
            correct_class.append(j)


# In[89]:


print(correct_class)


# In[90]:


from matplotlib import pyplot as plt
for i in range(len(y_pred)):
    if(y_pred[i] != correct_class[i]):
        plt.imshow(X_test[i], interpolation='nearest')
        plt.show()
        print('Predicted as->', idx2class[(y_pred[i])])
        print('Correct class->', idx2class[(correct_class[i])])


# Images that were incorrectly classified^^
