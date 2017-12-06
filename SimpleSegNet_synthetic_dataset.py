
# coding: utf-8

# # Simple implementation of Simple Traffic Light Segmentation Neural Net
# 
# 

# # Datasets
# 
# I will create a dataset of generic shapes and train the neural net to segment the images first by colour and then by shapes.
# 

# ## Colour data set

# In[1]:


import numpy as np
import pandas as pd
import re


# Show images
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import glob

import random


# In[2]:


class TrainingImage:
    """ 
    This class handles the creation of training
    images and the associated label image
    """
    #Colors classes    
    BLACK_CLASS = 0
    WHITE_CLASS = 1
    RED_CLASS = 2
    LIME_CLASS = 3
    BLUE_CLASS = 4
    YELLOW_CLASS = 5
    CYAN_CLASS = 6
    MAGENTA_CLASS = 7
    SILVER_CLASS = 8
    GRAY_CLASS = 9
    MAROON_CLASS = 10
    OLIVE_CLASS = 11
    GREEN_CLASS = 12
    PURPLE_CLASS = 13
    TEAL_CLASS = 14
    NAVY_CLASS = 15
    
    COLOR = {
        BLACK_CLASS : (0,0,0),
        WHITE_CLASS : (255,255,255),
        RED_CLASS : (255,0,0),
#         LIME_CLASS : (0,255,0),
#         BLUE_CLASS : (0,0,255),
#         YELLOW_CLASS : (255,255,0),
#         CYAN_CLASS : (0,255,255),
#         MAGENTA_CLASS : (255,0,255),
#         SILVER_CLASS : (192,192,192),
#         GRAY_CLASS : (128,128,128),
#         MAROON_CLASS : (128,0,0),
#         OLIVE_CLASS : (128,128,0),
#         GREEN_CLASS : (0,128,0),
#         PURPLE_CLASS : (128,0,128),
#         TEAL_CLASS : (0,128,128),
#         NAVY_CLASS : (0,0,128),
    }
    
    def __init__(self, size=(224,224, 3), backgd_color_class=BLACK_CLASS):
        self.size = size
        self.createImageBackground(backgd_color_class)
        
    def clear(self):
        self.image = np.zeros(self.size, dtype=np.uint8)
        self.label = np.zeros(self.size[0:2] + (len(TrainingImage.COLOR.keys()), ) , dtype=np.uint8)
        
    def createImageBackground(self, backgd_color_class):
        self.backgd_color_class = backgd_color_class
        self.clear()
        for channel in range(0, 3):
            self.image[:,:,channel] = TrainingImage.COLOR[self.backgd_color_class][channel]
        self.label[:,:, self.backgd_color_class] = 1
    
    def createRectangle(self, pt1, pt2, color_class=WHITE_CLASS):
        cv2.rectangle(self.image, 
                      pt1[::-1], pt2[::-1], TrainingImage.COLOR[color_class], thickness=cv2.FILLED)
        self.label[pt1[0]:pt2[0],pt1[1]:pt2[1], :] =  0 # Clear exisiting labels in target position
        self.label[pt1[0]:pt2[0],pt1[1]:pt2[1], color_class] =  1
        
    def addRandRectangle(self, color_class=WHITE_CLASS):
        image_size = self.image.shape
        pt1 = (random.randint(0, 150), random.randint(0, 100))
        pt2 = (random.randint(pt1[0]+50, image_size[0]-1), random.randint(pt1[1]+50, image_size[1]-1))
        self.createRectangle( pt1, pt2, color_class=color_class)

image_size = (224,224, 3)
trainim = TrainingImage(size=image_size , backgd_color_class=TrainingImage.BLACK_CLASS)
trainim.createRectangle( (50,150) , (150,200), color_class=TrainingImage.RED_CLASS)
trainim.addRandRectangle(TrainingImage.WHITE_CLASS)

# Plot background 
plt.imshow(trainim.image)
plt.title('image')
#plt.show()

# Plot background 
#plt.imshow(trainim.label[:,:,11:14])
plt.imshow(trainim.label.argmax(axis=2), cmap='hot')
plt.title('label')
#plt.show()

print('Image size {}'.format(trainim.image.shape))


# In[3]:
# Generate color training dataset
from functools import reduce


NUM_TRAIN = 512
num_classes = len(TrainingImage.COLOR.keys())
image_size = (224,224, 3)
label_size = (224,224, num_classes)
size_in_memory = 1
image_mem = reduce(lambda a,b : a*b ,image_size, 1)
label_mem = reduce(lambda a,b : a*b ,label_size, 1)
total_train_mem = NUM_TRAIN * (image_mem + label_mem)
print('Total size of the use of memory : {}MB'.format(total_train_mem/(1024*1024)))

# Synthetic data generator
def generateRectTrainData(X, y, num_samples):
    trainim = TrainingImage(size=image_size)
    for i in range(num_samples):
        trainim.createImageBackground(TrainingImage.BLACK_CLASS)
        trainim.addRandRectangle(color_class=random.randint(1, num_classes-1))

        X[i] = trainim.image
        y[i] = trainim.label
    return X, y

X_TRAIN_SIZE = (NUM_TRAIN,) + image_size
Y_TRAIN_SIZE = (NUM_TRAIN,) + image_size[0:2] + (num_classes, )
X_train = np.zeros(X_TRAIN_SIZE, dtype=np.uint8)
y_train = np.zeros(Y_TRAIN_SIZE, dtype=np.uint8)
X_train, y_train = generateRectTrainData(X_train, y_train, NUM_TRAIN)

NUM_VAL = 64
X_VAL_SIZE = (NUM_VAL,) + image_size
Y_VAL_SIZE = (NUM_VAL,) + image_size[0:2] + (num_classes, )
X_val = np.zeros(X_VAL_SIZE, dtype=np.uint8)
y_val = np.zeros(Y_VAL_SIZE, dtype=np.uint8)
X_val, y_val = generateRectTrainData(X_val, y_val, NUM_VAL)

# In[4]:


# Plot color evaluation dataset

for idx in range(3,7):

    fig = plt.figure(1)
    ax = fig.add_subplot(1,2,1)
    imsh = ax.imshow( X_train[idx], cmap='magma')

    ax = fig.add_subplot(1,2,2)
    imsh = ax.imshow( y_train[idx,:,:,0], cmap='magma')
    plt.colorbar(imsh)
    #plt.show()


# In[ ]:
# # Model
# The original paper uses VGG for implementing the detector.
# 
# 
# TODO : Will try experimenting with Xception and MobileNet  
# https://keras.io/applications/#usage-examples-for-image-classification-models
# 

# In[5]:


# Extract features from an arbitrary intermediate layer with VGG19

from segmodel import SegModel

# Extract pooling layers out of VGG-16
num_classes = len(TrainingImage.COLOR.keys())

segModel = SegModel(num_classes) 
model = segModel.getModel()

from keras import optimizers
optimizer_selected = optimizers.Adam(lr=1e-3)

#model.compile(optimizer=optimizer_selected, loss='categorical_crossentropy')
my_loss={'final_merge': segModel.dice_coef_loss, 'soft_out': 'categorical_crossentropy'}
my_loss_weights={'final_merge': 1., 'soft_out': 0.1}
model.compile(optimizer=optimizer_selected, loss=my_loss, loss_weights = my_loss_weights , 
              metrics={'final_merge':segModel.dice_coef, 'soft_out':'categorical_accuracy'} )

    
history = model.fit(x=X_train, y=[y_train, y_train], batch_size=20, epochs=10, 
                    validation_split=0.0, validation_data=(X_val, [y_val, y_val]), shuffle=True,
                    steps_per_epoch=None, validation_steps=None)


# In[ ]:


import time
# Plot loss function
plt.plot(history.epoch,history.history['loss'])
plt.legend(('loss'))
plt.grid('on')
#plt.show()

#img = X_train[10]

# Create a brand-new image for testing
trainim = TrainingImage(size=image_size , backgd_color_class=TrainingImage.BLACK_CLASS)
trainim.createRectangle( (50,150) , (150,200), color_class=TrainingImage.RED_CLASS)
trainim.addRandRectangle(TrainingImage.WHITE_CLASS)

img = trainim.image


start = time.time()
y_out = model.predict(img.reshape(1,224,224,3))
end = time.time()
print('Time for inference : {}ms'.format((end - start)*1000))


seg = y_out[1]
seg = seg.reshape(224,224, num_classes)

seg_classes=seg.argmax(axis=2)

plt.imshow(img)
plt.show()

fig = plt.figure(1)
fig.clf()
ax = fig.add_subplot(2,2,1)
imsh = ax.imshow( seg[:,:,0], cmap='magma')
fig.colorbar(imsh)

ax = fig.add_subplot(2,2,2)
imsh = ax.imshow( seg[:,:,1], cmap='magma')
fig.colorbar(imsh)

ax = fig.add_subplot(2,2,3)
imsh = ax.imshow( seg[:,:,2], cmap='magma')
fig.colorbar(imsh)

ax = fig.add_subplot(2,2,4)
imsh = ax.imshow( seg_classes, cmap='magma' )
fig.colorbar(imsh)
plt.show()



# In[ ]:





# In[ ]:




