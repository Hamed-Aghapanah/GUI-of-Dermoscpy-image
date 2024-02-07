#def deeep(I):
# =============================================================================
# Import libs
# =============================================================================
# from sklearn import cross_validation 
# from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV,cross_validate

import time
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten,MaxPool2D,BatchNormalization,Dropout,concatenate, Conv1D
import os
import scipy.io
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping , ModelCheckpoint
import os
from keras.optimizers import Adam
import scipy.io
from scipy.io import loadmat
from keras import Input,layers
from keras.models import load_model,Model
from keras import regularizers
from sklearn.metrics import classification
from sklearn.utils import class_weight
from keras.applications.resnet50 import ResNet50
import keras.backend as K
import tensorflow as tf
import cv2
import numpy as np 
# =============================================================================
# Initialize
# =============================================================================
path='D://001phd//2OpenCV//project1//'
path=os.path.abspath(__file__)
path=path[0:-26]
print (path)
#    test_path='ISIC_2019_Test_Input/'
#    test_path_1='test//'
img_rows=128
img_cols=128
# =============================================================================
# Functions
# =============================================================================
def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

## =============================================================================
## model 5  ResNet50
## =============================================================================
mobile = ResNet50(include_top=False,input_shape=(img_rows,img_cols,3),weights=None,pooling='avg')
model5 = Sequential()
model5.add(mobile)
model5.add(layers.Dropout(0.5))
model5.add(layers.Dense(9, activation='softmax'))
model5.summary()
model5.compile(optimizer=Adam(lr=2e-5,decay=1e-6),loss='categorical_crossentropy',metrics=['accuracy',f1])
model5.summary()
model5.load_weights(path+'myownmodel_Adamopt_ResNet50.hdf5')
#########################################
import numpy as np
import cv2
X_test = []
Target = []
for index in range(9):
    for i in range(3):
        img =I# cv2.imread(path + 'test//' +str(index+1)+'//'+i)
        img_resized = cv2.resize(img,(128,128),interpolation=cv2.INTER_AREA)
        img_resized = img_resized / 255
        X_test.append(img_resized)
        Target.append(index)
X_test = np.array(X_test)
 
preds5=model5.predict(X_test  , verbose=1)
################## Yes #########
decoded_output5 = np.argmax(preds5,axis=1)

predictions =  decoded_output5[1]+1
#    predictions=decoded_output5
#return ( predictions )     