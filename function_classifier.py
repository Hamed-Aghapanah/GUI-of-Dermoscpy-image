#function_classifier
def crop_image(I):
    import cv2 
    import numpy as np


def prediction_deep(I):
    import cv2
    from keras.models import Model
    from keras.layers import Input, Dense
    from keras.models import load_model
    model = load_model('CNN2D_3.h5')
    img_rows=600
    img_cols=450
    input_tensor1 = Input(shape=(img_rows,img_cols,3))
    model.load_weights('CNN2D_3_weights.h5')
    A=cv2.imread('k8.jpg')
#    prediction = model.predict(A)
    prediction=8
    return(prediction)