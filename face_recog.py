"""
This module is used for feature extraction
"""
from keras_vggface.vggface import VGGFace
import time
from keras import backend as K
from numpy.core.arrayprint import dtype_is_implied
import tensorflow as tf
import cv2
import numpy as np

K.clear_session()
device = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(device[0], True)
tf.config.experimental.set_virtual_device_configuration(device[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

#can select vgg16 resnet50 senet50
vgg_features = VGGFace(model="resnet50", include_top=False, input_shape=(224, 224, 3), pooling='avg')

def init():
    #open the model once for faster inference
    img = np.random.randint(0,255,size=(224,224,3))
    img = np.asarray((img,))
    result = vgg_features.predict(img)


def extract(img):
    """
    preprocess and extract an image to feature

    :param img: image array with shape (224,224,3) or list of images
    :return: numpy array with shape (,2024)
    """
    img = np.asarray(img, dtype=np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(224,224))
    img = np.asarray((img,))
    result = vgg_features.predict(img)
    return result

def recognize(feat, regis_feat, thresh=70):
    """
    find the most match of a feature to all registered feature

    :param feat: extracted feature from vgg_face
    :type feat: numpy array in shape (1,2024)
    :param regis_feat: list of registered features
    :type regis_feat: list
    :param thresh: threshold for comparing distance
    :type thresh: int

    :return: index of regis_feat that has minimum distance. If not found return None
    """
    regis_feat= np.asarray(regis_feat)
    distances = [np.linalg.norm(f) for f in regis_feat-feat]
    #debug
    print(f"distances debug : {distances}")
    if(np.min(distances) <= thresh):
        return np.argmin(distances)
    else:
        return None

if __name__ == "__main__":

    img = cv2.imread("sample_image/earth_regis.jpg")
    result = extract(img)
    print(type(result))
    print(result.shape)
    print(result)

    #img_batch = np.asarray([img,img,img])
    #result_batch = extract(img_batch, batch=True)
    #print(result_batch.shape)
    #print(result_batch)



