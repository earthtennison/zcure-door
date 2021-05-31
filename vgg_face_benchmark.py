from keras_vggface.vggface import VGGFace
import time
from keras import backend as K
import tensorflow as tf
import cv2
import numpy as np

K.clear_session()
device = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(device[0], True)
tf.config.experimental.set_virtual_device_configuration(device[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

#can select vgg16 resnet50 senet50
vgg_features = VGGFace(model="senet50", include_top=False, input_shape=(224, 224, 3), pooling='avg')

img = np.random.randint(0,255,size=(224,224,3))
img = np.asarray((img,))
result = vgg_features.predict(img)

print("open the model")
start = time.time()
for i in range(10):
    result = vgg_features.predict(img)
    end = time.time()
    print(end-start)
    start=end



